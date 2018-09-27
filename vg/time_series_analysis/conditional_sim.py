from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import map
from builtins import range
from builtins import object
from past.utils import old_div
from collections import namedtuple

import numpy as np
try:
    import picos
except ImportError:
    print("PICOS not found.")
from scipy import optimize
from tqdm import tqdm

import vg.helpers as my
from vg.time_series_analysis import spectral, time_series

InterimFactory = namedtuple("interim", ("x", "sol", "norm_inner",
                                        "interpolator",
                                        "alpha",
                                        "realizations_compl",
                                        "success"))


def simulate(autocov, T, cov=None, cond_eq=None, cond_mean=None,
             pool_size=None, verbose=False):
    """Simulate a 1d time series with given autocovariance and honouring
    conditioning values at the conditioning indices.

    Parameter
    ---------
    autocov : callable or sequence of callables
        Autocovariance model returning a covariance for a given distance. For
        multivariate simulation, a sequence of length K of callables is needed.
    T : int
        Number of timesteps to simulate.
    cov : (K, K) ndarray, optional
        Covariance matrix. If None, simulation will be univariate.
    cond_eq : sequence containing an int and a float sequence, optional
        Equality conditioning indices and values.
        First sequence contains the indices with elements between
        0 < t < T (or 2d indices with 0 < k < K, 0 < t < T).
        Second sequence contains the conditioning values.
    cond_mean : sequence of nested ints and floats, optional
        Mean conditioning indices and values.
        First sequence contains sequences of indices with elements between
        0 < t < T (or 2d indices with 0 < k < K, 0 < t < T).
        Second sequence contains the conditioning values.
    verbose : boolean, optional
    """
    if cond_eq is None:
        cond_eq_ii, cond_eq_vals = (), ()
    else:
        cond_eq_ii, cond_eq_vals = cond_eq
    if cond_mean is None:
        cond_mean_ii, cond_mean_vals = (), ()
    else:
        cond_mean_ii, cond_mean_vals = cond_mean
    # it is easier to stay flexible with a 2d shape of conditions, even if we
    # simulate univariate time series
    cond_eq_vals, cond_mean_vals = list(map(np.atleast_2d, (cond_eq_vals,
                                                       cond_mean_vals)))
    total_conds = sum(map(np.size, (cond_eq_ii, cond_mean_vals)))
    n_extra_realizations = 2 * total_conds
    n_realizations = total_conds
    if cov is None:
        spec = spectral.Spectral(autocov, T, pool_size=pool_size)
    else:
        spec = spectral.MultiSpectral(autocov, cov, T, pool_size=pool_size)
    realizations = spec.sim_n(n_realizations)

    a_norm = 1.1
    while a_norm > 1.:
        # add more realizations
        # if we are far away from the sphere, add more (we might be
        # doing this for a while...)
        n_extra = int(a_norm * n_extra_realizations)
        n_realizations += n_extra
        realizations = np.concatenate((realizations, spec.sim_n(n_extra)))

        prob = picos.Problem()
        # set up objective function
        uu = prob.add_variable("u", n_realizations)
        prob.set_objective("min", abs(uu) ** 2)

        if cond_eq is not None:
            # add equality constraints
            # list comprehension because it can deal with 1d and 2d
            y_cond_values = np.array([realization[cond_eq_ii]
                                      for realization in realizations]).T
            eq = picos.new_param("eq", y_cond_values)
            prob.add_constraint(eq * uu == cond_eq_vals)

        if cond_mean is not None:
            # add mean constraints
            # the list comprehension acrobatics are done here because the
            # tuples in cond_mean_ii might not be of equal length (conditioned
            # regions are allowed to have different sizes). This prevents more
            # numpyesque approaches.
            y_cond_values = np.array([[rea[ij].mean()
                                       for rea in realizations]
                                      for ij in cond_mean_ii])
            y_cond_values = y_cond_values.reshape(cond_mean_vals.shape[1],
                                                  n_realizations)
            mean = picos.new_param("mean", y_cond_values)
            prob.add_constraint(mean * uu == cond_mean_vals)

        prob.solve(solver="cvxopt", verbose=int(verbose))
        # retrieve solution as array
        uu_arr = np.array(uu.value)[:, 0]
        # are we inside the sphere?
        a_norm = np.dot(uu_arr, uu_arr)
        if verbose:
            print("n_conditions: %d, n_realizations: %d, a_norm = %.3f" %
                  (total_conds, n_realizations, a_norm))

    # find solution outside the hypersphere by pushing one weight out of it
    prob.add_constraint(uu[0] == 1.5)
    prob.solve(solver="cvxopt", verbose=0)
    vv_arr = np.array(uu.value)[:, 0]

    # somewhere between uu and vv lies the surface of the hypersphere. find it!
    def root_func(t):
        xx = uu_arr + t * (vv_arr - uu_arr)
        return np.dot(xx, xx) - 1
    t = optimize.brentq(root_func, 0., 1.)

    # construct the resulting time series
    return np.tensordot(uu_arr + t * (vv_arr - uu_arr), realizations, axes=1)


class SimulateMulti(object):
    def __init__(self, T, autocovs, cov, pool_size=None, verbose=False,
                 sum_vars=None):
        """Simulate a multivariate time series using Random Mixing with
        Singular Value Decomposition.

        Parameter
        ---------
        T : int
            Number of simulated timesteps
        autocov : callable or sequence of callables
            Autocovariance model returning a covariance for a given time lag.
        cov : (K, K) array, optional
            Covariance of the disaggregated time series.
        pool_size : None or int, optional
            Do not draw fresh random fields, but use a pool of this size to
            sample from.
        sum_vars : int or sequence of ints
            Disaggregate these variables according to sum (instead of mean).
        verbose : boolean, optional
        """
        self.autocovs = autocovs
        self.cov = cov
        self.K = cov.shape[0]

        self.agg_funcs = self.K * [np.mean]
        if isinstance(sum_vars, int):
            sum_vars = [sum_vars]
        if sum_vars is not None:
            for sum_var in sum_vars:
                self.agg_funcs[sum_var] = np.sum

        self.pool_size = pool_size
        self.verbose = verbose
        # to be filled by self.cond_eq and self.cond_mean
        self.conds_names = ("conds_eq_%s",
                            "conds_mean_%s",
                            "conds_trans_mean_%s")
        self.conds_values_names = tuple([name % "values"
                                         for name in self.conds_names])
        self.conds_ij_names = tuple([name % "ij"
                                     for name in self.conds_names])
        self._T_original = T
        # for T property
        self._T = T
        self.reset()
        # are we doing nonlinear means? will be set in self.cond_trans_means
        self.mean_trans = self.t_kwds = None
        # we want to maintain the pool in subsequent self.run calls
        self.n_realizations = None

        if self.verbose:
            print("Generating unconditional time series.")

        self.spec = spectral.MultiSpectral(autocovs, cov, T,
                                           pool_size=pool_size,
                                           verbose=verbose)

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        # if value > self.T:
        #     self.spec.T = spectral.MultiSpectral(autocovs, cov, value,
        #                                        pool_size=pool_size,
        #                                        verbose=verbose)
        #     # T_extra = value - self.T
        #     # spec_extra = spectral.MultiSpectral(self.autocovs,
        #     #                                     self.cov, T_extra,
        #     #                                     pool_size=self.pool_size,
        #     #                                     verbose=self.verbose)
        #     # self.spec = np.concatenate((self.spec, spec_extra))
        if value != self.T:
            self.spec.T = value
        self._T = value

    def cond_eq(self, ij, values):
        """Add equality constraints."""
        self.conds_eq_ij = np.asarray(ij)
        if self.conds_eq_ij.ndim == 2:
            self.conds_eq_ij = self.conds_eq_ij[None, :]
        self.conds_eq_values = values

    def cond_mean(self, ij, values):
        """Add mean constraint."""
        # nans in values are interpreted as unconditioned
        non_nan_mask = ~np.isnan(values)
        self.conds_mean_ij = np.asarray(ij)[non_nan_mask]
        if np.ndim(self.conds_mean_ij) == 2:
            self.conds_mean_ij = self.conds_mean_ij[None, :]
        self.conds_mean_values = values[non_nan_mask]

    def cond_trans_mean(self, ij, values, trans, t_kwds=None):
        """Add a non-linear mean constraint with a transformation function.
        """
        # nans in values are interpreted as unconditioned
        non_nan_mask = ~np.isnan(values)
        self.conds_trans_mean_ij = np.asarray(ij)[non_nan_mask]
        if np.ndim(self.conds_trans_mean_ij) == 2:
            self.conds_trans_mean_ij = self.conds_trans_mean_ij[None, :]
        self.conds_trans_mean_values = values[non_nan_mask]
        if np.ndim(self.conds_trans_mean_values) == 1:
            self.conds_trans_mean_values = \
                self.conds_trans_mean_values[:, None]
        self.mean_trans = trans
        if "doys" not in t_kwds:
            t_kwds["doys"] = None
        else:
            self.T = len(t_kwds["doys"])
        self.t_kwds = t_kwds

    def reset(self):
        for name in self.conds_names:
            setattr(self, name % "values", None)
            setattr(self, name % "ij", None)
        self.T = self._T_original

    @property
    def cv_total(self):
        """Returns all conditioning values."""
        vals = [getattr(self, attr_name)
                for attr_name in self.conds_values_names
                if (getattr(self, attr_name) is not None
                    and "trans" not in attr_name)]
        return np.squeeze(np.concatenate(vals))

    def gen_A(self, realizations):
        """This supplies the left-hand side of the equation system, which
        consists of the values of the unconditioned realizations at
        the conditioning locations.
        """
        left_side = []
        if self.conds_eq_ij is not None:
            for cond_eq_ij in self.conds_eq_ij:
                rows, cols = list(map(list, cond_eq_ij))
                left_side += [realizations[:, rows, cols]]
        if self.conds_mean_ij is not None:
            for cond_mean_ij in self.conds_mean_ij:
                rows, cols = list(map(list, cond_mean_ij))
                mean = np.mean(realizations[:, rows, cols], axis=1)
                left_side += [mean[:, None]]
        return np.hstack(left_side).T

    def run(self, inner_thresh=.15):
        n_conditions = len(self.cv_total)
        if self.n_realizations is None:
            self.n_realizations = (int(5. * n_conditions)
                                   if self.pool_size is None
                                   else self.pool_size)
        if self.n_realizations < n_conditions:
            self.n_realizations = n_conditions
        self.n_extra = 2 * int(n_conditions)
        if self.mean_trans is not None:
            self.n_extra += len(self.conds_trans_mean_values)
        realizations = self.spec.sim_n(self.n_realizations)
        norm_inner = 42
        while norm_inner > inner_thresh:
            A = self.gen_A(realizations)
            U, S, V = np.linalg.svd(A)
            c = np.dot(self.cv_total, U)
            norm_inner = np.sum((old_div(c, S)) ** 2)
            if self.verbose > 1:
                print("\tn_realizations: %d, norm: %.3f" %
                      (self.n_realizations, norm_inner))

            if norm_inner > inner_thresh:
                extra_realizations = self.spec.sim_n(self.n_extra)
                realizations = np.concatenate((realizations,
                                               extra_realizations))
                self.n_realizations += self.n_extra

        s = np.sum((old_div(c, S)) * V.T[:, :S.shape[0]], axis=1)
        interpolator = np.tensordot(s, realizations, axes=1)

        realizations_compl = self.spec.sim_n(self.n_realizations)
        cv_total_compl = np.zeros_like(self.cv_total)
        norm_compl = 0.
        alpha = np.random.uniform(-1, 1, A.shape[1] - n_conditions)
        while norm_compl < (1. - norm_inner):
            if self.verbose > 1:
                print("Finding complementary solution.")
            A = self.gen_A(realizations_compl)
            A1 = A[:n_conditions, :n_conditions]
            try:
                A1_inv = np.linalg.inv(A1)
            except np.linalg.LinAlgError:
                if self.verbose:
                    print("LinAlgError in second step. Shuffling.")
                np.random.shuffle(realizations_compl)
                A = self.gen_A(realizations_compl)
                A1 = A[:n_conditions, :n_conditions]
                continue
            A1_inv_y = np.dot(A1_inv, cv_total_compl)
            # we pad this sucker at the right side, so that the
            # addition can deal with the shapes of Ax + A(alpha*sol)
            x = np.hstack((A1_inv_y, np.zeros(A.shape[1] - n_conditions)))

            # solutions that do not change the conditioned values
            sol = np.dot(A1_inv, A[:, n_conditions:]).T
            # this is the same as:
            # sol = np.array([np.dot(A1_inv, A[:, col])
            #                 for col in range(n_conditions, A.shape[1])])

            # pad right side with negative identity
            idd = -1. * np.identity(A.shape[1] - n_conditions)
            sol = np.hstack((sol, idd))

            # each combination of Ax + A(alpha*sol) is a valid solution
            # alpha = np.random.uniform(-1, 1, A.shape[1] - n_conditions)
            len_diff = A.shape[1] - n_conditions - len(alpha)
            if len_diff > 0:
                alpha = np.concatenate((alpha,
                                        np.random.uniform(-1, 1, len_diff)))
            if self.mean_trans is not None:
                interim = InterimFactory(x=x, sol=sol,
                                         norm_inner=norm_inner,
                                         interpolator=interpolator,
                                         alpha=alpha,
                                         realizations_compl=realizations_compl,
                                         success=False)
                interim = self._fit_transformed_mean(interim)
                if not interim.success:
                    realizations_compl = interim.realizations_compl
                    continue
                else:
                    alpha = interim.alpha

            xx = x + np.dot(alpha, sol)
            norm_compl = np.dot(xx, xx)

        tt = np.sqrt(old_div((1. - norm_inner), norm_compl))
        return (interpolator +
                np.tensordot(tt * xx, realizations_compl, axes=1))

    def _fit_transformed_mean(self, interim):
        doys = self.t_kwds["doys"]
        targets = np.full((self.K, self.spec.T), np.nan)
        conds_trans_mean_ij = np.squeeze(self.conds_trans_mean_ij)
        for i, cond_ij in enumerate(conds_trans_mean_ij):
            row, col = list(map(list, cond_ij))
            targets[row, col] = self.conds_trans_mean_values[i]

        targets = np.where(np.isfinite(targets),
                           self.mean_trans(targets, doys),
                           np.nan)
        agg_funcs = [self.agg_funcs[row_ii[0]]
                     for row_ii, _ in conds_trans_mean_ij]
        mean_targets = np.array([agg(targets[list(map(list, cond_ij))])
                                 for agg, cond_ij in
                                 zip(agg_funcs,
                                     conds_trans_mean_ij)])
        # mean_targets = np.array([np.mean(targets[map(list, cond_ij)])
        #                          for cond_ij in conds_trans_mean_ij])
        if np.all(np.isnan(targets)):
            t_means = 0
            t_stds = 1
        else:
            t_means = np.nanmean(targets, axis=1)[:, None]
            t_stds = np.nanstd(targets, axis=1)[:, None]
            # don't blow up!
            t_stds[t_stds < 1] = 1

        def z_trans(x):
            return (x - t_means) / t_stds

        @my.cache("solution")
        def squared_error(alpha):
            xx = interim.x + np.dot(alpha, interim.sol)
            norm_compl = np.dot(xx, xx)
            tt = np.sqrt(old_div((1. - interim.norm_inner), norm_compl))
            sim = (interim.interpolator +
                   np.tensordot(tt * xx, interim.realizations_compl,
                                axes=1))
            sim_trans = self.mean_trans(sim, doys)
            sim_trans_mean = np.array([agg(sim_trans[list(map(list, cond_ij))])
                                       for agg, cond_ij in
                                       zip(agg_funcs, conds_trans_mean_ij)])
            # sim_trans_mean = np.array([np.mean(sim_trans[map(list, cond_ij)])
            #                            for cond_ij in conds_trans_mean_ij])

            return np.sum((z_trans(sim_trans_mean) -
                           z_trans(mean_targets)) ** 2)

        if squared_error.solution is not None:
            alpha = interim.alpha
            alpha[:-self.n_extra] = squared_error.solution
            interim = interim._replace(alpha=alpha)
        tol = 1e-1
        result = optimize.minimize(squared_error, interim.alpha,
                                   tol=tol,
                                   # method="Nelder-Mead",
                                   # method="COBYLA",
                                   # method="trust-ncg",
                                   # options=dict(gtol=tol, disp=True),
                                   options=dict(
                                       disp=self.verbose,
                                       gtol=tol)
                                   )
        alpha = result.x
        rmse = np.sqrt(result.fun /
                       max(1, len(conds_trans_mean_ij)))
        if rmse > .5:
            n_throwaway = int(old_div(len(interim.realizations_compl), 3))
            realizations_compl = \
                np.concatenate((interim.realizations_compl[n_throwaway:],
                                self.spec.sim_n(self.n_extra + n_throwaway)))
            # np.random.shuffle(realizations_compl)
            if self.verbose:
                print(("RMSE: %.4f. " % rmse) +
                      "drawing more random time series (%d + %d)" %
                      (len(interim.realizations_compl), self.n_extra))
            squared_error.solution = alpha
            self.n_extra = int(1.2 * self.n_extra)
            interim = interim._replace(success=False, alpha=alpha,
                                       realizations_compl=realizations_compl)
            return interim
        else:
            squared_error.clear_cache()
            interim = interim._replace(success=True, alpha=alpha)
        return interim


def disaggregate(data, autocov, disagg_len, cov=None, trans=None, t_kwds=None,
                 **kwds):
    """Disaggregate a time series using conditional simulation.

    Parameter
    ---------
    data : (T,) or (K, T) array
        T - number of time steps, K - number of variables
    autocov : callable or sequence of callables
        Autocovariance model returning a covariance for a given distance.
    disagg_len : int
        How many disaggregated timesteps fit in one aggregated.
    cov : (K, K) array, optional
        If supplied data is 2d, covariances have to be supplied.
    trans : None or (K,)-len sequence of callables
        Transformation function to be applied to the mean conditioning.
    t_kwds : None or dict, optional
        keyword-arguments for the trans callable.
    **kwds : optional
        Will be passed on to `simulate`.
    """
    data = np.atleast_2d(data)
    K = data.shape[0]
    # for avoiding unintended array multiplication
    disagg_len = int(disagg_len)
    # length of the disaggregated time series
    T = data.shape[1] * disagg_len
    # all the data are mean conditioning values!
    if K > 1:
        cond_ii = tuple([(disagg_len * [k],  # which variable
                          list(range(t, t + disagg_len)))  # how many timesteps
                         for k in range(K)
                         for t in range(0, T, disagg_len)])
    else:
        cond_ii = np.arange(T).reshape(-1, disagg_len)
    cond_vals = data.ravel()
    if cov is None:
        return simulate(autocov, T, cov=cov, cond_mean=(cond_ii, cond_vals))

    sim = SimulateMulti(T, autocov, cov, **kwds)
    if trans is None:
        sim.cond_mean(cond_ii, cond_vals)
    else:
        # condition on the first vector of values from data, otherwise
        # there is no first step in random mixing
        sim.cond_eq([list(range(K)), K * [0]],
                    data[:, 0][None, :])
        sim.cond_trans_mean(cond_ii, cond_vals, trans, t_kwds)
    return sim.run()


def disaggregate_piecewice(data, autocov, disagg_len, cov=None,
                           thresh=.2, pool_size=None, trans=None,
                           t_kwds=None, sum_vars=None, verbose=False):
    """Disaggregate a time series using conditional simulation.
    `disaggregate` is slow for long time series. This version disaggregates
    in a piecewise fashion. Pieces are as long as the correlation length.

    Parameter
    ---------
    data : (T,) or (K, T) array
        T - number of time steps, K - number of variables
    autocov : ndarray, callable or sequence of callables
        Autocovariance model returning a covariance for a given distance.
        If given as ndarray, it is interpreted as data of which the
        autocovariace will be estimated with time_series.auto_cov.
    disagg_len : int
        How many disaggregated timesteps fit in one aggregated.
    cov : (K, K) array, optional
        If supplied data is 2d, covariances have to be supplied.
    thresh : float, optional
        Autocorrelation threshold for estimation of the correlation length.
    trans : None or (K,)-len sequence of callables
        Transformation function to be applied to the mean conditioning.
    t_kwds : None or dict, optional
        keyword-arguments for the trans callable.
    sum_vars : int or sequence of ints
        Disaggregate these variables according to sum (instead of mean).
    verbose : boolean, optional
        Spit out some information (mostly on different lenghts).
    """
    if t_kwds is None:
        t_kwds = {}
    # for avoiding unintended array multiplication
    disagg_len = int(disagg_len)
    data = np.atleast_2d(data)
    K = data.shape[0]

    autocov = np.atleast_1d(autocov)
    if type(autocov[0]) is np.ndarray:
        acs = []
        for ac_i, ac in enumerate(autocov):
            def ac(ac_i=0):
                return lambda h: time_series.auto_cov(autocov[ac_i], h)

            acs += [ac(ac_i)]
    else:
        acs = autocov
    # finding the correlation range
    corr_lens = []
    for ac in acs:
        corr, corr_len = 1, 0
        while corr >= thresh:
            corr_len += 1
            corr = abs(ac(corr_len))
        corr_lens += [corr_len]
    corr_len = max(corr_lens)
    if len(acs) == 1:
        acs = acs[0]
    else:
        acs = list(acs)

    # the number of timesteps in the disaggregated time series
    T = data.shape[1] * disagg_len
    # number of disaggregated time steps that should be equal to the last
    # values in the last chunk
    n_overlap = max(corr_len // 10, 2)
    # the chunks should be multiples of disagg_len (plus overlap)
    chunk_len = (corr_len // disagg_len + 1) * disagg_len + n_overlap
    data_sim = np.empty((T // (chunk_len - n_overlap) + 1,
                         K,
                         chunk_len - n_overlap))

    # as the indices stay the same in every chunk, we can pre-compile them
    # (outside of the next loop)
    if K == 1:
        cond_eq_ii = list(range(n_overlap))
        cond_mean_ii = np.arange(n_overlap, chunk_len).reshape(-1,
                                                               disagg_len)
    else:
        cond_eq_ii = (tuple(np.arange(K).repeat(n_overlap)),
                      tuple(K * list(range(n_overlap))))
        cond_mean_ii = tuple([(disagg_len * [k],
                               list(range(t, t + disagg_len)))
                              for k in range(K)
                              for t in range(n_overlap, chunk_len,
                                              disagg_len)])

    # reshape input data so that we can iterate through it the same way as
    # with data_sim
    n_chunks = data_sim.shape[0]
    chunk_len_agg = old_div(data_sim.shape[-1], disagg_len)
    data_chunked = [data[:, t: t + chunk_len_agg]
                    for t in range(0, data.shape[1] + chunk_len_agg - 1,
                                   chunk_len_agg)]

    if "doys" in t_kwds:
        doys = t_kwds["doys"]
        doys_chunked = [doys[t * disagg_len
                             - (old_div(n_overlap, 2) if t > 0 else 0):
                             disagg_len * (t + chunk_len_agg)
                             + (old_div(n_overlap, 2) if t > 0 else 0)
                             ]
                        for t in range(0, data.shape[1] + chunk_len_agg - 1,
                                       chunk_len_agg)]
        if not data_chunked[-1]:
            n_chunks -= 1
            data_chunked = data_chunked[:-1]
            doys_chunked = doys_chunked[:-1]
        kwds = dict(doys=doys_chunked[0])
    else:
        kwds = {}

    if verbose:
        print("corr_len: %d, chunk_len: %d, n_overlap: %d, n_chunks: %d" %
              (corr_len, chunk_len, n_overlap, n_chunks))

    data_sim[0] = disaggregate(data_chunked[0], acs, disagg_len, cov,
                               pool_size=pool_size, trans=trans,
                               t_kwds=kwds, sum_vars=sum_vars,
                               verbose=False)
    if n_chunks == 1:
        return data_sim[0]

    if cov is not None:
        sim = SimulateMulti(chunk_len, acs, cov, pool_size=pool_size,
                            verbose=verbose)
    for i in tqdm(range(1, n_chunks), disable=(not verbose)):
        cond_eq_vals = data_sim[i - 1, :, -n_overlap:].ravel()
        cond_mean_vals = data_chunked[i].ravel()
        # chunks are not garanteed to have chunk_len size (last one is smaller)
        if "doys" in kwds and kwds["doys"] is not None:
            if np.array(cond_mean_ii)[:, 1].max() > len(doys_chunked[i]):
                rest_len = (data_chunked[i].shape[1] * disagg_len
                            + old_div(n_overlap, 2))
                cols = [list(range(t, min(t + disagg_len, rest_len)))
                        for t in range(n_overlap, rest_len, disagg_len)]
                if K > 1:
                    cond_mean_ii = tuple([(len(col) * [k], col)
                                          for k in range(K)
                                          for col in cols])
                else:
                    cond_mean_ii = tuple(cols)

        if cov is None:
            data_sim_pad = simulate(acs, chunk_len, cov,
                                    cond_eq=(cond_eq_ii, cond_eq_vals),
                                    cond_mean=(cond_mean_ii, cond_mean_vals),
                                    pool_size=pool_size,
                                    verbose=False)
        else:
            sim.reset()
            sim.cond_eq(cond_eq_ii, cond_eq_vals)
            if trans is None:
                sim.cond_mean(cond_mean_ii, cond_mean_vals)
            else:
                if "doys" in t_kwds:
                    kwds = dict(doys=doys_chunked[i])
                else:
                    kwds = {}
                sim.cond_trans_mean(cond_mean_ii, cond_mean_vals,
                                    trans, t_kwds=kwds)
            data_sim_pad = sim.run()
        try:
            data_sim[i] = np.atleast_2d(data_sim_pad)[:, n_overlap:]
        except ValueError:
            data_sim[i] = np.atleast_2d(data_sim_pad)[:, old_div(n_overlap,2):]
    return np.hstack(tuple(data_sim))[:, :T]


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", RuntimeWarning)
    np.random.seed(1)
    import vg
    import config_konstanz_disag
    vg.conf = vg.vg_base.conf = vg.vg_plotting.conf = config_konstanz_disag
    met_vg = vg.VG(("R",
                    # "theta",
                    # "ILWR",
                    # "Qsw",
                    "rh",
                    "u", "v"
                    ),
                   # refit=True,
                   verbose=True)
    met_vg.simulate(
        # T=40
        T=2 * 365,
        primary_var="R",
    )
    met_vg.disaggregate_rm(refit="rh")
    met_vg.plot_corr(hourly=True)
    met_vg.plot_daily_cycles()
    met_vg.plot_meteogramm(hourly=True)
