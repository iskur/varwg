from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
import warnings
#------------------------------------------------------------------------------#
def simulated_annealing(func, x0, args=(), k=.99, M=None, step_length=None,
                        T0=None, step_length0=5, feps=1e-5, constraints=None,
                        callback=None):
    """Simulated Annealing algorithm with support for constraints, estimation
    for acceptance rate based initial temperature and gradient based 
    calculation of the step lengths."""
    x0 = list(x0)
    n_dims = len(x0)
    M = 20 * n_dims if M is None else M
    if constraints:
        feasible = True
        for constraint in constraints:
            feasible *= constraint(x0)
        if not feasible:
            raise ValueError
    try:
        initial_obj = func(x0, *args)
    except ValueError:
        warnings.warn("Initial solution produced non-finite result.")
        raise
    #--------------------------------------------------------------------------#
    def get_new_candidate(solution):
        local_step_length = step_length
        def gen_new(old):
            return old + local_step_length * (np.random.random(n_dims) - .5)
        if constraints is None:
            return gen_new(solution[:])
        else:
            counter, counter_limit, meta_counter = 0, 100, 1
            # generate candidates until all constraints are met
            feasible = False
            while not feasible:
                new = gen_new(solution[:])
                feasible = True
                for constraint in constraints:
                    feasible *= constraint(new)
                counter += 1
                if counter > counter_limit:
                    print ("Info: Could not find a feasible neighbor in %d "
                           % (counter_limit * 2 ** meta_counter)
                           + "iterations")
                    counter = 0
                    meta_counter += 1
                    counter_limit *= 2
                    # note, that this changes the step_length only in the
                    # local scope!
                    local_step_length *= .75
                if counter_limit * 2 ** meta_counter > 1e6:
                    warnings.warn("Giving up on finding a feasible neighbor")
                    raise RuntimeError
        return new
    #--------------------------------------------------------------------------#
    def initial_acceptance_rate(T0, best_obj_sofar=None, best_sol_sofar=None):
        """Return an acceptance rate for M timesteps starting at x0."""
        decisions = np.zeros(M, dtype=float)
        old_solution = x0[:]
        old_obj = func(old_solution, *args)
        counter, counter_limit, meta_counter = 0, 100, 1
        for ii in range(M):
            candidate = get_new_candidate(old_solution)
            counter += 1
            try:
                new_obj = func(candidate, *args)
            except ValueError:
                # do not accept solutions that produce non-finite output
                if counter > counter_limit:
                    warnings.warn("Warning: Got %d non-finite solutions "
                           % (counter_limit * 2 ** meta_counter)
                           + "during start-up.")
                    counter = 0
                    meta_counter += 1
                    counter_limit *= 2
                continue
            if np.random.random() < np.exp(old_div((old_obj - new_obj), T0)):
                decisions[ii] = 1.
                old_solution, old_obj = candidate, new_obj
                if (best_obj_sofar is not None and (old_obj < best_obj_sofar)):
                    best_obj_sofar, best_sol_sofar = old_obj, old_solution
        if best_obj_sofar is None:
            return decisions.mean()
        else:
            return decisions.mean(), best_obj_sofar, best_sol_sofar
    #--------------------------------------------------------------------------#
    def get_average_gradient(step_length0):
        """Average a gradient of the objective function by taking an M-step
        random walk around a solution."""
        if step_length0 is None:
            step_length0 = 1.
        grad = np.zeros(n_dims, dtype=float)
        old_sol, old_obj = x0, initial_obj
        cur_step = 10
        for ii in range(M):
            dx = step_length0 * (np.random.random(n_dims) - .5)
            new_sol = old_sol + dx
            if constraints:
                feasible = True
                for constraint in constraints:
                    feasible *= constraint(new_sol)
                if not feasible:
                    step_length0 *= .75
                    ii -= 1
                    continue
            try:
                new_obj = func(new_sol, *args)
            except ValueError:
                cur_step *= .75
                continue
            grad += old_div((new_obj - old_obj), dx)
            old_obj, old_sol = new_obj, new_sol
        grad /= np.sum(grad ** 2) ** .5
        return grad
    #--------------------------------------------------------------------------#
    def get_T0():
        """Estimates a starting temperature for which we get an acceptance rate
        that is  between 40-80%."""
        # take a random walk in the 'hood and check the objective values there
        hood, hood_obj = [get_new_candidate(x0)], [initial_obj]
        for ii in range(1, M):
            hood += [get_new_candidate(hood[ii - 1])]
            try:
                hood_obj += [func(hood[-1], *args)]
            except ValueError:
                ii -= 1
                continue
        hood_obj = np.array(hood_obj)
        # also, record the best solution and its value
        best_obj_sofar = np.min(hood_obj)
        best_sol_sofar = hood[np.argmin(hood_obj)]
        mean_exp_difference = np.mean(np.exp(initial_obj -
                                             hood_obj[hood_obj > initial_obj]))

        f_smaller = old_div(np.sum(hood_obj < initial_obj), float(M))
        if f_smaller < .6:
#            def rel_change(T0):
#                return (f_smaller -
#                        np.mean(np.exp((initial_obj - hood[hood > initial_obj])
#                                       / T0)) -
#                        .6) ** 2
#            T0 = optimize.fmin(rel_change, 1)[0]
            T0 = old_div(np.log(.6 - f_smaller), np.log(mean_exp_difference))
        else:
            T0 = old_div(np.log(f_smaller), np.log(mean_exp_difference))

        if T0 <= 0:
            warnings.warn("T0 was %.3f. Starting over" % T0)
            return get_T0()

        find_temp_counter = 0
        accept, best_obj_sofar, best_sol_sofar = \
            initial_acceptance_rate(T0, best_obj_sofar, best_sol_sofar)
        while (accept < .4) or (accept > .8):
            if accept < .4:
                T0 *= 1 + 2 * (1 - accept)
            else:
                T0 /= 1 + 2 * (1 - accept)
            accept, best_obj_sofar, best_sol_sofar = \
                initial_acceptance_rate(T0, best_obj_sofar, best_sol_sofar)
            find_temp_counter += 1
        print ("Found suitable starting temperature after %d objective " %
               (M * (find_temp_counter + 1) + M)
               + "function evaluations.")
        return T0, best_obj_sofar, best_sol_sofar
    #--------------------------------------------------------------------------#
    if step_length is None:
        step_length = get_average_gradient(step_length0)
        if step_length0 is not None:
            step_length *= step_length0
        print ("Step lengths are: " +
               ", ".join("%.3f" % abs(step) for step in step_length))
    if T0 is None:
        T0, best_obj_sofar, best_sol_sofar = get_T0()
    else:
        best_obj_sofar, best_sol_sofar = initial_obj, x0

    print("Initial obj. value: %.3f" % initial_obj)
    print("Best obj. value during start-up: %.3f" % best_obj_sofar)
    old_obj = conv_obj = initial_obj
    conv_sol = best_sol_sofar
    rel_change, old_sol, T, m = feps + 1, best_sol_sofar, T0, 0
    iteration_counter = 0
    counter, counter_limit, meta_counter = 0, 100, 1
    while (iteration_counter < 5 * M) or (abs(rel_change) > feps):
        try:
            new_sol = get_new_candidate(old_sol)
        except RuntimeError:
            break
        counter += 1
        try:
            new_obj = func(new_sol, *args)
        except ValueError:
            if counter > counter_limit:
                warnings.warn("Warning: Discarded %d solutions that produced "
                       % (counter_limit * 2 ** meta_counter)
                       + "a non-finite objective function value.")
                counter = 0
                meta_counter += 1
                counter_limit *= 2
            continue

        if ((new_obj < old_obj) or
            np.random.random() < np.exp(old_div((old_obj - new_obj), T))):
            old_sol, old_obj = new_sol, new_obj
            counter_limit, meta_counter = 100, 1
            if new_obj < best_obj_sofar:
                best_sol_sofar, best_obj_sofar = new_sol, new_obj
        m += 1

        if m == M:
            m = 0
            T *= k
            if np.all(conv_sol != old_sol):
                step_length = old_div((conv_obj - old_obj), (conv_sol - old_sol))
                # include a bit of noise, to avoid zeros in step_length
                step_length += .5 * step_length.mean() * \
                    (np.random.random(n_dims) - .5)
                step_length /= np.sum(step_length ** 2) ** .5
                step_length0 *= k
                step_length *= step_length0
            rel_change = old_div((old_obj - conv_obj), abs(conv_obj))
            conv_obj, conv_sol = old_obj, old_sol
            if callback:
                callback(conv_obj)
            print ("Iteration: %d, obj. value: %.3f, Temp: %f, " %
                   (iteration_counter + 1, old_obj, T) +
                   "rel. change: %.3f, " % rel_change)
#                     +
#                   "Step lengths: %s" %
#                   ", ".join("%.3f" % abs(sl) for sl in step_length)

        iteration_counter += 1
    print ("Converged after %d iterations. Obj. value: %.3f (Initial " %
           (iteration_counter + 1, best_obj_sofar) +
           "obj.: %.3f)" % initial_obj)
    print("Solution: ", best_sol_sofar)
    return best_sol_sofar
#------------------------------------------------------------------------------#
