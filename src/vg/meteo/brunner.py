import bottleneck
import numpy as np
from scipy import stats
import xarray as xr


def brunner_compound(sti, spi, sequential=False, progress=False):
    """Rank-based hot-dry index.


    Notes
    -----
    Brunner 2021 uses E-GPD for precipitation and a STI index for
    temperature. This implementation just uses empirical ranks.

    References
    ----------
    Brunner, Manuela I., Eric Gilleland, and Andrew W. Wood.
    “Space–Time Dependence of Compound Hot–Dry Events in the United
    States: Assessment Using a Multi-Site Multi-Variable Weather
    Generator.” Earth System Dynamics 12, no. 2 (May 19, 2021):
    621–34. https://doi.org/10.5194/esd-12-621-2021.
    """
    if progress:
        from tqdm import tqdm as progress
    else:

        def progress(x, *args, **kwds):
            return x

    assert len(sti) == len(spi)
    sti_ranks = bottleneck.nanrankdata(sti, axis=-1)
    spi_ranks = bottleneck.nanrankdata(-spi, axis=-1)
    try:
        if sequential:
            raise MemoryError
        bc = np.mean(
            (sti_ranks[..., None, :] <= sti_ranks[..., :, None])
            & (spi_ranks[..., None, :] <= spi_ranks[..., :, None]),
            axis=-1,
        )
        nan_mask = np.isnan(sti_ranks) | np.isnan(spi_ranks)
        bc[nan_mask] = np.nan
    except MemoryError:
        shape_before = sti_ranks.shape
        sti_ranks = sti_ranks.reshape(-1, shape_before[-1])
        spi_ranks = spi_ranks.reshape(-1, shape_before[-1])
        bc = np.full_like(sti_ranks, np.nan)
        for row_i in progress(range(len(sti_ranks)), total=len(sti_ranks)):
            for i, (sti_rank, spi_rank) in enumerate(
                zip(sti_ranks[row_i], spi_ranks[row_i])
            ):
                if np.isnan(sti_rank) or np.isnan(spi_rank):
                    continue
                sti_row = sti_ranks[row_i]
                spi_row = spi_ranks[row_i]
                finite_mask = np.isfinite(sti_row) & np.isfinite(spi_row)
                if np.any(finite_mask):
                    bc[row_i, i] = np.mean(
                        (sti_row <= sti_rank) & (spi_row <= spi_rank),
                        # (sti_row[finite_mask] <= sti_rank)
                        # & (spi_row[finite_mask] <= spi_rank),
                        axis=-1,
                    )
        bc = bc.reshape(shape_before)
    return bc


def _measure_ar(
    varname, distribution, agg_func_name, obs_ar, weeks=1, reference=None
):
    obs_ar = obs_ar.transpose(..., "time").rolling(time=weeks * 7)
    obs_ar = getattr(obs_ar, agg_func_name)()
    if reference is None:
        reference = obs_ar
    else:
        reference = reference.rolling(time=weeks * 7)
        reference = getattr(reference, agg_func_name)()
    months = reference.time.dt.month
    param_names = [param.name for param in distribution._param_info()]
    param_coords = {key: obs_ar.coords[key] for key in obs_ar.dims}
    param_coords["param"] = param_names
    params = xr.DataArray(
        np.empty(obs_ar.shape + (len(param_names),)),
        coords=param_coords,
    )
    for month, monthly_data in reference.groupby(months):
        flat = monthly_data.data.reshape(-1, monthly_data.sizes["time"])
        params_flat = np.array(
            [distribution.fit(row[np.isfinite(row)]) for row in flat]
        )
        params.loc[dict(time=(months == month))] = params_flat.reshape(
            monthly_data.shape[:-1] + (1, len(param_names))
        )
    measure_data = stats.norm.ppf(
        distribution.cdf(
            obs_ar,
            **{
                key: val.data
                for key, val in params.to_dataset("param").items()
            },
        )
    )
    measure_coords = obs_ar.coords.copy()
    measure_ar = xr.DataArray(
        measure_data, coords=measure_coords, name=varname
    )
    return measure_ar


def STI_ds(obs_ds, weeks=1, reference=None):
    sti_ar = _measure_ar(
        "sti",
        stats.norm,
        "mean",
        obs_ds.to_dataarray("variable"),
        weeks=weeks,
        reference=reference,
    )
    return xr.Dataset(dict(sti=sti_ar))


def STI_ar(obs_ar, weeks=1, reference=None):
    return _measure_ar(
        "sti",
        stats.norm,
        "mean",
        obs_ar,
        weeks=weeks,
        reference=reference,
    )


def SPI_ds(obs_ds, weeks=1, reference=None):
    spi_ar = _measure_ar(
        "spi",
        stats.genpareto,
        "sum",
        obs_ds.to_dataarray("variable"),
        weeks=weeks,
        reference=reference,
    )
    return xr.Dataset(dict(spi=spi_ar))


def SPI_ar(obs_ar, weeks=1, reference=None):
    return _measure_ar(
        "spi",
        stats.genpareto,
        "sum",
        obs_ar,
        weeks=weeks,
        reference=reference,
    )


if __name__ == "__main__":
    import dwd_opendata

    theta = dwd_opendata.load_data(
        ("Stötten", "Konstanz"), "air_temperature", time="hourly"
    ).to_dataset("station")
    theta = (
        theta.interpolate_na("time")
        .transpose(..., "time")
        .squeeze()
        .resample(time="1d")
        .sum()
        .sel(time=slice("1990", "2020"))
        .dropna("time")
    )
    sti = STI_ds(theta, weeks=6)

    prec = dwd_opendata.load_data(
        ("Stötten", "Konstanz"), "precipitation", time="hourly"
    ).to_dataset("station")
    prec = (
        prec.interpolate_na("time")
        .transpose(..., "time")
        .squeeze()
        .resample(time="1d")
        .sum()
        .sel(time=slice("1990", "2020"))
        .dropna("time")
    )
    spi = SPI_ds(prec, weeks=6)
