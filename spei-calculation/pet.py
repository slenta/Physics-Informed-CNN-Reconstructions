import pyet
import pandas as pd
import xarray as xr


def pet(
    tmean=None,
    wind=None,
    rs=None,
    elevation=None,
    latitude=None,
    tmax=None,
    tmin=None,
    rh=None,
    method="all",
):
    """
    Compute potential evapotranspiration (PET) using the PyEt package.

    Parameters
    ----------
    tmean : pandas.Series or xarray.DataArray
        Mean daily temperature (°C)
    wind : Series/DataArray
        Wind speed at 2 m height (m/s)
    rs : Series/DataArray
        Solar radiation (MJ/m2/day) or sunshine duration
    elevation : float
        Elevation above sea level (m)
    latitude : float
        Latitude in radians
    tmax, tmin : Series/DataArray
        Daily Tmax and Tmin, optional depending on method
    rh : Series/DataArray
        Relative humidity (%)
    method : str
        PyEt method name (e.g. "pm", "hargreaves", "makkink", …)
        or "all" to compute all methods.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        PET estimates for the requested method(s).
    """

    if method == "all":
        return pyet.calculate_all(tmean, wind, rs, elevation, latitude, tmax, tmin, rh)

    # Explicit method call (examples: "pm", "makkink", "hargreaves")
    if not hasattr(pyet, method):
        raise ValueError(
            f"Unknown PET method '{method}'. Check pyet.<TAB> for options."
        )

    pet_func = getattr(pyet, method)

    if method == "hamon":
        return pet_func(tmean, latitude)

    return pet_func(
        tmean,
        wind=wind,
        rs=rs,
        elevation=elevation,
        lat=latitude,
        tmax=tmax,
        tmin=tmin,
        rh=rh,
    )
