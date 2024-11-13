"""Get 2021 IMDAA data into the right format to run with the cGAN."""
# %%
import xarray as xr
import matplotlib.pyplot as plt
import geospatial_utils as geo


if __name__ == "__main__":
    file = "../alison-data/truth/_input/imdaa_totalprecip_2021.nc"
    outfile = '../alison-data/truth/_input/tp_formatted_2021.nc'

    # process IMDAA file to be in right format
    print('\nFormatting IMDAA file...')
    if False:
        ds = xr.open_dataset(file)
        ds = ds.sel(longitude=slice(80, 95), latitude=slice(10, 25))
        ds = ds.rename({'APCP_sfc': 'tp_mean'})
        ds['tp_sd'] = ds['tp_mean'] * 0.
        ds.to_netcdf(outfile)

    # make coarsened (4x) forecast dataset
    print("\nMaking forecast data...")
    forecast = geo.resample_dataset(outfile, (32, 32), method="sum")
    forecast['tp_mean'] = forecast['tp_mean'].expand_dims({'forecast': 1}, axis=0)
    forecast['tp_sd'] = forecast['tp_sd'].expand_dims({'forecast': 1}, axis=0)
    forecast.to_netcdf('../alison-data/forecast/2021/tp2.nc')
    # %%
    # %%make truth dataset
    print("\nMaking truth data...")
    truth = geo.resample_dataset(outfile, (128, 128), method='sum')
    truth.to_netcdf('../alison-data/truth/_input/tp_2021.nc')

    #  make files in hourly form as specified for cGAN
    for time in truth['time'].values:
        ds_time = truth.sel(time=time)
        date = ds_time['time.date'].values.item().strftime('%Y%m%d')
        hour = str(ds_time['time.hour'].values).zfill(2)
        ds_time.to_netcdf(f"../alison-data/truth/{date}_{hour}.nc4")

    print("Done.")
# %%
