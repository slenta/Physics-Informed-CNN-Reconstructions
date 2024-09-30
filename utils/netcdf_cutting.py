import numpy as np
import config as cfg
import xarray as xr
import h5py
import cdo

cdo = cdo.Cdo()

cfg.set_train_args()

ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9.nc")
ofile = f"{cfg.im_dir}Image_r9_newgrid.nc"
# cdo.sellonlatbox(-65, -5, 20, 69, input = ifile, output = ofile)
ds_compare = xr.load_dataset(f"{cfg.im_dir}Image_r9_newgrid.nc")

length = 752
depth = 20
mode = "assimilation_full"
compare = ds_compare.thetao.values
n = compare.shape

lat = np.array(ds_compare.lat.values)
lon = np.array(ds_compare.lon.values)
time = np.array(ds_compare.time.values)[:length]

f = h5py.File(
    f"{cfg.val_dir}{cfg.save_part}/validation_{cfg.resume_iter}_{mode}_{cfg.eval_im_year}.hdf5",
    "r",
)

cname = ["gt", "output", "image", "mask"]

for name in cname:
    globals()[name] = np.array([f.get(name)]).squeeze(axis=0)
    globals()[name] = globals()[name][:, :, : n[2], : n[3]]

ds = xr.Dataset(
    data_vars=dict(
        output=(["x", "y"], output),
        gt=(["x", "y"], gt),
        mask=(["x", "y"], mask),
        image=(["x", "y"], image),
    ),
    coords=dict(lon=(["lon"], lon), lat=(["lat"], lat)),
    attrs=dict(description=f"Neural network outputs"),
)

ifile = f"{cfg.val_dir}{cfg.save_part}/nn_outputs_{cfg.resume_iter}_{mode}_{cfg.eval_im_year}.nc"
ofile = f"{cfg.val_dir}{cfg.save_part}/nn_outputs_{cfg.resume_iter}_{mode}_{cfg.eval_im_year}_cut.nc"

ds.to_netcdf(ifile)

cdo.sellonlatbox(cfg.lon1, cfg.lon2, cfg.lat1, cfg.lat2, input=ifile, output=ofile)
