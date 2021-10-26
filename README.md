# ML Climate Reconstruction

This repository stores code for the implementation of machine learning based reconstruction of climate information.

Table of contents
=================

* [Table of Contents](#table-of-contents)
	* [Data processing](#data-processing)


## Dataloader.py
* Mask assimilation data using 
	cdo -f nc4 -z zip_1 -ymonmul /work/uo1075/u301617/Assimiliationslauf/tos_Omon_MPI-ESM-LR_asSEIKERAf_r8i8p4_195801-202010.nc -lec,9999 -sellevel,6 -selname,tho /work/uo1075/u301617/masken/en4_202001_202012_1744x872_GR15L40.nc /work/uo1075/u301617/Asi_maskiert/tos_r8_mask_en4_2020.nc &

* Process to hdf5 files using the preprocessing function
	* extracts variables from netCDF4 files using xarray; posssibility to plot images

* Class Maskdataset: Loads temperature data from hdf5 datasets for mask, image and masked image from predefined paths
	* converts them to pytorch tensors for later use in CNN
	* returns mask, image and masked image tensors
	* prossibility to extract length of image vector for later use


