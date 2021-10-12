# ML Climate Reconstruction

This repository stores code for the implementation of machine learning based reconstruction of climate information.

Table of contents
=================

* [Table of Contents](#table-of-contents)
	* [Data processing](#data-processing)


## Data processing
* Mask assimilation data using 
	cdo -f nc4 -z zip_1 -ymonmul /work/uo1075/u301617/Assimiliationslauf/tos_Omon_MPI-ESM-LR_asSEIKERAf_r8i8p4_195801-202010.nc -lec,9999 -sellevel,6 -selname,tho /work/uo1075/u301617/masken/en4_202001_202012_1744x872_GR15L40.nc /work/uo1075/u301617/Asi_maskiert/tos_r8_mask_en4_2020.nc &

* Insert into dataprocessing.py:
	* complete into symmetric shape
	* plot using plt.imshow
	* save as hdf5 file using h5py


