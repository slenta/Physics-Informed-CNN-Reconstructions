"""
Split a netCDF file along the member dimension into separate files.

Each member will be saved to a separate netCDF file with the member number in the filename.

Usage:
    python split_nc_by_member.py --input input_file.nc --output_dir output_directory --member_dim member
"""

import argparse
import xarray as xr
from pathlib import Path
from tqdm import tqdm


def split_nc_by_member(input_file, output_dir, member_dim="member", output_prefix=None):
    """
    Split a netCDF file along the member dimension into separate files.
    
    Parameters
    ----------
    input_file : str or Path
        Path to the input netCDF file
    output_dir : str or Path
        Directory where output files will be saved
    member_dim : str, optional
        Name of the member dimension (default: "member")
    output_prefix : str, optional
        Prefix for output filenames. If None, uses input filename stem.
    """
    # Convert to Path objects
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output prefix
    if output_prefix is None:
        output_prefix = input_file.stem
    
    print(f"Loading {input_file}...")
    ds = xr.open_dataset(input_file)
    
    # Check if member dimension exists
    if member_dim not in ds.dims:
        raise ValueError(f"Dimension '{member_dim}' not found in dataset. Available dimensions: {list(ds.dims.keys())}")
    
    # Get member values
    members = ds[member_dim].values
    n_members = len(members)
    
    print(f"Found {n_members} members in dimension '{member_dim}'")
    print(f"Splitting into individual files...")
    
    # Split and save each member
    for member_idx in tqdm(range(n_members), desc="Processing members"):
        # Select single member
        ds_member = ds.isel({member_dim: member_idx})
        
        # Create output filename
        member_value = members[member_idx]
        output_file = output_dir / f"{output_prefix}_member{int(member_value):02d}.nc"
        
        # Save to file
        ds_member.to_netcdf(output_file)
        
    print(f"\nSuccessfully split {n_members} members into {output_dir}")
    ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Split a netCDF file along the member dimension into separate files"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input netCDF file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--member_dim", "-m",
        type=str,
        default="member",
        help="Name of member dimension (default: member)"
    )
    parser.add_argument(
        "--output_prefix", "-p",
        type=str,
        default=None,
        help="Prefix for output filenames (default: input filename)"
    )
    
    args = parser.parse_args()
    
    split_nc_by_member(
        input_file=args.input,
        output_dir=args.output_dir,
        member_dim=args.member_dim,
        output_prefix=args.output_prefix
    )


if __name__ == "__main__":
    main()
