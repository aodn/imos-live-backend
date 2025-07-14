import argparse
import sys
import s3fs
import xarray as xr
import holoviews as hv
from hvplot import xarray
import hvplot
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hv.extension('matplotlib')

def get_dataset(date):
    """
    Fetch GSLA dataset for a specific date from IMOS S3 bucket.

    Args:
        date: datetime object for the target date

    Returns:
        xarray Dataset with GSLA data
    """
    try:
        s3 = s3fs.S3FileSystem(anon=True)
        file_list = s3.ls(f"imos-data/IMOS/OceanCurrent/GSLA/NRT/{date.year}/")

        # Find file matching the date
        matching_file = next((f for f in file_list if date.strftime("%Y%m%d") in f), None)

        if not matching_file:
            raise FileNotFoundError(f"No GSLA file found for date {date.strftime('%Y-%m-%d')}")

        logger.info(f"Loading GSLA data from: {matching_file}")

        ds = xr.open_dataset(s3.open(matching_file))

        # add coordinate reference system info so that xarray can interpret it
        ds.attrs['crs'] = ccrs.PlateCarree()  # "WGS84"

        # return slightly smaller subset to speed up and avoid issues around LON=180...
        subset = ds.sel(TIME=date.strftime("%Y-%m-%d"), LATITUDE=slice(-50, 0), LONGITUDE=slice(110, 170))

        logger.info(f"Dataset loaded successfully for {date.strftime('%Y-%m-%d')}")
        return subset

    except Exception as e:
        logger.error(f"Error loading dataset for {date.strftime('%Y-%m-%d')}: {e}")
        raise

def to_png_overlay(dataset_in, filename):
    """
    Create a PNG overlay visualization of GSLA data.

    Args:
        dataset_in: xarray Dataset containing GSLA data
        filename: Output filename for the PNG
    """
    try:
        # Plot data in web mercator projection
        # quadmesh is a latitude-longitude grid, where each point has a value.
        mplt = dataset_in.GSLA.hvplot.quadmesh(
            title='',
            grid=False,
            cmap='viridis',
            geo=True,
            coastline="10m",
            hover=True,
            colorbar=False,
            height=700,
            projection='Mercator',
            xaxis=None,
            yaxis=None
        )

        # Save plot without a frame or padding, with NaN as transparent
        # and with a higher than normal resolution
        fig = hvplot.render(mplt, backend="matplotlib")
        fig.axes[0].set_frame_on(False)
        fig.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)

        logger.info(f"Created overlay PNG: {filename}")

    except Exception as e:
        logger.error(f"Error creating overlay PNG {filename}: {e}")
        raise

def to_png_input(dataset_in, filename):
    """
    Convert netcdf file to png image, including information of an area's ocean current per a period of time.

    Args:
        dataset_in: xarray Dataset containing current data
        filename: Output filename for the PNG
    """
    try:
        dataset_in["ALPHA"] = np.logical_not(np.logical_and(dataset_in.UCUR.isnull(), dataset_in.VCUR.isnull()))

        # create new dataset with NaNs removed and rescaled to 0-255
        dataset_in["UCUR_NEW"] = dataset_in.UCUR.fillna(0.)
        dataset_in["VCUR_NEW"] = dataset_in.VCUR.fillna(0.)

        UCUR_MIN, UCUR_MAX = dataset_in.UCUR_NEW.min(), dataset_in.UCUR_NEW.max()
        VCUR_MIN, VCUR_MAX = dataset_in.VCUR_NEW.min(), dataset_in.VCUR_NEW.max()

        # rescale the data to 0-255 for display
        dataset_in["UCUR_NEW"] = 255 * (dataset_in.UCUR_NEW - UCUR_MIN) / (UCUR_MAX - UCUR_MIN)
        dataset_in["VCUR_NEW"] = 255 * (dataset_in.VCUR_NEW - VCUR_MIN) / (VCUR_MAX - VCUR_MIN)

        dataset_in = dataset_in.squeeze()

        stacked = dataset_in.reindex(LATITUDE=list(reversed(dataset_in.LATITUDE)))
        stacked = stacked.stack(z=["LATITUDE", "LONGITUDE"])

        # get the U, V and ALPHA values from the stacked dataset
        Us, Vs, ALPHAs = stacked.UCUR_NEW.values, stacked.VCUR_NEW.values, stacked.ALPHA.values

        # convert data to a png, with U and V in the R and G channels and the show particle flag in B channel
        img_data = []

        for i, (U, V, ALPHA) in enumerate(zip(Us, Vs, ALPHAs)):
            img_data.extend([int(U), int(V), 255*ALPHA, 255])

        img = Image.frombytes('RGBA', (dataset_in.sizes['LONGITUDE'], dataset_in.sizes['LATITUDE']), bytes(img_data))
        img.save(filename)

        logger.info(f"Created input PNG: {filename}")

    except Exception as e:
        logger.error(f"Error creating input PNG {filename}: {e}")
        raise

def to_json_value(dataset_in, filename):
    """
    Convert to a 2-d array including original ucur, vcur and gsla value.

    Args:
        dataset_in: xarray Dataset containing the data
        filename: Output filename for the JSON
    """
    try:
        lat_min, lat_max = dataset_in.LATITUDE.min().values.item(), dataset_in.LATITUDE.max().values.item()
        lat_offset = 0.5 * (lat_max - lat_min) / len(dataset_in.LATITUDE)
        lon_min, lon_max = dataset_in.LONGITUDE.min().values.item(), dataset_in.LONGITUDE.max().values.item()
        lon_offset = 0.5 * (lon_max - lon_min) / len(dataset_in.LONGITUDE)

        dataset_in["UCUR_NEW"] = dataset_in.UCUR.fillna(0.)
        dataset_in["VCUR_NEW"] = dataset_in.VCUR.fillna(0.)
        dataset_in["GSLA_NEW"] = dataset_in.GSLA.fillna(0.)
        dataset_in = dataset_in.squeeze()

        dataset_in = dataset_in.reindex(LATITUDE=list(reversed(dataset_in.LATITUDE)))

        u = dataset_in["UCUR_NEW"].values
        v = dataset_in["VCUR_NEW"].values
        gsla = dataset_in["GSLA_NEW"].values

        combined = np.stack((u, v, gsla), axis=-1).tolist()
        rounded = [[[round(u, 3), round(v, 3), round(gsla, 3)] for u, v, gsla in row] for row in combined]
        output = {
            "width": dataset_in.sizes["LONGITUDE"],
            "height": dataset_in.sizes["LATITUDE"],
            "latRange": [lat_min - lat_offset, lat_max + lat_offset],
            "lonRange": [lon_min - lon_offset, lon_max + lon_offset],
            "data": rounded
        }

        with open(filename, "w") as f:
            json.dump(output, f, separators=(',', ':'))

        logger.info(f"Created data JSON: {filename}")

    except Exception as e:
        logger.error(f"Error creating data JSON {filename}: {e}")
        raise

def to_json_meta(dataset_in, filename):
    """
    Get the bounds and uRange and vRange metadata.

    Args:
        dataset_in: xarray Dataset containing the data
        filename: Output filename for the metadata JSON
    """
    try:
        lat_min, lat_max = dataset_in.LATITUDE.min().values.item(), dataset_in.LATITUDE.max().values.item()
        lat_offset = 0.5 * (lat_max - lat_min) / len(dataset_in.LATITUDE)
        lon_min, lon_max = dataset_in.LONGITUDE.min().values.item(), dataset_in.LONGITUDE.max().values.item()
        lon_offset = 0.5 * (lon_max - lon_min) / len(dataset_in.LONGITUDE)

        metadata = {
            "latRange": [lat_min - lat_offset, lat_max + lat_offset],
            "lonRange": [lon_min - lon_offset, lon_max + lon_offset],
            "uRange": [dataset_in.UCUR.min().values.item(), dataset_in.UCUR.max().values.item()],
            "vRange": [dataset_in.VCUR.min().values.item(), dataset_in.VCUR.max().values.item()]
        }

        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Created metadata JSON: {filename}")

    except Exception as e:
        logger.error(f"Error creating metadata JSON {filename}: {e}")
        raise

def create_gsla_data_for_date(date, base_dir):
    """
    Create all GSLA data products for a specific date.

    Args:
        date: datetime object for the target date
        base_dir: Base directory for output files
    """
    try:
        logger.info(f"Processing GSLA data for {date.strftime('%Y-%m-%d')}")

        # Get the dataset for this date
        dataset = get_dataset(date)

        # Create output directory
        save_dir = base_dir / date.strftime("%Y-%m-%d")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate all output files
        to_png_overlay(dataset, save_dir / "gsla_overlay.png")
        to_png_input(dataset, save_dir / "gsla_input.png")
        to_json_meta(dataset, save_dir / "gsla_meta.json")
        to_json_value(dataset, save_dir / "gsla_data.json")

        logger.info(f"Successfully processed GSLA data for {date.strftime('%Y-%m-%d')}")

    except Exception as e:
        logger.error(f"Error processing GSLA data for {date.strftime('%Y-%m-%d')}: {e}")
        raise

def validate_dates(dates):
    """
    Validate date format and log any issues.

    Args:
        dates: List of date strings

    Returns:
        List of validated datetime objects
    """
    validated_dates = []

    for date_str in dates:
        try:
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            validated_dates.append(date)
        except ValueError as e:
            logger.error(f"❌ Invalid date format: {date_str} (expected YYYY-MM-DD)")
            raise ValueError(f"Invalid date format: {date_str}")

    return validated_dates

def main():
    """Main function with improved error handling and logging."""
    parser = argparse.ArgumentParser(
        description="Generate GSLA overlay/input/meta/data files for specified dates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gsla_processing_script.py --output_base_dir ./output --dates 2025-06-19
  python gsla_processing_script.py --output_base_dir ./output --dates 2025-06-19 2025-06-20
        """
    )
    parser.add_argument(
        "--output_base_dir",
        type=Path,
        required=True,
        help="Directory where processed images and metadata will be saved.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="List of dates in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--log_level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Set the logging level (default: INFO).",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Ensure output directory exists
        args.output_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_base_dir.absolute()}")

        # Validate dates
        validated_dates = validate_dates(args.dates)

        # Process each date
        successful_dates = 0
        for date in validated_dates:
            try:
                create_gsla_data_for_date(date, args.output_base_dir)
                successful_dates += 1
            except Exception as e:
                logger.error(f"Failed to process {date.strftime('%Y-%m-%d')}: {e}")

        logger.info(f"Successfully processed {successful_dates}/{len(validated_dates)} dates")

        if successful_dates == 0:
            logger.error("No dates were processed successfully")
            return 1
        elif successful_dates < len(validated_dates):
            logger.warning("Some dates failed to process")
            return 1
        else:
            logger.info("All dates processed successfully!")
            return 0

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())