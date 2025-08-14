import argparse
import sys
import s3fs
import xarray as xr
import holoviews as hv
from hvplot import xarray
import hvplot
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from shapely.ops import transform
import pyproj
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import datetime
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# switches rendering engine from Bokeh to Matplotlib
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
        subset = ds.sel(TIME=date.strftime("%Y-%m-%d"), LATITUDE=slice(-60, 10), LONGITUDE=slice(90, 180))

        logger.info(f"Dataset loaded successfully for {date.strftime('%Y-%m-%d')}")
        return subset

    except Exception as e:
        logger.error(f"Error loading dataset for {date.strftime('%Y-%m-%d')}: {e}")
        raise

def to_png_overlay(dataset_in, filename):
    """
    Create a PNG overlay visualization of GSLA data with transparent land areas.
    Properly aligned for Web Mercator (EPSG:3857) projection.

    Args:
        dataset_in: xarray Dataset containing GSLA data
        filename: Output filename for the PNG
    """
    try:

        # Interpolate missing data
        dataset = dataset_in.copy()
        dataset['GSLA'] = dataset.GSLA.interpolate_na(dim='LONGITUDE', method='linear')
        dataset['GSLA'] = dataset.GSLA.interpolate_na(dim='LATITUDE', method='linear')
        dataset['GSLA'] = dataset.GSLA.ffill(dim='LONGITUDE').bfill(dim='LONGITUDE')
        dataset['GSLA'] = dataset.GSLA.ffill(dim='LATITUDE').bfill(dim='LATITUDE')

        # Get geographic bounds
        lat_bounds = (float(dataset.LATITUDE.min()), float(dataset.LATITUDE.max()))
        lon_bounds = (float(dataset.LONGITUDE.min()), float(dataset.LONGITUDE.max()))

        # Transform geographic bounds to Web Mercator (EPSG:3857)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x_min, y_min = transformer.transform(lon_bounds[0], lat_bounds[0])
        x_max, y_max = transformer.transform(lon_bounds[1], lat_bounds[1])

        # explicitly set, because it needs to be same in frontend.
        custom_viridis =  [
            '#440154', '#440255', '#440357', '#450558', '#45065A', '#45085B', '#46095C', '#460B5E', '#460C5F',
            '#460E61', '#470F62', '#471163', '#471265', '#471466', '#471567', '#471669', '#47186A', '#48196B',
            '#481A6C', '#481C6E', '#481D6F', '#481E70', '#482071', '#482172', '#482273', '#482374', '#472575',
            '#472676', '#472777', '#472878', '#472A79', '#472B7A', '#472C7B', '#462D7C', '#462F7C', '#46307D',
            '#46317E', '#45327F', '#45347F', '#453580', '#453681', '#443781', '#443982', '#433A83', '#433B83',
            '#433C84', '#423D84', '#423E85', '#424085', '#414186', '#414286', '#404387', '#404487', '#3F4587',
            '#3F4788', '#3E4888', '#3E4989', '#3D4A89', '#3D4B89', '#3D4C89', '#3C4D8A', '#3C4E8A', '#3B508A',
            '#3B518A', '#3A528B', '#3A538B', '#39548B', '#39558B', '#38568B', '#38578C', '#37588C', '#37598C',
            '#365A8C', '#365B8C', '#355C8C', '#355D8C', '#345E8D', '#345F8D', '#33608D', '#33618D', '#32628D',
            '#32638D', '#31648D', '#31658D', '#31668D', '#30678D', '#30688D', '#2F698D', '#2F6A8D', '#2E6B8E',
            '#2E6C8E', '#2E6D8E', '#2D6E8E', '#2D6F8E', '#2C708E', '#2C718E', '#2C728E', '#2B738E', '#2B748E',
            '#2A758E', '#2A768E', '#2A778E', '#29788E', '#29798E', '#287A8E', '#287A8E', '#287B8E', '#277C8E',
            '#277D8E', '#277E8E', '#267F8E', '#26808E', '#26818E', '#25828E', '#25838D', '#24848D', '#24858D',
            '#24868D', '#23878D', '#23888D', '#23898D', '#22898D', '#228A8D', '#228B8D', '#218C8D', '#218D8C',
            '#218E8C', '#208F8C', '#20908C', '#20918C', '#1F928C', '#1F938B', '#1F948B', '#1F958B', '#1F968B',
            '#1E978A', '#1E988A', '#1E998A', '#1E998A', '#1E9A89', '#1E9B89', '#1E9C89', '#1E9D88', '#1E9E88',
            '#1E9F88', '#1EA087', '#1FA187', '#1FA286', '#1FA386', '#20A485', '#20A585', '#21A685', '#21A784',
            '#22A784', '#23A883', '#23A982', '#24AA82', '#25AB81', '#26AC81', '#27AD80', '#28AE7F', '#29AF7F',
            '#2AB07E', '#2BB17D', '#2CB17D', '#2EB27C', '#2FB37B', '#30B47A', '#32B57A', '#33B679', '#35B778',
            '#36B877', '#38B976', '#39B976', '#3BBA75', '#3DBB74', '#3EBC73', '#40BD72', '#42BE71', '#44BE70',
            '#45BF6F', '#47C06E', '#49C16D', '#4BC26C', '#4DC26B', '#4FC369', '#51C468', '#53C567', '#55C666',
            '#57C665', '#59C764', '#5BC862', '#5EC961', '#60C960', '#62CA5F', '#64CB5D', '#67CC5C', '#69CC5B',
            '#6BCD59', '#6DCE58', '#70CE56', '#72CF55', '#74D054', '#77D052', '#79D151', '#7CD24F', '#7ED24E',
            '#81D34C', '#83D34B', '#86D449', '#88D547', '#8BD546', '#8DD644', '#90D643', '#92D741', '#95D73F',
            '#97D83E', '#9AD83C', '#9DD93A', '#9FD938', '#A2DA37', '#A5DA35', '#A7DB33', '#AADB32', '#ADDC30',
            '#AFDC2E', '#B2DD2C', '#B5DD2B', '#B7DD29', '#BADE27', '#BDDE26', '#BFDF24', '#C2DF22', '#C5DF21',
            '#C7E01F', '#CAE01E', '#CDE01D', '#CFE11C', '#D2E11B', '#D4E11A', '#D7E219', '#DAE218', '#DCE218',
            '#DFE318', '#E1E318', '#E4E318', '#E7E419', '#E9E419', '#ECE41A', '#EEE51B', '#F1E51C', '#F3E51E',
            '#F6E61F', '#F8E621', '#FAE622', '#FDE724'
        ]

        # Create plot with explicit extent in Web Mercator
        mplt = dataset.GSLA.hvplot.quadmesh(
            title='',
            grid=False,
            cmap=custom_viridis,
            geo=True,
            coastline=False,
            hover=False,
            colorbar=False,
            height=700,
            projection='Mercator',
            xaxis=None,
            yaxis=None
        )

        # Render and save plot
        fig = hvplot.render(mplt, backend="matplotlib")
        fig.axes[0].set_frame_on(False)

        # Set explicit limits in Web Mercator coordinates
        fig.axes[0].set_xlim(x_min, x_max)
        fig.axes[0].set_ylim(y_min, y_max)

        fig.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)

        # Get image dimensions
        img = Image.open(filename).convert('RGBA')
        width, height = img.size

        # Load and project land data
        land_shp = shpreader.natural_earth(resolution='10m', category='physical', name='land')
        land_reader = shpreader.Reader(land_shp)
        bbox = box(lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1])
        relevant_geoms = [record.geometry for record in land_reader.records()
                          if record.geometry.intersects(bbox)]

        # Transform geometries to Web Mercator
        projected_geoms = []
        for geom in relevant_geoms:
            projected_geom = transform(transformer.transform, geom)
            # Apply small buffer if needed (0 means no buffer)
            buffered_geom = projected_geom.buffer(0)
            if not buffered_geom.is_empty:
                projected_geoms.append(buffered_geom)

        # Create rasterio transform using Web Mercator bounds
        transform_obj = from_bounds(x_min, y_min, x_max, y_max, width, height)

        # Rasterize land geometries
        land_mask = rasterize(
            projected_geoms,
            out_shape=(height, width),
            transform=transform_obj,
            fill=0,
            default_value=1,
            dtype=np.uint8
        )

        # Apply transparency to land areas
        img_array = np.array(img)
        img_array[land_mask.astype(bool), 3] = 0  # Set alpha to 0 for land areas

        # Save final image
        Image.fromarray(img_array, 'RGBA').save(filename)
        plt.close(fig)

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

        speed = np.sqrt(u * u + v * v)
        direction = np.arctan2(v, u) * 180 / np.pi
        direction = np.where(direction < 0, direction + 360, direction)

        combined = np.stack((speed, direction, gsla), axis=-1).tolist()
        rounded = [[[round(speed, 2), round(direction, 2), round(gsla, 2)]
                    for speed, direction, gsla in row] for row in combined]

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
    Get the raw bounds, bounds with offset adjusted and uRange and vRange metadata.
    raw bounds will be used for overlay layer. adjusted bounds will be used for particle layer and map bounds.

    Args:
        dataset_in: xarray Dataset containing the data
        filename: Output filename for the metadata JSON
    """
    try:
        lat_min, lat_max = dataset_in.LATITUDE.min().values.item(), dataset_in.LATITUDE.max().values.item()
        lat_offset = 0.5 * (lat_max - lat_min) / len(dataset_in.LATITUDE)
        lon_min, lon_max = dataset_in.LONGITUDE.min().values.item(), dataset_in.LONGITUDE.max().values.item()
        lon_offset = 0.5 * (lon_max - lon_min) / len(dataset_in.LONGITUDE)

        dataset_in["UCUR_NEW"] = dataset_in.UCUR.fillna(0.)
        dataset_in["VCUR_NEW"] = dataset_in.VCUR.fillna(0.)
        dataset_in["GSLA_NEW"] = dataset_in.GSLA.fillna(0.)

        u = dataset_in["UCUR_NEW"].values
        v = dataset_in["VCUR_NEW"].values
        speed = np.sqrt(u * u + v * v)

        metadata = {
            "rawLatRange": [lat_min, lat_max],
            "rawLonRange": [lon_min, lon_max],
            "latRange": [lat_min - lat_offset, lat_max + lat_offset],
            "lonRange": [lon_min - lon_offset, lon_max + lon_offset],
            "uRange": [dataset_in["UCUR_NEW"].min().values.item(), dataset_in["UCUR_NEW"].max().values.item()],
            "vRange": [dataset_in["VCUR_NEW"].min().values.item(), dataset_in["VCUR_NEW"].max().values.item()],
            "speedRange": [round(speed.min().item(),2), round(speed.max().item(),2)],
            "gslaRange":[round(dataset_in["GSLA_NEW"].min().values.item(),2), round(dataset_in["GSLA_NEW"].max().values.item(),2)]
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