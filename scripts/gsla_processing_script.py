import argparse
import sys
import s3fs
import xarray as xr
import holoviews as hv
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
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

def to_png_overlay(dataset_in, filename, vmin=-1.2, vmax=1.2, linthresh=0.1, brightness=0.7):
    """
    Create a PNG overlay visualization of GSLA data with transparent land areas.
    Uses matplotlib with SymLogNorm for logarithmic scaling that handles negative values.

    Args:
        dataset_in: xarray Dataset containing GSLA data
        filename: Output filename for the PNG
        vmin: Minimum value for colormap normalization, -1.2 is the min of gsla.
        vmax: Maximum value for colormap normalization, 1.2 is the max of gsla.
        linthresh: Linear threshold for SymLogNorm. Values between -linthresh and +linthresh
                  are scaled linearly, outside this range logarithmically. Default 0.1.
        brightness: Brightness factor (0.0 = black, 1.0 = original, >1.0 = brighter). Default 0.7.
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

        # Transform coordinate arrays to Web Mercator
        lon_2d, lat_2d = np.meshgrid(dataset.LONGITUDE.values, dataset.LATITUDE.values)
        x_coords, y_coords = transformer.transform(lon_2d.flatten(), lat_2d.flatten())
        x_coords = x_coords.reshape(lon_2d.shape)
        y_coords = y_coords.reshape(lat_2d.shape)

        # this is from Gabriela.Semolinipilo@csiro.au and this should be same in frontend when generate legend.
        colors_list = [
            (0, 0, 0.482),
            (0, 0, 0.50218),
            (0, 0, 0.52534),
            (0, 0, 0.54956),
            (0, 0.025383, 0.58095),
            (0, 0.07278, 0.61329),
            (0, 0.12013, 0.64869),
            (0, 0.16728, 0.68794),
            (0, 0.22396, 0.73282),
            (0, 0.28639, 0.78747),
            (0, 0.34613, 0.84284),
            (0, 0.40579, 0.89829),
            (0, 0.47426, 0.9398),
            (0, 0.54918, 0.97544),
            (0, 0.61362, 0.99171),
            (0, 0.66965, 0.99559),
            (0, 0.7232, 0.98594),
            (0, 0.77437, 0.96076),
            (0, 0.80867, 0.89862),
            (0, 0.82905, 0.81967),
            (0, 0.82884, 0.7443),
            (0, 0.80974, 0.66939),
            (0, 0.7863, 0.59404),
            (0, 0.76642, 0.51917),
            (0, 0.74988, 0.44772),
            (0, 0.7339, 0.37945),
            (0, 0.72899, 0.32053),
            (0, 0.72878, 0.26877),
            (0, 0.74456, 0.21724),
            (0, 0.76053, 0.16994),
            (0, 0.79162, 0.12556),
            (0, 0.82383, 0.082186),
            (0, 0.85513, 0.046833),
            (0, 0.88238, 0.011428),
            (0.18068, 0.91064, 0),
            (0.41904, 0.94198, 0),
            (0.56948, 0.966, 0),
            (0.696, 0.98928, 0),
            (0.77211, 0.99599, 0),
            (0.83585, 0.9961, 0),
            (0.87785, 0.97825, 0),
            (0.91384, 0.95503, 0),
            (0.93912, 0.92141, 0),
            (0.96858, 0.88564, 0),
            (0.98192, 0.85183, 0),
            (0.99397, 0.82057, 0),
            (0.996, 0.78241, 0),
            (0.99795, 0.74317, 0),
            (0.99787, 0.69933, 0),
            (0.99408, 0.65588, 0),
            (0.97754, 0.61753, 0),
            (0.95985, 0.57266, 0),
            (0.93568, 0.53352, 0),
            (0.91237, 0.48932, 0),
            (0.88869, 0.45078, 0),
            (0.86057, 0.40665, 0),
            (0.8327, 0.36373, 0),
            (0.80153, 0.31613, 0),
            (0.77014, 0.26528, 0),
            (0.73781, 0.21383, 0),
            (0.71859, 0.16203, 0),
            (0.69843, 0.10697, 0),
            (0.67819, 0.055343, 0),
            (0.659, 0, 0),
        ]

        custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors_list, N=256)

        # Create figure with specific size (adjust as needed)
        fig, ax = plt.subplots(figsize=(10, 7), dpi=600/100)  # 600 DPI equivalent

        # Remove all axes, frames, and whitespace
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Create SymLogNorm for logarithmic scaling with negative value support
        norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

        # Create pcolormesh plot in Web Mercator coordinates with SymLogNorm
        im = ax.pcolormesh(
            x_coords,
            y_coords,
            dataset.GSLA.values,
            cmap=custom_cmap,
            norm=norm,
            shading='auto'
        )

        # Set exact limits to Web Mercator bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Ensure aspect ratio is equal
        ax.set_aspect('equal')

        # Save the plot
        fig.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True, dpi=600)

        # Get image dimensions for land masking
        img = Image.open(filename).convert('RGBA')
        width, height = img.size

        # Load and project land data (same as original)
        land_shp = shpreader.natural_earth(resolution='10m', category='physical', name='land')
        land_reader = shpreader.Reader(land_shp)
        bbox = box(lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1])
        relevant_geoms = [record.geometry for record in land_reader.records()
                          if record.geometry.intersects(bbox)]

        # Transform geometries to Web Mercator
        projected_geoms = []
        for geom in relevant_geoms:
            projected_geom = transform(transformer.transform, geom)
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

        # Apply transparency to land areas and adjust brightness
        img_array = np.array(img)

        # Apply brightness adjustment to RGB channels only (preserve alpha)
        if brightness != 1.0:
            # Using optimized HSL conversion functions
            def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
                """Convert RGB to HSL with optimized vectorized operations."""
                r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                max_val = np.max(rgb, axis=-1)
                min_val = np.min(rgb, axis=-1)
                diff = max_val - min_val

                # Lightness
                l = (max_val + min_val) / 2

                # Saturation
                s = np.zeros_like(l)
                nonzero = diff > 1e-10  # avoid division by zero
                s[nonzero & (l <= 0.5)] = diff[nonzero & (l <= 0.5)] / (max_val + min_val)[nonzero & (l <= 0.5)]
                s[nonzero & (l > 0.5)] = diff[nonzero & (l > 0.5)] / (2 - max_val - min_val)[nonzero & (l > 0.5)]

                # Hue - vectorized calculation
                h = np.zeros_like(l)
                mask = diff > 1e-10
                hue_raw = np.zeros(rgb.shape[:-1] + (3,))
                hue_raw[..., 0] = ((g - b) / diff) % 6     # if red is max
                hue_raw[..., 1] = ((b - r) / diff) + 2     # if green is max
                hue_raw[..., 2] = ((r - g) / diff) + 4     # if blue is max

                # Pick hue channel based on argmax
                idx = np.argmax(rgb, axis=-1)
                h[mask] = hue_raw[..., 0][mask] * (idx == 0)[mask] \
                          + hue_raw[..., 1][mask] * (idx == 1)[mask] \
                          + hue_raw[..., 2][mask] * (idx == 2)[mask]
                h = (h / 6) % 1  # normalize to [0,1)

                return np.stack([h, s, l], axis=-1)

            def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
                """Convert HSL to RGB with optimized vectorized operations."""
                h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
                rgb = np.zeros_like(hsl)

                # Grayscale (saturation = 0)
                mask_gray = s == 0
                rgb[mask_gray] = l[mask_gray, np.newaxis]

                # Colorful pixels
                mask_color = ~mask_gray
                if np.any(mask_color):
                    h_color = h[mask_color]
                    s_color = s[mask_color]
                    l_color = l[mask_color]

                    q = np.where(l_color < 0.5,
                                 l_color * (1 + s_color),
                                 l_color + s_color - l_color * s_color)
                    p = 2 * l_color - q

                    # Broadcast for r,g,b with hue offsets
                    t = np.stack([
                        h_color + 1/3,  # Red
                        h_color,        # Green
                        h_color - 1/3   # Blue
                    ], axis=-1)
                    t = t % 1  # wrap around [0,1)

                    # Apply piecewise function
                    rgb_color = np.empty_like(t)
                    rgb_color = np.where(t < 1/6, p[..., None] + (q - p)[..., None] * 6 * t, rgb_color)
                    rgb_color = np.where((t >= 1/6) & (t < 1/2), q[..., None], rgb_color)
                    rgb_color = np.where((t >= 1/2) & (t < 2/3),
                                         p[..., None] + (q - p)[..., None] * (2/3 - t) * 6,
                                         rgb_color)
                    rgb_color = np.where(t >= 2/3, p[..., None], rgb_color)
                    rgb[mask_color] = rgb_color

                return rgb

            # Convert to HSL, adjust lightness, convert back
            rgb_data = img_array[:, :, :3] / 255.0  # Normalize to 0-1
            hsl_data = rgb_to_hsl(rgb_data)
            hsl_data[..., 2] = np.clip(hsl_data[..., 2] * brightness, 0, 1)  # Adjust lightness
            rgb_adjusted = hsl_to_rgb(hsl_data)
            img_array[:, :, :3] = (rgb_adjusted * 255).clip(0, 255).astype(np.uint8)


        # Set alpha to 0 for land areas (after brightness adjustment)
        img_array[land_mask.astype(bool), 3] = 0

        # Save final image
        Image.fromarray(img_array, 'RGBA').save(filename)
        plt.close(fig)

    except Exception as e:
        print(f"Error creating overlay PNG {filename}: {e}")
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