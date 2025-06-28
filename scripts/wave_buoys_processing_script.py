import math
from typing import TypedDict, Optional
import xarray as xr
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_dates(dates: list[str]) -> bool:
    """
    Validate that all dates are in correct format and from the same year.

    Args:
        dates: List of date strings in YYYY-MM-DD format and day should be 01.

    Returns:
        True if valid, False otherwise
    """
    if not dates:
        logger.error("No dates provided")
        return False

    try:
        parsed_dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        years = {date.year for date in parsed_dates}

        if len(years) > 1:
            logger.error(f"Dates span multiple years: {years}")
            return False

        return True
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return False

def truncate(value, decimals=2):
    """
    Truncate a float value to a specified number of decimal places.

    Args:
        value: The float value to truncate
        decimals: Number of decimal places to keep (default: 2)

    Returns:
        Truncated float value
    """
    if np.isnan(value):
        return None
    factor = 10 ** decimals
    return math.trunc(value * factor) / factor

def generate_all_catalog_urls_per_year(year: int, buoys: list) -> list:
    """
    Generate catalog URLs for all specified buoys for a given year.

    Args:
        year: The year to generate URLs for
        buoys: List of buoy names/identifiers

    Returns:
        List of dictionaries containing URL and buoy information
    """
    # Base URL template for IMOS wave buoy catalog pages
    catalog_base_url = "https://thredds.aodn.org.au/thredds/catalog/IMOS/COASTAL-WAVE-BUOYS/WAVE-BUOYS/REALTIME/WAVE-PARAMETERS/{buoy}/{year}/catalog.html"

    # Create URL-buoy pairs for each buoy
    return [
        {
            "url": catalog_base_url.format(buoy=buoy, year=year),
            "buoy": buoy
        }
        for buoy in buoys
    ]

class NCFileInfo(TypedDict):
    """Type definition for NetCDF file information."""
    url: str
    buoy: str
    year: int

def fetch_nc_files_with_retry(source_url: str, buoy: str, dates: list[str], max_retries: int = 3) -> list[NCFileInfo]:
    """
    Fetch NetCDF files with retry logic and better error handling.

    Args:
        source_url: URL of the catalog page to scrape
        buoy: Buoy identifier
        dates: List of dates in 'YYYY-MM-DD' format and day should be 01.
        max_retries: Maximum number of retry attempts

    Returns:
        List of NCFileInfo dictionaries
    """
    for attempt in range(max_retries):
        try:
            return fetch_nc_files(source_url, buoy, dates)
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {buoy}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to fetch {buoy} after {max_retries} attempts")
                return []
        except Exception as e:
            logger.error(f"Unexpected error fetching {buoy}: {e}")
            return []

def fetch_nc_files(source_url: str, buoy: str = "APOLLO-BAY", dates: list[str] = ['2025-01-01']) -> list[NCFileInfo]:
    """
    Fetch all NetCDF file URLs from a catalog page.

    Args:
        source_url: URL of the catalog page to scrape
        buoy: Buoy identifier (default: "APOLLO-BAY")
        dates: List of dates in 'YYYY-MM-DD' format to filter files and day should be 01.

    Returns:
        List of NCFileInfo dictionaries containing URL, buoy, and year information
    """
    # Base URL template for accessing NetCDF files via OPeNDAP
    base_url = "https://thredds.aodn.org.au/thredds/dodsC/IMOS/COASTAL-WAVE-BUOYS/WAVE-BUOYS/REALTIME/WAVE-PARAMETERS/{buoy}/{year}/{name}"

    # Fetch the catalog page with timeout
    response = requests.get(source_url, timeout=30)
    response.raise_for_status()

    formatted_dates = [d.replace('-', '') for d in dates]
    year = datetime.strptime(dates[0], '%Y-%m-%d').year

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.select("a > tt")

    # Collect all NetCDF file URLs
    nc_urls = []
    for elem in elements:
        name = elem.text.strip()
        if name.endswith(".nc") and any(date in name for date in formatted_dates):
            full_url = base_url.format(buoy=buoy, year=year, name=name)
            nc_urls.append({"url": full_url, "buoy": buoy, "year": year})

    logger.info(f"Found {len(nc_urls)} NetCDF files for {buoy}")
    return nc_urls

def fetch_all_nc_files_per_year(dates: list[str], buoys: list, max_workers: int = 5) -> list[NCFileInfo]:
    """
    Get all NetCDF file URLs for given dates and buoys using concurrent processing.

    Args:
        dates: List of dates in 'YYYY-MM-DD' format and day should be 01.
        buoys: List of buoy identifiers
        max_workers: Maximum number of concurrent workers

    Returns:
        List of NCFileInfo dictionaries for all NetCDF files found
    """
    if not validate_dates(dates):
        return []

    # Generate catalog URLs for all buoys
    year = datetime.strptime(dates[0], '%Y-%m-%d').year
    url_pairs = generate_all_catalog_urls_per_year(year, buoys)
    nc_files = []

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_buoy = {
            executor.submit(fetch_nc_files_with_retry, pair["url"], pair["buoy"], dates): pair["buoy"]
            for pair in url_pairs
        }

        # Collect results as they complete
        for future in as_completed(future_to_buoy):
            buoy = future_to_buoy[future]
            try:
                result = future.result()
                if result:
                    nc_files.extend(result)
                    logger.info(f"Successfully processed {buoy}")
            except Exception as e:
                logger.error(f"Error processing {buoy}: {e}")

    logger.info(f"Total NetCDF files found: {len(nc_files)}")
    return nc_files

def process_dataset_safely(nc_file_info: dict, daily_data: dict) -> bool:
    """
    Safely process a single NetCDF dataset with proper resource management.

    Args:
        nc_file_info: Dictionary containing URL, buoy, and year info
        daily_data: Dictionary to store processed data

    Returns:
        True if processing succeeded, False otherwise
    """
    ds = None
    try:
        # Open the NetCDF dataset using xarray with chunking for memory efficiency
        ds = xr.open_dataset(nc_file_info['url'], chunks={'TIME': 100})

        # Extract latitude and longitude
        lat_val = ds.LATITUDE.values[0]
        lon_val = ds.LONGITUDE.values[0]

        lat = truncate(lat_val)
        lon = truncate(lon_val)

        if lat is None or lon is None:
            logger.warning(f"Invalid coordinates for {nc_file_info['buoy']}")
            return False

        # Extract time data and convert to dates
        times = pd.to_datetime(ds.TIME.values)

        # Group data by date
        for time_val in times:
            date_str = time_val.strftime('%Y-%m-%d')

            if date_str not in daily_data:
                daily_data[date_str] = []

            # Create feature for this buoy on this date
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]  # GeoJSON uses [longitude, latitude] order
                },
                "properties": {
                    "date": date_str,
                    "buoy": nc_file_info["buoy"],
                    "year": nc_file_info["year"],
                    "timestamp": time_val.isoformat()
                }
            }

            # Check if this buoy is already added for this date (avoid duplicates)
            existing_buoys = [f["properties"]["buoy"] for f in daily_data[date_str]]
            if nc_file_info["buoy"] not in existing_buoys:
                daily_data[date_str].append(feature)

        return True

    except Exception as e:
        logger.error(f"Error processing {nc_file_info['url']}: {e}")
        return False
    finally:
        # Ensure dataset is properly closed
        if ds is not None:
            try:
                ds.close()
            except:
                pass

def generate_buoy_locations_per_day(buoys: list, dates: list[str], output_dir: str = 'buoy_locations') -> None:
    """
    Get geographic locations for all specified buoys for each day and save as separate GeoJSON files.
    Creates one GeoJSON file per day containing all buoy locations for that day.

    Args:
        buoys: List of buoy identifiers
        dates: List of dates in 'YYYY-MM-DD' format and day should be 01.
        output_dir: Output directory for GeoJSON files

    Returns:
        None (saves GeoJSON files to disk)
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        return

    # Get all NetCDF file URLs for all buoys for the specified dates
    all_nc_files = fetch_all_nc_files_per_year(dates, buoys)

    if not all_nc_files:
        logger.warning(f"No NetCDF files found for dates {'-'.join(dates)}")
        return

    # Dictionary to store daily data: {date_string: [buoy_data, ...]}
    daily_data = {}

    # Process each NetCDF file with progress tracking
    successful_files = 0
    total_files = len(all_nc_files)

    for i, nc_file_info in enumerate(all_nc_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {nc_file_info['buoy']}")

        if process_dataset_safely(nc_file_info, daily_data):
            successful_files += 1

        # Memory cleanup every 10 files
        if i % 10 == 0:
            import gc
            gc.collect()

    logger.info(f"Successfully processed {successful_files}/{total_files} files")

    # Create and save GeoJSON files for each date
    files_created = 0
    for date_str, features in daily_data.items():
        if features:  # Only create file if there are features for this date
            try:
                # Create GeoJSON FeatureCollection
                geojson = {
                    "type": "FeatureCollection",
                    "metadata": {
                        "date": date_str,
                        "buoy_count": len(features),
                        "generated_at": datetime.now().isoformat()
                    },
                    "features": features
                }

                # Set filename for this date
                filename = f"buoy_locations_{date_str}.geojson"
                filepath = os.path.join(output_dir, filename)

                # Save GeoJSON to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(geojson, f, indent=2, ensure_ascii=False)

                files_created += 1
                logger.info(f"Created: {filepath} with {len(features)} buoys")

            except (OSError, json.JSONEncodeError) as e:
                logger.error(f"Error saving file for {date_str}: {e}")

    logger.info(f"Successfully created {files_created} daily GeoJSON files in '{output_dir}' directory")

def create_daily_geojson_safely(url: str, buoy: str, output_dir: str = 'buoy_details') -> Optional[list]:
    """
    Create GeoJSON with hourly time series data for each day with improved error handling.

    Args:
        url: URL of the NetCDF file
        buoy: Buoy identifier
        output_dir: Output directory

    Returns:
        List of created file info or None if failed
    """
    ds = None
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Open dataset with chunking for memory efficiency
        ds = xr.open_dataset(url, chunks={'TIME': 100})
        daily_groups = ds.groupby(ds.TIME.dt.date)
        created_files = []

        for date, daily_data in daily_groups:
            logger.info(f"Processing summary for {buoy} on {date}...")

            if len(daily_data.TIME) == 0:
                continue

            # Calculate daily statistics
            lat = float(daily_data.LATITUDE.values[0])
            lon = float(daily_data.LONGITUDE.values[0])

            # Group by hour and calculate hourly means
            hourly_groups = daily_data.groupby(daily_data.TIME.dt.hour)

            # Initialize properties with hourly data for each variable
            properties = {
                "date": str(date),
                "location": buoy,
                "records_count": int(len(daily_data.TIME)),
                "time_range": {
                    "start": str(daily_data.TIME.min().values),
                    "end": str(daily_data.TIME.max().values)
                }
            }

            # Add hourly time series for each data variable
            for var_name in daily_data.data_vars:
                hourly_data = []

                for hour, hourly_data_group in hourly_groups:
                    if len(hourly_data_group.TIME) > 0:
                        try:
                            # Get the first timestamp of this hour and convert to milliseconds
                            hour_timestamp = hourly_data_group.TIME.min().values
                            timestamp_ms = int(pd.to_datetime(hour_timestamp).timestamp() * 1000)

                            # Calculate hourly mean
                            hourly_mean = hourly_data_group[var_name].mean().values

                            # Handle NaN values
                            if not np.isnan(hourly_mean):
                                hourly_data.append([timestamp_ms, float(hourly_mean)])
                        except Exception as e:
                            logger.warning(f"Error processing hour {hour} for {var_name}: {e}")
                            continue

                # Sort by timestamp
                hourly_data.sort(key=lambda x: x[0])

                # Add to properties
                properties[var_name] = {
                    "name": var_name,
                    "data": hourly_data
                }

            # Create single feature with hourly time series
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": properties
            }

            geojson = {
                "type": "FeatureCollection",
                "metadata": {
                    "type": "daily_hourly_timeseries",
                    "date": str(date),
                    "location": buoy,
                    "generated_at": datetime.now().isoformat(),
                    "description": "Hourly averaged wave data"
                },
                "features": [feature]
            }

            # Save to file
            filename = f"{buoy}_{date.strftime('%Y-%m-%d')}.geojson"
            filepath = os.path.join(output_dir, filename)

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(geojson, f, indent=2, default=str, ensure_ascii=False)

                created_files.append({
                    'date': date,
                    'filename': filename,
                    'filepath': filepath,
                    'records': len(daily_data.TIME),
                    'hourly_points': sum(len(var_data.get('data', []))
                                         for var_data in properties.values()
                                         if isinstance(var_data, dict) and 'data' in var_data)
                })

            except (OSError, json.JSONEncodeError) as e:
                logger.error(f"Error saving file {filepath}: {e}")

        logger.info(f"✅ Created {len(created_files)} hourly time series GeoJSON files for {buoy}")
        return created_files

    except Exception as e:
        logger.error(f"Error processing dataset for {buoy}: {e}")
        return None
    finally:
        # Ensure dataset is properly closed
        if ds is not None:
            try:
                ds.close()
            except:
                pass

def generate_geojson_per_buoy_per_day_by_year(dates: list[str], buoys: list, max_workers: int = 3) -> None:
    """
    Generate GeoJSON files for each buoy for the specified dates with concurrent processing.

    Args:
        dates: List of dates in 'YYYY-MM-DD' format and day should be 01.
        buoys: List of buoy identifiers
        max_workers: Maximum number of concurrent workers
    """
    all_file_pairs = fetch_all_nc_files_per_year(dates, buoys)

    if not all_file_pairs:
        logger.warning("No NetCDF files found to process")
        return

    successful_files = 0
    total_files = len(all_file_pairs)

    # Process files with limited concurrency to manage memory
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {
            executor.submit(create_daily_geojson_safely, info['url'], info['buoy'], 'buoy_details'): info
            for info in all_file_pairs
        }

        for future in as_completed(future_to_info):
            file_info = future_to_info[future]
            try:
                result = future.result()
                if result is not None:
                    successful_files += 1
                    logger.info(f"Successfully processed {file_info['buoy']}")
            except Exception as e:
                logger.error(f"Error processing {file_info['buoy']}: {e}")

    logger.info(f"Successfully processed {successful_files}/{total_files} buoy datasets")

# Default buoy list
DEFAULT_BUOYS = [
    "APOLLO-BAY", "BENGELLO", "BOB", "BRIGHTON", "CAPE-BRIDGEWATER",
    "CEDUNA", "CENTRAL", "COLLAROY", "CORAL-BAY", "HILLARYS",
    "KARUMBA", "NORTH-KANGAROO-ISLAND", "OCEAN-BEACH", "ROBE",
    "SHARK-BAY", "STORM-BAY", "TANTABIDDI", "TATHRA", "TORBAY-WEST", "WOOLI"
]

def main():
    """Main function with improved argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        description="Generate wave buoy GeoJSON files for specified dates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wave_buoys_processing_script.py --output_base_dir ./output --dates 2025-01-01 2025-02-01
  python wave_buoys_processing_script.py --output_base_dir ./output --dates 2025-01-01 --buoys APOLLO-BAY BOB
        """
    )
    parser.add_argument(
        "--output_base_dir",
        type=Path,
        required=True,
        help="Directory where processed files will be saved.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        required=True,
        help="List of dates in YYYY-MM-DD format and day should be 01.",
    )
    parser.add_argument(
        "--buoys",
        nargs="*",
        default=DEFAULT_BUOYS,
        help="List of buoy identifiers (default: all available buoys).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of concurrent workers (default: 5).",
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

    # Validate inputs
    if not validate_dates(args.dates):
        logger.error("Invalid dates provided. Exiting.")
        return 1

    # Ensure output directory exists and is writable
    try:
        args.output_base_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(args.output_base_dir)
        logger.info(f"Working directory: {args.output_base_dir.absolute()}")
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot access output directory {args.output_base_dir}: {e}")
        return 1

    try:
        # Run processing functions
        logger.info("Starting buoy location processing...")
        generate_buoy_locations_per_day(args.buoys, args.dates)

        logger.info("Starting detailed buoy data processing...")
        generate_geojson_per_buoy_per_day_by_year(args.dates, args.buoys, args.max_workers)

        logger.info("Processing completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        return 1

if __name__ == "__main__":
    exit(main())