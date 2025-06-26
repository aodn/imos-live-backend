import math
from typing import TypedDict
import xarray as xr
import datetime
import json
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def truncate(value, decimals=2):
    """
    Truncate a float value to a specified number of decimal places.

    Args:
        value: The float value to truncate
        decimals: Number of decimal places to keep (default: 2)

    Returns:
        Truncated float value
    """
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


def fetch_nc_files(source_url: str, buoy: str = "APOLLO-BAY", year: int = 2025) -> list[NCFileInfo]:
    """
    Fetch all NetCDF file URLs from a catalog page.

    Args:
        source_url: URL of the catalog page to scrape
        buoy: Buoy identifier (default: "APOLLO-BAY")
        year: Year for the data (default: 2025)

    Returns:
        List of NCFileInfo dictionaries containing URL, buoy, and year information
    """
    import requests
    from bs4 import BeautifulSoup

    # Base URL template for accessing NetCDF files via OPeNDAP
    base_url = "https://thredds.aodn.org.au/thredds/dodsC/IMOS/COASTAL-WAVE-BUOYS/WAVE-BUOYS/REALTIME/WAVE-PARAMETERS/{buoy}/{year}/{name}"

    # Fetch the catalog page
    response = requests.get(source_url)
    response.raise_for_status()

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.select("a > tt")

    # Collect all NetCDF file URLs
    nc_urls = []
    for elem in elements:
        name = elem.text.strip()
        if name.endswith(".nc"):
            full_url = base_url.format(buoy=buoy, year=year, name=name)
            nc_urls.append({"url": full_url, "buoy": buoy, "year": year})

    return nc_urls


def fetch_all_nc_files_per_year(year: int, buoys: list) -> list[NCFileInfo]:
    """
    Get all NetCDF file URLs for a given year and list of buoys.
    Each buoy will have all available NetCDF files for each month.

    Args:
        year: The year to fetch data for
        buoys: List of buoy identifiers

    Returns:
        List of NCFileInfo dictionaries for all NetCDF files found
    """
    # Generate catalog URLs for all buoys
    url_pairs = generate_all_catalog_urls_per_year(year, buoys)
    nc_files = []

    # Fetch NetCDF files from each buoy's catalog
    for url_pair in url_pairs:
        try:
            nc_file = fetch_nc_files(url_pair["url"], buoy=url_pair["buoy"], year=year)
            if nc_file:
                nc_files.extend(nc_file)
        except Exception as e:
            print(f"Error fetching from {url_pair}: {e}")

    return nc_files


def generate_buoy_locations_per_day(buoys: list, year: int = 2025, output_dir='buoy_locations') -> None:
    """
    Get geographic locations for all specified buoys for each day and save as separate GeoJSON files.
    Creates one GeoJSON file per day containing all buoy locations for that day.

    Args:
        buoys: List of buoy identifiers
        year: Year to fetch data from (default: 2025)
        output_dir: Output directory for GeoJSON files (default: 'buoy_locations_geojson')

    Returns:
        None (saves GeoJSON files to disk)
    """
    import json
    import os
    from datetime import datetime
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # Get all NetCDF file URLs for all buoys for the year
    all_nc_files = fetch_all_nc_files_per_year(year, buoys)

    if not all_nc_files:
        print(f"No NetCDF files found for year {year}")
        return

    # Dictionary to store daily data: {date_string: [buoy_data, ...]}
    daily_data = {}

    # Process each NetCDF file
    for nc_file_info in all_nc_files:
        try:
            # Open the NetCDF dataset using xarray
            ds = xr.open_dataset(nc_file_info['url'])

            # Extract latitude and longitude
            lat = truncate(ds.LATITUDE.values[0])
            lon = truncate(ds.LONGITUDE.values[0])

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
                        "year": year,
                        "timestamp": time_val.isoformat()
                    }
                }

                # Check if this buoy is already added for this date (avoid duplicates)
                existing_buoys = [f["properties"]["buoy"] for f in daily_data[date_str]]
                if nc_file_info["buoy"] not in existing_buoys:
                    daily_data[date_str].append(feature)

            # Close the dataset to free memory
            ds.close()

        except Exception as e:
            print(f"Error processing {nc_file_info['url']}: {e}")
            continue

    # Create and save GeoJSON files for each date
    files_created = 0
    for date_str, features in daily_data.items():
        if features:  # Only create file if there are features for this date
            # Create GeoJSON FeatureCollection
            geojson = {
                "type": "FeatureCollection",
                "metadata": {
                    "date": date_str,
                    "buoy_count": len(features)
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
            print(f"Created: {filepath} with {len(features)} buoys")

    print(f"Successfully created {files_created} daily GeoJSON files in '{output_dir}' directory")


def create_daily_geojson(url,buoy, output_dir='daily_buoy_details'):
    """
    Create GeoJSON with hourly time series data for each day
    """
    os.makedirs(output_dir, exist_ok=True)

    ds=xr.open_dataset(url)
    daily_groups = ds.groupby(ds.TIME.dt.date)
    created_files = []

    for date, daily_data in daily_groups:
        print(f"Processing summary for {date}...")

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
                    # Get the first timestamp of this hour and convert to milliseconds
                    hour_timestamp = hourly_data_group.TIME.min().values
                    timestamp_ms = int(pd.to_datetime(hour_timestamp).timestamp() * 1000)

                    # Calculate hourly mean
                    hourly_mean = hourly_data_group[var_name].mean().values

                    # Handle NaN values
                    if not np.isnan(hourly_mean):
                        hourly_data.append([timestamp_ms, float(hourly_mean)])

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
                "generated_at": datetime.datetime.now().isoformat(),
                "description": "Hourly averaged wave data"
            },
            "features": [feature]
        }

        # Save to file
        filename = f"{buoy}_{date.strftime('%Y-%m-%d')}.geojson"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)

        created_files.append({
            'date': date,
            'filename': filename,
            'filepath': filepath,
            'records': len(daily_data.TIME),
            'hourly_points': len([item for var_data in properties.values()
                                  if isinstance(var_data, dict) and 'data' in var_data
                                  for item in var_data['data']])
        })



    print(f"\n✅ Created {len(created_files)} hourly time series GeoJSON files")
    return created_files


def generate_geojson_per_buoy_per_day_by_year(year: int, buoys) -> None:
    """
    Generate GeoJSON files for each buoy for the specified year.
    """
    all_file_paris = fetch_all_nc_files_per_year(year, buoys)
    for file_info in all_file_paris:
        create_daily_geojson(file_info['url'], file_info['buoy'], 'daily_buoy_details')

buoys = [
    "APOLLO-BAY",
    "BENGELLO",
    "BOB",
    "BRIGHTON",
    "CAPE-BRIDGEWATER",
    "CEDUNA",
    "CENTRAL",
    "COLLAROY",
    "CORAL-BAY",
    "HILLARYS",
    "KARUMBA",
    "NORTH-KANGAROO-ISLAND",
    "OCEAN-BEACH",
    "ROBE",
    "SHARK-BAY",
    "STORM-BAY",
    "TANTABIDDI",
    "TATHRA",
    "TORBAY-WEST",
    "WOOLI"
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate wave buoy GeoJSON files for specified dates."
    )
    parser.add_argument(
        "--output_base_dir",
        type=Path,
        required=True,
        help="Directory where processed files will be saved.",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        required=True,
        help="List of dates in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    # Only expect one year from the dates, the reason defined as list of dates is to make it consistent with gsla_preprocessing_script.py.
    year = int(args.dates[0].split('-')[0])

    # Change to the output directory
    os.chdir(args.output_base_dir)

    # Run your existing functions
    generate_buoy_locations_per_day(buoys, year)
    generate_geojson_per_buoy_per_day_by_year(year, buoys)
