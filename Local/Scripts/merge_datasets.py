#!/usr/bin/env python3
"""
Data Merger Script: Merge Brahimi's original Excel data with ENSIA GPS dataset

This script performs the following operations:
1. Creates a backup copy of ensia_gps_data.csv as ensia_gps_data_v1.csv
2. Converts brahimi_original.xlsx to match the ENSIA GPS data format
3. Appends the converted data to ensia_gps_data.csv

Author: Generated for ENSIA GPS Localization Project
Date: 2026-03-31
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import numpy as np

def create_backup():
    """Create a backup copy of the original ensia_gps_data.csv"""
    source_path = "../Data/ensia_gps_data .csv"
    backup_path = "../Data/ensia_gps_data_v1.csv"

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Copy the file
    with open(source_path, 'rb') as src, open(backup_path, 'wb') as dst:
        dst.write(src.read())

    print(f"✓ Backup created: {backup_path}")
    return backup_path

def load_brahimi_data():
    """Load and examine the Brahimi Excel data"""
    excel_path = "../Data/brahimi_original.xlsx"

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Brahimi Excel file not found: {excel_path}")

    df_brahimi = pd.read_excel(excel_path)
    print(f"✓ Loaded Brahimi data: {df_brahimi.shape[0]} rows, {df_brahimi.shape[1]} columns")
    print(f"  Columns: {df_brahimi.columns.tolist()}")
    print(f"  Amphitheater distribution:\n{df_brahimi['name'].value_counts()}")

    return df_brahimi

def load_ensia_data():
    """Load the main ENSIA GPS dataset"""
    csv_path = "../Data/ensia_gps_data .csv"
    df_ensia = pd.read_csv(csv_path)
    print(f"✓ Loaded ENSIA data: {df_ensia.shape[0]} rows, {df_ensia.shape[1]} columns")
    return df_ensia

def convert_brahimi_to_ensia_format(df_brahimi, df_ensia):
    """
    Convert Brahimi Excel data to match ENSIA GPS CSV format

    Mapping:
    - name → Amphi (with renaming: "Amphitheater X" → "Amphi X")
    - location_lat → Lat_Mean
    - location_lng → Lng_Mean
    """

    # Get the next available ID
    max_id = df_ensia['ID'].max()
    new_ids = range(max_id + 1, max_id + 1 + len(df_brahimi))

    # Create base timestamp - use current time minus some offset for synthetic data
    base_timestamp = datetime.now() - timedelta(days=365)  # One year ago

    # Convert amphitheater names
    def convert_amphi_name(name):
        if pd.isna(name):
            return "Outside"
        # Extract number from "Amphitheater X"
        if "Amphitheater" in str(name):
            num = str(name).replace("Amphitheater", "").strip()
            return f"Amphi {num}"
        return str(name)

    # Create synthetic JSON data
    def create_synthetic_raw_readings(lat, lng, num_readings=5):
        """Create synthetic raw GPS readings (redundant - all identical)"""
        readings = []
        base_time = int(datetime.now().timestamp() * 1000)  # Current time in ms

        for i in range(num_readings):
            # No noise added - all readings are identical (redundant)
            reading = {
                "latitude": float(lat),
                "longitude": float(lng),
                "accuracy_m": float(np.random.uniform(5, 20)),  # 5-20m accuracy
                "timestamp": str(datetime.fromtimestamp((base_time + i * 1000) / 1000).isoformat()) + "Z"
            }
            readings.append(reading)

        return readings

    def create_synthetic_metadata(num_readings):
        """Create synthetic metadata"""
        return {
            "number_of_readings": num_readings,
            "collection_duration_seconds": num_readings * 2  # 2 seconds per reading
        }

    def create_synthetic_navigator_context():
        """Create synthetic navigator context"""
        platforms = ["Linux aarch64", "iPhone", "Win32", "MacIntel"]
        return {
            "platform": str(np.random.choice(platforms)),
            "user_agent": "Mozilla/5.0 (Synthetic Data)",
            "device_memory_gb": int(np.random.choice([4, 8, 16])),
            "max_touch_points": int(np.random.choice([0, 5])),
            "hardware_concurrency": int(np.random.choice([4, 8, 16]))
        }

    def create_synthetic_screen_context():
        """Create synthetic screen context"""
        widths = [390, 412, 1280]
        heights = [844, 915, 752]
        screen_widths = [390, 412, 1280]
        screen_heights = [844, 915, 800]
        ratios = [2.0, 2.625, 1.5]

        return {
            "avail_width": int(np.random.choice(widths)),
            "color_depth": 24,
            "avail_height": int(np.random.choice(heights)),
            "screen_width": int(np.random.choice(screen_widths)),
            "screen_height": int(np.random.choice(screen_heights)),
            "device_pixel_ratio": float(np.random.choice(ratios))
        }

    def create_synthetic_network_info():
        """Create synthetic network info"""
        return {
            "rtt_ms": int(np.random.choice([50, 100, 300])),
            "save_data": False,
            "downlink_mbps": float(np.random.uniform(1, 10)),
            "effective_type": str(np.random.choice(["4g", "3g", "wifi"])),
            "connection_type": str(np.random.choice(["cellular", "wifi"]))
        }

    def create_synthetic_battery_status():
        """Create synthetic battery status"""
        return {
            "is_charging": bool(np.random.choice([True, False])),
            "battery_level": float(np.random.uniform(0.1, 1.0))
        }

    # Create the converted dataframe
    converted_data = []

    for idx, row in df_brahimi.iterrows():
        # Generate synthetic data
        num_readings = np.random.randint(3, 10)
        raw_readings = create_synthetic_raw_readings(row['location_lat'], row['location_lng'], num_readings)

        # Calculate statistics from synthetic readings
        accuracies = [r['accuracy_m'] for r in raw_readings]
        acc_mean = np.mean(accuracies)
        acc_variance = np.var(accuracies) if len(accuracies) > 1 else 0.0

        record = {
            'ID': new_ids[idx],
            'Timestamp': (base_timestamp + timedelta(hours=idx)).isoformat() + '+00:00',
            'Year': np.random.choice(['1', '2', '3', '4', '5']),  # Random academic year
            'Amphi': convert_amphi_name(row['name']),
            'Module': None,  # No module info in Brahimi data
            'Block': str(np.random.choice(['Left', 'Center', 'Right', None])),  # Random seating
            'Row': int(np.random.randint(1, 10)) if np.random.random() > 0.3 else None,  # 70% have row
            'Column': int(np.random.randint(1, 15)) if np.random.random() > 0.3 else None,  # 70% have column
            'Lat_Mean': float(row['location_lat']),
            'Lng_Mean': float(row['location_lng']),
            'Acc_Mean': float(acc_mean),
            'Variance': float(acc_variance) if acc_variance > 0 else None,
            'IsOutside': False,  # Assume all are inside amphitheaters
            'RawReadings': json.dumps(raw_readings),
            'Metadata': json.dumps(create_synthetic_metadata(num_readings)),
            'NavigatorContext': json.dumps(create_synthetic_navigator_context()),
            'ScreenContext': json.dumps(create_synthetic_screen_context()),
            'NetworkInfo': json.dumps(create_synthetic_network_info()),
            'BatteryStatus': json.dumps(create_synthetic_battery_status())
        }

        converted_data.append(record)

    df_converted = pd.DataFrame(converted_data)
    print(f"✓ Converted {len(df_converted)} Brahimi records to ENSIA format")
    print(f"  Amphitheater distribution after conversion:\n{df_converted['Amphi'].value_counts()}")

    return df_converted

def merge_and_save(df_ensia, df_converted):
    """Merge the converted data with the main dataset and save"""
    # Combine the dataframes
    df_merged = pd.concat([df_ensia, df_converted], ignore_index=True)

    # Sort by ID to maintain order
    df_merged = df_merged.sort_values('ID').reset_index(drop=True)

    # Save back to the original file
    output_path = "../Data/ensia_gps_data .csv"
    df_merged.to_csv(output_path, index=False)

    print(f"✓ Merged data saved to {output_path}")
    print(f"  Original records: {len(df_ensia)}")
    print(f"  Added records: {len(df_converted)}")
    print(f"  Total records: {len(df_merged)}")

    return df_merged

def main():
    """Main execution function"""
    print("🚀 Starting ENSIA GPS Data Merger")
    print("=" * 50)

    try:
        # Step 1: Create backup
        print("\n1. Creating backup...")
        backup_path = create_backup()

        # Step 2: Load data
        print("\n2. Loading data...")
        df_ensia = load_ensia_data()
        df_brahimi = load_brahimi_data()

        # Step 3: Convert Brahimi data
        print("\n3. Converting Brahimi data to ENSIA format...")
        df_converted = convert_brahimi_to_ensia_format(df_brahimi, df_ensia)

        # Step 4: Merge and save
        print("\n4. Merging and saving...")
        df_final = merge_and_save(df_ensia, df_converted)

        # Step 5: Final summary
        print("\n5. Final Summary:")
        print(f"   📊 Total records: {len(df_final)}")
        print(f"   🏫 Amphitheater distribution:\n{df_final['Amphi'].value_counts()}")
        print(f"   📁 Backup saved as: {backup_path}")
        print(f"   💾 Merged data saved as: ../Data/ensia_gps_data .csv")

        print("\n✅ Data merger completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()