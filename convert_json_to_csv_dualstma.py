"""
convert_json_to_csv_dualstma.py

Converts METO-S2S JSON to frame-format CSV for DualSTMA.

JSON point format (verified):
  [0]  timestamp_ms
  [1]  LAT_norm
  [2]  LON_norm
  [3]  SOG_norm
  [4]  Heading_norm
  [5]  sailing_distance_norm
  [6]  vessel_type       (categorical)
  [7]  vessel_category   (categorical, 1-12)
  [8]  vessel_length     (binned, 1-20)
  [9]  vessel_width      (binned, 0-20)
  [10] LON_raw           (absolute degrees)
  [11] LAT_raw           (absolute degrees)
  [12] MMSI

Output CSV columns:
  frame_id, vessel_id,
  LON, LAT,                    # normalized
  SOG, Heading,                # normalized
  LON_abs, LAT_abs,            # absolute degrees
  vessel_type, vessel_length, vessel_width  # static features

Usage:
  python convert_json_to_csv_dualstma.py
"""

import os
import json
import pandas as pd

DATASET_DIR = "dataset/marinecadastre_2021"
SPLITS      = ["train", "val", "test"]

# JSON point field indices
IDX_TIMESTAMP    = 0
IDX_LAT_NORM     = 1
IDX_LON_NORM     = 2
IDX_SOG_NORM     = 3
IDX_HEAD_NORM    = 4
IDX_DIST_NORM    = 5
IDX_VESSEL_TYPE  = 6
IDX_VESSEL_CAT   = 7
IDX_VESSEL_LEN   = 8
IDX_VESSEL_WID   = 9
IDX_LON_RAW      = 10
IDX_LAT_RAW      = 11
IDX_MMSI         = 12


def load_json(split):
    path = os.path.join(DATASET_DIR, f"{split}.json")
    print(f"  Loading {path} ...")
    with open(path) as f:
        data = json.load(f)
    print(f"  {len(data)} trajectories")
    return data


def json_to_csv_dualstma(data, out_path):
    rows = []
    for traj in data:
        if not traj:
            continue
        mmsi = int(traj[0][IDX_MMSI])
        for point in traj:
            frame_id = int(point[IDX_TIMESTAMP]) // (10 * 60 * 1000)
            rows.append({
                "frame_id":      frame_id,
                "vessel_id":     mmsi,
                "LON":           point[IDX_LON_NORM],
                "LAT":           point[IDX_LAT_NORM],
                "SOG":           point[IDX_SOG_NORM],
                "Heading":       point[IDX_HEAD_NORM],
                "LON_abs":       point[IDX_LON_RAW],
                "LAT_abs":       point[IDX_LAT_RAW],
                "vessel_type":   int(point[IDX_VESSEL_TYPE]),
                "vessel_length": int(point[IDX_VESSEL_LEN]),
                "vessel_width":  int(point[IDX_VESSEL_WID]),
            })

    df = (pd.DataFrame(rows)
            .sort_values(["frame_id", "vessel_id"])
            .reset_index(drop=True))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"  Rows:    {len(df):,}")
    print(f"  Vessels: {df['vessel_id'].nunique():,}")
    print(f"  Saved:   {out_path}")
    return df


def main():
    print("=" * 60)
    print("METO-S2S JSON → DualSTMA CSV Converter")
    print("=" * 60)

    for split in SPLITS:
        print(f"\n  [{split.upper()}]")
        data = load_json(split)
        out_csv = os.path.join(
            DATASET_DIR, f"dualstma_{split}", f"day_{split}.csv"
        )
        json_to_csv_dualstma(data, out_csv)

    print("\nDone!")
    print("=" * 60)
    print("Next step:")
    print("  python train_dualstma.py --dataset marinecadastre_2021")
    print("=" * 60)


if __name__ == "__main__":
    main()
