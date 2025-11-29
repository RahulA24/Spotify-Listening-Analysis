import pandas as pd
import json
import os
import glob
from typing import Optional

def load_data(raw_data_path: str) -> pd.DataFrame:
    """
    General ETL loader for Spotify data.
    Works with both the older “Standard” history export and the newer “Extended” format.
    """
    print(f"[INFO] Starting ETL. Reading files from: {raw_data_path}")
    
    files = glob.glob(os.path.join(raw_data_path, "*.json"))
    if not files:
        raise FileNotFoundError(f"[ERROR] No JSON files found in {raw_data_path}. Check the path again.")
    
    data_frames = []
    
    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Some exports contain a single dict instead of a list. Normalize it.
                if isinstance(data, dict):
                    data = [data]

                data_frames.append(pd.DataFrame(data))
        except Exception as e:
            print(f"[WARNING] Could not read {file}: {e}")
    
    # Merge all JSON files into one DataFrame
    df = pd.concat(data_frames, ignore_index=True)
    
    # --- FORMAT DETECTION LOGIC ---
    
    # Standard format: endTime, artistName, trackName, msPlayed
    if "artistName" in df.columns:
        print("[INFO] Identified Standard Format. Applying column renaming.")
        df = df.rename(columns={
            "endTime": "ts",
            "artistName": "master_metadata_album_artist_name",
            "trackName": "master_metadata_track_name",
            "msPlayed": "ms_played"
        })
    
    # Extended format already uses the newer naming scheme
    elif "master_metadata_album_artist_name" in df.columns:
        print("[INFO] Identified Extended Format. Keeping original column names.")
    
    # --- CLEANUP AND FIXING TYPES ---
    
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    
    df["master_metadata_track_name"] = df["master_metadata_track_name"].fillna("Unknown Track")
    df["master_metadata_album_artist_name"] = df["master_metadata_album_artist_name"].fillna("Unknown Artist")
    
    # --- FILTERING OUT NOISE ---
    
    initial_count = len(df)

    # Both formats contain ms_played, so filter short or noisy plays
    if "ms_played" in df.columns:
        
        if "skipped" in df.columns:
            # Extended format: trust the skip flag when available
            df["skipped"] = df["skipped"].fillna(False).astype(bool)
            df = df[(df["ms_played"] > 5000) | (df["skipped"] == True)]
        
        else:
            # Standard format: no skip flag available, so remove very short plays
            df = df[df["ms_played"] > 10000]
    
    filtered_rows = initial_count - len(df)
    print(f"[SUCCESS] ETL finished. Final rows: {len(df)} (Filtered out {filtered_rows} short/noisy records).")
    
    return df
