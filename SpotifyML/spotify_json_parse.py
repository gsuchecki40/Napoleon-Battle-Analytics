import pandas as pd
import json
import glob
import os

def load_spotify_history(folder_path):
    # Grab all streaming history JSON files
    files = glob.glob(os.path.join(folder_path, "Streaming_History_Audio*.json"))
    
    dfs = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dfs.append(pd.DataFrame(data))
    
    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_data(df):
    # Rename columns to something sensible
    df = df.rename(columns={
        'ts': 'timestamp',
        'ms_played': 'ms_played',
        'master_metadata_track_name': 'track',
        'master_metadata_album_artist_name': 'artist',
        'master_metadata_album_album_name': 'album',
        'reason_start': 'reason_start',
        'reason_end': 'reason_end',
    })
    
    # Drop podcasts
    df = df[df['track'].notna()]
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create skip label
    df['skipped'] = df['reason_end'].isin(['fwdbtn', 'endplay']).astype(int)
    
    # Drop plays under 10 seconds (accidental clicks)
    df = df[df['ms_played'] > 10000]
    
    return df

# Run it
df = load_spotify_history('path/to/your/spotify/data')
df = clean_data(df)

print(df.shape)
print(df.head())
print(df['skipped'].value_counts())