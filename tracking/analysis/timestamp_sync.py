#%%
# 
import pandas as pd
from datetime import datetime

# File paths
tracking_file = '/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/tracking/data/2025-07-29_18-14-24:3699_basler_tracking_markers.csv'
robot_file = '/home/alberto/Documents/ActiveSensingCollectives_lab/Ro-BATs/tracking/data/TIMESTAMPS_134.34.226.238_2025-07-29__18-14-05.csv'

# Load CSVs
tracking_df = pd.read_csv(tracking_file)
robot_df = pd.read_csv(robot_file)

# Helper to parse timestamps using posix (UNIX) timestamps
def parse_ts(ts):
    # If ts is already a float/int, just return as datetime
    try:
        return datetime.fromtimestamp(float(ts))
    except Exception:
        raise ValueError(f"Cannot parse POSIX timestamp: {ts}")

# Extract timestamp columns (adjust these to your actual column names)
tracking_df['parsed_ts'] = tracking_df.iloc[:,0].apply(parse_ts)
robot_df['parsed_ts'] = robot_df.iloc[:,0].apply(parse_ts)
#%% 

# Find closest matches
results = []
for idx, row in tracking_df.iterrows():
    ts = row['parsed_ts']
    # Find the closest timestamp in timestamps_df
    diffs = (robot_df['parsed_ts'] - ts).abs()
    min_idx = diffs.idxmin()
    closest_ts = robot_df.loc[min_idx, 'parsed_ts']
    # Adjust for GMT-2 (i.e., subtract 2 hours from both timestamps)
    ts_gmt2 = ts - pd.Timedelta(hours=2)
    closest_ts_gmt2 = closest_ts - pd.Timedelta(hours=2)
    results.append({
        'idx': idx,
        'tracking_ts_unix': ts_gmt2.timestamp(),
        'tracking_ts_gmt2': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'closest_timestamp_idx': min_idx,
        'closest_robot_ts_unix': closest_ts_gmt2.timestamp(),
        'closest_robot_ts_gmt2': closest_ts.strftime('%Y-%m-%d %H:%M:%S'),
        'time_difference': abs((ts_gmt2 - closest_ts_gmt2).total_seconds())
    })

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('closest_timestamps.csv', index=False)
print("Saved closest timestamp matches to closest_timestamps.csv")
# %%
