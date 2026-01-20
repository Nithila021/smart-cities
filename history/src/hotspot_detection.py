import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime
import folium
from folium.plugins import HeatMap

# Load data
df = pd.read_csv('NYPD_Complaint_Data_YTD.csv')

# Convert dates and times
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'])
df['cmplnt_fr_tm'] = pd.to_datetime(df['cmplnt_fr_tm'], format='%H:%M:%S').dt.time
df['hour'] = df['cmplnt_fr_tm'].apply(lambda x: x.hour)
df['day_of_week'] = df['cmplnt_fr_dt'].dt.dayofweek
df['month'] = df['cmplnt_fr_dt'].dt.month

# Spatial clustering for hotspots
coords = df[['latitude', 'longitude']].dropna().values
kms_per_radian = 6371.0088
epsilon = 0.5 / kms_per_radian  # 0.5km radius

db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
df['cluster'] = db.labels_

# Filter hotspot clusters (excluding noise points labeled -1)
hotspots = df[df['cluster'] != -1].groupby('cluster').agg({
    'latitude': 'mean',
    'longitude': 'mean',
    'cmplnt_num': 'count',
    'ofns_desc': lambda x: x.mode()[0],
    'law_cat_cd': lambda x: x.mode()[0]
}).rename(columns={'cmplnt_num': 'crime_count'})

# Visualize hotspots
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
HeatMap(data=df[['latitude', 'longitude']].dropna().values.tolist(), radius=15).add_to(m)
m.save('crime_hotspots.html')


#TEMPORAL ANALYSIS
# Temporal patterns in hotspots
hotspot_df = df[df['cluster'] != -1].copy()

# Hourly patterns
hourly_patterns = hotspot_df.groupby(['cluster', 'hour']).size().unstack().fillna(0)

# Weekly patterns
weekly_patterns = hotspot_df.groupby(['cluster', 'day_of_week']).size().unstack().fillna(0)

# Seasonal patterns
monthly_patterns = hotspot_df.groupby(['cluster', 'month']).size().unstack().fillna(0)

#VICTIM DEMOGRAPHIC ANALYSIS
# Analyze victim demographics in hotspots
victim_patterns = hotspot_df.groupby(['cluster', 'vic_age_group', 'vic_race', 'vic_sex']).agg({
    'cmplnt_num': 'count',
    'ofns_desc': lambda x: x.value_counts().index[0]
}).rename(columns={'cmplnt_num': 'count', 'ofns_desc': 'most_common_offense'})

# Example: Elderly victim analysis
elderly_crimes = hotspot_df[hotspot_df['vic_age_group'] == '65+'].groupby('cluster').agg({
    'cmplnt_num': 'count',
    'ofns_desc': lambda x: x.mode()[0],
    'prem_typ_desc': lambda x: x.mode()[0]
})

# Female victim analysis
female_crimes = hotspot_df[hotspot_df['vic_sex'] == 'F'].groupby('cluster').agg({
    'cmplnt_num': 'count',
    'ofns_desc': lambda x: x.mode()[0],
    'hour': lambda x: x.mode()[0]
})

