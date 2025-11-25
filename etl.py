#!/usr/bin/env python3
"""
Pittsburgh Transit Atlas - ETL Pipeline
Transforms raw POGOH and PRT data into integration metrics for visualization
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

print("=" * 80)
print("PITTSBURGH TRANSIT ATLAS - ETL PIPELINE")
print("=" * 80)

# ============================================================================
# 1. DATA INGESTION
# ============================================================================

print("\n[STEP 1] INGESTING DATA...")

# Load POGOH Station Locations
print("  → Loading POGOH station metadata...")
stations = pd.read_excel('dataset/pogoh-station-locations-october-2025.xlsx')
print(f"    ✓ Loaded {len(stations)} bike stations")

# Load all POGOH trip data (Nov 2024 - Oct 2025)
print("  → Loading POGOH trip data (12 months)...")
trip_files = [
    'november-2024.xlsx', 'december-2024.xlsx',
    'january-2025.xlsx', 'february-2025.xlsx', 'march-2025.xlsx',
    'april-2025.xlsx', 'may-2025.xlsx', 'june-2025.xlsx',
    'july-2025.xlsx', 'august-2025.xlsx', 'september-2025.xlsx',
    'pogoh-october-2025.xlsx'
]

trips_list = []
for file in trip_files:
    df = pd.read_excel(f'dataset/{file}')
    trips_list.append(df)
    print(f"    • {file}: {len(df):,} trips")

trips = pd.concat(trips_list, ignore_index=True)
print(f"    ✓ Total trips loaded: {len(trips):,}")

# Load PRT Bus Stop Data
print("  → Loading PRT bus stop usage data...")
bus_stops = pd.read_csv('dataset/PRT_Bus_Stop_Usage_Unweighted_-5413281589865626035.csv', encoding='utf-8-sig')
print(f"    ✓ Loaded {len(bus_stops)} bus stops")

# ============================================================================
# 2. DATA CLEANING
# ============================================================================

print("\n[STEP 2] CLEANING DATA...")

# Clean POGOH Trips
print("  → Cleaning POGOH trip data...")
trips['Start Date'] = pd.to_datetime(trips['Start Date'])
trips['End Date'] = pd.to_datetime(trips['End Date'])

# Filter outliers (Duration > 4 hours = 14,400 seconds)
initial_count = len(trips)
trips = trips[trips['Duration'] <= 14400]
print(f"    • Filtered {initial_count - len(trips):,} trips > 4hrs")
print(f"    • Remaining trips: {len(trips):,}")

# Add temporal features
trips['month'] = trips['Start Date'].dt.month
trips['hour'] = trips['Start Date'].dt.hour
trips['day_of_week'] = trips['Start Date'].dt.dayofweek
trips['year_month'] = trips['Start Date'].dt.to_period('M')
trips['day_name'] = trips['Start Date'].dt.day_name()

# Add Campus Corridor Flag (CMU/Pitt Bounding Box)
print("  → Flagging Campus Corridor trips...")
CAMPUS_LAT_MIN, CAMPUS_LAT_MAX = 40.435, 40.450
CAMPUS_LON_MIN, CAMPUS_LON_MAX = -79.970, -79.940

# Merge with station coordinates
trips = trips.merge(
    stations[['Id', 'Latitude', 'Longitude']],
    left_on='Start Station Id',
    right_on='Id',
    how='left',
    suffixes=('', '_start')
).rename(columns={'Latitude': 'start_lat', 'Longitude': 'start_lon'})

trips = trips.merge(
    stations[['Id', 'Latitude', 'Longitude']],
    left_on='End Station Id',
    right_on='Id',
    how='left',
    suffixes=('', '_end')
).rename(columns={'Latitude': 'end_lat', 'Longitude': 'end_lon'})

# Calculate displacement (straight-line distance in meters)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula"""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

trips['displacement'] = haversine_distance(
    trips['start_lat'], trips['start_lon'],
    trips['end_lat'], trips['end_lon']
)

# Flag Campus Corridor trips (start OR end in campus area)
trips['is_campus'] = (
    ((trips['start_lat'] >= CAMPUS_LAT_MIN) & (trips['start_lat'] <= CAMPUS_LAT_MAX) &
     (trips['start_lon'] >= CAMPUS_LON_MIN) & (trips['start_lon'] <= CAMPUS_LON_MAX)) |
    ((trips['end_lat'] >= CAMPUS_LAT_MIN) & (trips['end_lat'] <= CAMPUS_LAT_MAX) &
     (trips['end_lon'] >= CAMPUS_LON_MIN) & (trips['end_lon'] <= CAMPUS_LON_MAX))
)

campus_count = trips['is_campus'].sum()
print(f"    ✓ Campus Corridor trips identified: {campus_count:,} ({campus_count/len(trips)*100:.1f}%)")
print(f"    ✓ Added temporal and spatial features")

# Clean Bus Stops
print("  → Cleaning bus stop data...")
# Keep only relevant columns
ridership_cols = [col for col in bus_stops.columns if col.startswith('R_W_')]
bus_stops_clean = bus_stops[['stop_id', 'stop_name', 'latitude', 'longitude', 'HOOD'] + ridership_cols].copy()

# Remove duplicates (keep first occurrence)
bus_stops_clean = bus_stops_clean.drop_duplicates(subset='stop_id', keep='first')
print(f"    ✓ Cleaned to {len(bus_stops_clean)} unique stops")

# ============================================================================
# 3. INTEGRATION INDEX CALCULATION
# ============================================================================

print("\n[STEP 3] CALCULATING INTEGRATION INDEX...")

# Prepare bus stop coordinates
bus_coords = bus_stops_clean[['latitude', 'longitude']].values
station_coords = stations[['Latitude', 'Longitude']].values

# Calculate distance from each bus stop to nearest bike station
print("  → Computing spatial distances...")
dist_matrix = distance_matrix(bus_coords, station_coords)
nearest_station_dist = dist_matrix.min(axis=1)  # in degrees, need to convert

# Convert to meters (rough approximation: 1 degree ≈ 111,000 meters at this latitude)
nearest_station_dist_meters = nearest_station_dist * 111000

# Apply minimum distance floor of 10 meters
nearest_station_dist_meters = np.maximum(nearest_station_dist_meters, 10)

bus_stops_clean['nearest_dock_distance'] = nearest_station_dist_meters

# Count bike trips within 400m buffer for each bus stop
print("  → Counting bike trips within 400m buffer...")
bike_trips_nearby = []

for idx, bus_stop in bus_stops_clean.iterrows():
    bus_lat = bus_stop['latitude']
    bus_lon = bus_stop['longitude']

    # Calculate distance to all stations
    distances = haversine_distance(
        bus_lat, bus_lon,
        stations['Latitude'].values,
        stations['Longitude'].values
    )

    # Find stations within 400m
    nearby_station_ids = stations.loc[distances <= 400, 'Id'].values

    # Count trips starting from nearby stations
    trip_count = len(trips[trips['Start Station Id'].isin(nearby_station_ids)])
    bike_trips_nearby.append(trip_count)

bus_stops_clean['bike_trips_400m'] = bike_trips_nearby

# Get latest ridership (May 2025 - R_W_202505)
# R_W values are in thousands of boardings
latest_ridership_col = 'R_W_202505'
bus_stops_clean['bus_boardings'] = bus_stops_clean[latest_ridership_col].fillna(0) * 1000

# Calculate Integration Index
# Formula: I_b = (Bus_Boardings * log(Bike_Trips + 1)) / max(Distance, 10)
bus_stops_clean['integration_index'] = (
    bus_stops_clean['bus_boardings'] *
    np.log(bus_stops_clean['bike_trips_400m'] + 1) /
    bus_stops_clean['nearest_dock_distance']
)

print(f"    ✓ Integration Index calculated for {len(bus_stops_clean)} stops")
print(f"    • Mean Index: {bus_stops_clean['integration_index'].mean():.4f}")
print(f"    • Max Index: {bus_stops_clean['integration_index'].max():.4f}")

# ============================================================================
# 4. TRIP ARCHETYPES (K-MEANS CLUSTERING)
# ============================================================================

print("\n[STEP 4] CLUSTERING TRIP ARCHETYPES...")

# Prepare features for clustering
trips_clean = trips.dropna(subset=['displacement', 'Duration', 'hour'])

# Normalize features for K-Means
from sklearn.preprocessing import StandardScaler
features = trips_clean[['Duration', 'displacement', 'hour']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform K-Means clustering (4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
trips_clean['archetype'] = kmeans.fit_predict(features_scaled)

# Analyze clusters
print("  → Cluster analysis:")
archetype_labels = []
for i in range(4):
    cluster_trips = trips_clean[trips_clean['archetype'] == i]
    avg_duration = cluster_trips['Duration'].mean()
    avg_displacement = cluster_trips['displacement'].mean()
    avg_hour = cluster_trips['hour'].mean()
    weekend_pct = (cluster_trips['day_of_week'] >= 5).mean() * 100

    # Label based on characteristics
    if avg_hour >= 7 and avg_hour <= 9 or avg_hour >= 17 and avg_hour <= 19:
        label = "Commuter"
    elif weekend_pct > 40:
        label = "Leisure"
    elif avg_duration < 600:  # < 10 minutes
        label = "Last-Mile"
    else:
        label = "Errand"

    archetype_labels.append({
        'cluster': i,
        'label': label,
        'count': len(cluster_trips),
        'avg_duration': avg_duration,
        'avg_displacement': avg_displacement,
        'avg_hour': avg_hour,
        'weekend_pct': weekend_pct
    })

    print(f"    • Cluster {i} ({label}): {len(cluster_trips):,} trips")
    print(f"      Duration: {avg_duration/60:.1f}min, Distance: {avg_displacement:.0f}m, Hour: {avg_hour:.1f}")

# ============================================================================
# 5. TEMPORAL TRENDS (MONTHLY AGGREGATES)
# ============================================================================

print("\n[STEP 5] GENERATING TEMPORAL TRENDS...")

# Define month mapping with correct number of days
month_info = {
    '2024-11': ('Nov', 30),
    '2024-12': ('Dec', 31),
    '2025-01': ('Jan', 31),
    '2025-02': ('Feb', 28),
    '2025-03': ('Mar', 31),
    '2025-04': ('Apr', 30),
    '2025-05': ('May', 31),
    '2025-06': ('Jun', 30),
    '2025-07': ('Jul', 31),
    '2025-08': ('Aug', 31),
    '2025-09': ('Sep', 30),
    '2025-10': ('Oct', 31)
}

# POGOH Average Daily Trips
pogoh_monthly = trips.groupby('year_month').size()
pogoh_daily_avg = []
labels = []

for period_str, (label, days) in month_info.items():
    period = pd.Period(period_str, freq='M')
    if period in pogoh_monthly.index:
        daily_avg = pogoh_monthly[period] / days
        pogoh_daily_avg.append(daily_avg)
    else:
        pogoh_daily_avg.append(0)
    labels.append(label)

# PRT Average Daily Ridership (sum of R_W_YYYYMM for each month)
ridership_monthly_cols = {
    'R_W_202409': 'Sep',  # Note: 202409 is Sept 2024, but we start Nov 2024
    'R_W_202501': 'Jan',
    'R_W_202505': 'May'
}

# Map all available R_W columns to months - ALL DATA (Historical)
prt_historical = {}
all_rw_cols = [c for c in bus_stops_clean.columns if c.startswith('R_W_')]

for col in all_rw_cols:
    # Extract date part R_W_YYYYMM
    try:
        date_str = f"{col[4:8]}-{col[8:10]}"
        # R_W values are in thousands of boardings
        total = bus_stops_clean[col].fillna(0).sum() * 1000
        
        # ONLY add if total > 0 (Filter out blank historical data)
        if total > 0:
            prt_historical[date_str] = total
    except:
        continue

# Sort by date
sorted_dates = sorted(prt_historical.keys())

# Also keep the "sparse" data for the original chart logic if needed, but user requested "all not only 2025"
# We will replace prt_ridership with this fuller dataset structure, but we need to align with POGOH dates if we want comparison?
# The user said "prt data please show all not only 2025... axis (x axis) can be discrete"
# So we will export a separate object for the full PRT history.

print(f"    ✓ Generated {len(sorted_dates)} months of historical PRT data (non-zero only)")

# ... (Keep existing sparse logic for backward compatibility if needed, OR just export this new one)
# Let's add it to monthly_trends export

# ============================================================================
# 6. ADDITIONAL CHART DATA GENERATION
# ============================================================================

print("\n[STEP 6] GENERATING ADDITIONAL CHART DATA...")

# CHART 4: Directionality (Polar Chart - Bearing Analysis)
print("  → Calculating trip directionality (bearing angles)...")
def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate initial bearing between two points"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(x, y)
    return (np.degrees(bearing) + 360) % 360

trips_with_coords = trips.dropna(subset=['start_lat', 'start_lon', 'end_lat', 'end_lon'])
trips_with_coords['bearing'] = calculate_bearing(
    trips_with_coords['start_lat'], trips_with_coords['start_lon'],
    trips_with_coords['end_lat'], trips_with_coords['end_lon']
)

# Bin bearings into 8 cardinal directions
direction_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
direction_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
trips_with_coords['direction'] = pd.cut(trips_with_coords['bearing'], bins=direction_bins, labels=direction_labels, include_lowest=True)
direction_counts = trips_with_coords['direction'].value_counts().reindex(direction_labels, fill_value=0)
print(f"    ✓ Directionality calculated (Primary: {direction_counts.idxmax()} - {direction_counts.max():,} trips)")

# CHART 5: Demographics (Stacked Bar - Rider Type by Station)
print("  → Aggregating rider type demographics...")
# Get top 10 stations by trip volume
top_stations = trips.groupby('Start Station Name').size().nlargest(10).index
demo_data = trips[trips['Start Station Name'].isin(top_stations)].groupby(['Start Station Name', 'Rider Type']).size().unstack(fill_value=0)
print(f"    ✓ Demographics aggregated for {len(top_stations)} stations")

# CHART 6: Duration Distribution (Histogram)
print("  → Preparing duration distribution...")
duration_hist, bin_edges = np.histogram(trips['Duration'] / 60, bins=30, range=(0, 60))  # Convert to minutes
duration_bins = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
print(f"    ✓ Duration histogram: Mean {trips['Duration'].mean()/60:.1f}min, Median {trips['Duration'].median()/60:.1f}min")

# CHART 7: Correlation (Scatter - Bus Volume vs Bike Connectivity)
print("  → Calculating bus-bike correlation...")
# For each bus stop, get integration score vs ridership
correlation_data = bus_stops_clean[['stop_name', 'bus_boardings', 'integration_index', 'bike_trips_400m', 'HOOD']].copy()
correlation_data = correlation_data[correlation_data['bus_boardings'] > 0]  # Filter out zero-ridership stops
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(
    correlation_data['bus_boardings'],
    correlation_data['integration_index']
)
print(f"    ✓ Correlation: R²={r_value**2:.3f}, p={p_value:.4f}")

# CHART 8A: Hourly Heatmap (Day × Hour)
print("  → Generating hourly × daily heatmap...")
heatmap_data = trips.groupby(['day_name', 'hour']).size().reset_index(name='count')
days_ordered = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hours = list(range(24))
heatmap_pivot = heatmap_data.pivot_table(index='hour', columns='day_name', values='count', fill_value=0)
heatmap_pivot = heatmap_pivot.reindex(columns=days_ordered, fill_value=0)
peak_hour = heatmap_data.loc[heatmap_data['count'].idxmax()]
print(f"    ✓ Peak: {peak_hour['day_name']} at {int(peak_hour['hour'])}:00 ({int(peak_hour['count']):,} trips)")

# CHART 8B: Season × Hour Heatmap (NEW!)
print("  → Generating seasonal × hourly heatmap...")
# Define seasons based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # 9, 10, 11
        return 'Fall'

trips['season'] = trips['month'].apply(get_season)
seasonal_heatmap = trips.groupby(['season', 'hour']).size().reset_index(name='count')
seasons_ordered = ['Winter', 'Spring', 'Summer', 'Fall']
seasonal_pivot = seasonal_heatmap.pivot_table(index='hour', columns='season', values='count', fill_value=0)
seasonal_pivot = seasonal_pivot.reindex(columns=seasons_ordered, fill_value=0)
print(f"    ✓ Seasonal heatmap: {len(seasonal_pivot)} hours × {len(seasons_ordered)} seasons")

# CHART 9: Daily Timeseries (Real Daily, not monthly average)
print("  → Generating daily timeseries...")
trips['date'] = trips['Start Date'].dt.date
daily_pogoh = trips.groupby('date').size().reset_index(name='trips')
daily_pogoh['date_str'] = pd.to_datetime(daily_pogoh['date']).dt.strftime('%Y-%m-%d')

# Also generate campus vs city daily splits
daily_campus = trips[trips['is_campus']].groupby('date').size().reset_index(name='trips')
daily_campus['date_str'] = pd.to_datetime(daily_campus['date']).dt.strftime('%Y-%m-%d')

daily_city = trips[~trips['is_campus']].groupby('date').size().reset_index(name='trips')
daily_city['date_str'] = pd.to_datetime(daily_city['date']).dt.strftime('%Y-%m-%d')

# Merge to ensure all dates are present (fill missing with 0)
all_dates = pd.DataFrame({'date_str': daily_pogoh['date_str']})
daily_campus_full = all_dates.merge(daily_campus[['date_str', 'trips']], on='date_str', how='left').fillna(0)
daily_city_full = all_dates.merge(daily_city[['date_str', 'trips']], on='date_str', how='left').fillna(0)

print(f"    ✓ Daily timeseries: {len(daily_pogoh)} days (System + Campus/City splits)")

# CHART 10: Top 10 PRT Stops Near POGOH Stations
print("  → Finding top PRT stops near POGOH stations...")
# For each bus stop, check if it has ANY bike stations within 400m
bus_stops_near_pogoh = bus_stops_clean[bus_stops_clean['bike_trips_400m'] > 0].copy()
# Sort by bus boardings and get top 10
top_prt_near_pogoh = bus_stops_near_pogoh.nlargest(10, 'bus_boardings')[['stop_name', 'bus_boardings', 'bike_trips_400m', 'nearest_dock_distance']]
print(f"    ✓ Top 10 PRT stops near POGOH identified")

# Campus vs System-Wide Temporal Comparison
print("  → Separating Campus vs System-Wide temporal trends...")
pogoh_campus_monthly = trips[trips['is_campus']].groupby('year_month').size()
pogoh_city_monthly = trips[~trips['is_campus']].groupby('year_month').size()

pogoh_campus_daily = []
pogoh_city_daily = []
for period_str, (label, days) in month_info.items():
    period = pd.Period(period_str, freq='M')

    if period in pogoh_campus_monthly.index:
        pogoh_campus_daily.append(pogoh_campus_monthly[period] / days)
    else:
        pogoh_campus_daily.append(0)

    if period in pogoh_city_monthly.index:
        pogoh_city_daily.append(pogoh_city_monthly[period] / days)
    else:
        pogoh_city_daily.append(0)

print(f"    ✓ Campus/City split calculated")

# ============================================================================
# 7. EXPORT TO JSON
# ============================================================================

print("\n[STEP 6] EXPORTING DATA TO JSON...")

# Bus Stops GeoJSON
bus_stops_export = []
for _, stop in bus_stops_clean.iterrows():  # Export ALL stops
    bus_stops_export.append({
        'loc': [stop['latitude'], stop['longitude']],
        'name': stop['stop_name'],
        'vol': int(stop['bus_boardings']) if pd.notna(stop['bus_boardings']) else 0,
        'integration_score': float(stop['integration_index']) if pd.notna(stop['integration_index']) else 0,
        'bike_trips_nearby': int(stop['bike_trips_400m']),
        'hood': stop['HOOD'] if pd.notna(stop['HOOD']) else 'Unknown'
    })

# Bike Stations GeoJSON
bike_stations_export = []
station_trip_counts = trips.groupby('Start Station Id').size().to_dict()

for _, station in stations.iterrows():
    trip_count = station_trip_counts.get(station['Id'], 0)
    bike_stations_export.append({
        'loc': [station['Latitude'], station['Longitude']],
        'name': station['Name'],
        'trips': int(trip_count),
        'docks': int(station['Total Docks'])
    })

# Monthly Trends (ensure no NaN values) - ENHANCED with Campus/City split & Full PRT History
monthly_trends = {
    'labels': labels,
    'pogoh_daily_avg': [float(x) if pd.notna(x) else 0 for x in pogoh_daily_avg],
    'pogoh_campus_daily': [float(x) if pd.notna(x) else 0 for x in pogoh_campus_daily],
    'pogoh_city_daily': [float(x) if pd.notna(x) else 0 for x in pogoh_city_daily],
    # 'prt_daily_avg': [float(x) if pd.notna(x) else None for x in prt_ridership], # Replaced by full history
    'all_prt_labels': sorted_dates,
    'all_prt_values': [float(prt_historical[d]) for d in sorted_dates]
}

# Directionality Data (Polar Chart)
directionality = {
    'labels': direction_labels,
    'values': direction_counts.tolist()
}

# Demographics Data (Stacked Bar)
demographics = {
    'stations': demo_data.index.tolist(),
    'rider_types': demo_data.columns.tolist(),
    'data': demo_data.to_dict('list')
}

# Duration Distribution (Histogram)
duration_distribution = {
    'bins': [float(b) for b in duration_bins],
    'counts': duration_hist.tolist()
}

# Correlation Data (Scatter)
correlation_scatter = {
    'bus_boardings': correlation_data['bus_boardings'].tolist(),
    'integration_index': correlation_data['integration_index'].tolist(),
    'names': correlation_data['stop_name'].tolist(),
    'neighborhoods': correlation_data['HOOD'].fillna('Unknown').tolist(),
    'regression': {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value**2),
        'p_value': float(p_value)
    }
}

# Heatmap Data (Hour × Day)
heatmap = {
    'days': days_ordered,
    'hours': hours,
    'data': heatmap_pivot.values.tolist()
}

# Seasonal Heatmap Data (Hour × Season)
seasonal_heatmap_export = {
    'seasons': seasons_ordered,
    'hours': hours,
    'data': seasonal_pivot.values.tolist()
}

# Daily Timeseries (Real Daily Data with Campus/City splits)
daily_timeseries = {
    'dates': daily_pogoh['date_str'].tolist(),
    'pogoh_trips': daily_pogoh['trips'].tolist(),
    'pogoh_campus_trips': daily_campus_full['trips'].astype(int).tolist(),
    'pogoh_city_trips': daily_city_full['trips'].astype(int).tolist()
}

# Top 10 PRT Stops Near POGOH
top_prt_pogoh = {
    'stops': top_prt_near_pogoh['stop_name'].tolist(),
    'bus_boardings': top_prt_near_pogoh['bus_boardings'].tolist(),
    'bike_trips_nearby': top_prt_near_pogoh['bike_trips_400m'].tolist()
}

# Trip Archetypes (Aggregate by cluster for radar chart)
archetypes_export = []
for arch in archetype_labels:
    archetypes_export.append({
        'label': arch['label'],
        'count': arch['count'],
        'avg_duration_min': round(arch['avg_duration'] / 60, 1),
        'avg_distance_m': round(arch['avg_displacement'], 0),
        'peak_hour': round(arch['avg_hour'], 1)
    })

# ============================================================================
# STATION × ARCHETYPE ANALYSIS
# ============================================================================
print("  → Analyzing top stations per archetype...")

# Create archetype label mapping
archetype_map = {}
for arch in archetype_labels:
    archetype_map[arch['cluster']] = arch['label']

trips_clean['archetype_label'] = trips_clean['archetype'].map(archetype_map)

# Get top 3 stations per archetype by percentage
station_archetype_top = {}

for archetype in ['Commuter', 'Last-Mile', 'Errand', 'Leisure']:
    # Count trips by station for this archetype
    station_counts = trips_clean.groupby(['Start Station Name', 'archetype_label']).size().reset_index(name='trips')

    # Get total trips per station
    station_totals = trips_clean.groupby('Start Station Name').size().reset_index(name='total_trips')

    # Merge to calculate percentages
    station_archetype_pct = station_counts.merge(station_totals, on='Start Station Name')
    station_archetype_pct['pct'] = (station_archetype_pct['trips'] / station_archetype_pct['total_trips']) * 100

    # Filter for this archetype only
    archetype_data = station_archetype_pct[station_archetype_pct['archetype_label'] == archetype]

    # Get top 3 by percentage (must have at least 50 trips for statistical significance)
    archetype_data = archetype_data[archetype_data['total_trips'] >= 50]
    top_3 = archetype_data.nlargest(3, 'pct')

    station_archetype_top[archetype] = {
        'stations': top_3['Start Station Name'].tolist(),
        'percentages': top_3['pct'].round(1).tolist(),
        'trip_counts': top_3['trips'].astype(int).tolist(),
        'total_trips': top_3['total_trips'].astype(int).tolist()
    }

print(f"    ✓ Top stations per archetype calculated")

# Write JSON files
output_dir = Path('./processed_data')
output_dir.mkdir(exist_ok=True)
with open(output_dir / 'bus_stops_geo.json', 'w') as f:
    json.dump(bus_stops_export, f, indent=2)
print(f"    ✓ Exported {len(bus_stops_export)} bus stops to bus_stops_geo.json")

with open(output_dir / 'bike_stations_geo.json', 'w') as f:
    json.dump(bike_stations_export, f, indent=2)
print(f"    ✓ Exported {len(bike_stations_export)} bike stations to bike_stations_geo.json")

with open(output_dir / 'monthly_trends.json', 'w') as f:
    json.dump(monthly_trends, f, indent=2)
print(f"    ✓ Exported temporal trends to monthly_trends.json")

with open(output_dir / 'archetypes.json', 'w') as f:
    json.dump(archetypes_export, f, indent=2)
print(f"    ✓ Exported archetypes to archetypes.json")

with open(output_dir / 'directionality.json', 'w') as f:
    json.dump(directionality, f, indent=2)
print(f"    ✓ Exported directionality to directionality.json")

with open(output_dir / 'demographics.json', 'w') as f:
    json.dump(demographics, f, indent=2)
print(f"    ✓ Exported demographics to demographics.json")

with open(output_dir / 'duration_distribution.json', 'w') as f:
    json.dump(duration_distribution, f, indent=2)
print(f"    ✓ Exported duration distribution to duration_distribution.json")

with open(output_dir / 'correlation.json', 'w') as f:
    json.dump(correlation_scatter, f, indent=2)
print(f"    ✓ Exported correlation data to correlation.json")

with open(output_dir / 'heatmap.json', 'w') as f:
    json.dump(heatmap, f, indent=2)
print(f"    ✓ Exported heatmap to heatmap.json")

with open(output_dir / 'seasonal_heatmap.json', 'w') as f:
    json.dump(seasonal_heatmap_export, f, indent=2)
print(f"    ✓ Exported seasonal heatmap to seasonal_heatmap.json")

with open(output_dir / 'daily_timeseries.json', 'w') as f:
    json.dump(daily_timeseries, f, indent=2)
print(f"    ✓ Exported daily timeseries to daily_timeseries.json")

with open(output_dir / 'top_prt_pogoh.json', 'w') as f:
    json.dump(top_prt_pogoh, f, indent=2)
print(f"    ✓ Exported top PRT-POGOH stops to top_prt_pogoh.json")

with open(output_dir / 'station_archetypes.json', 'w') as f:
    json.dump(station_archetype_top, f, indent=2)
print(f"    ✓ Exported station archetypes to station_archetypes.json")

# ============================================================================
# 8. EXPORT CSV VERSIONS (For Easy Inspection)
# ============================================================================

print("\n[STEP 7] EXPORTING CSV VERSIONS...")

# 1. Bus Stops CSV
bus_stops_df = pd.DataFrame(bus_stops_export)
bus_stops_df[['lat', 'lon']] = pd.DataFrame(bus_stops_df['loc'].tolist(), index=bus_stops_df.index)
bus_stops_df = bus_stops_df.drop('loc', axis=1)
bus_stops_df.to_csv(output_dir / 'bus_stops_geo.csv', index=False)
print(f"    ✓ Exported bus_stops_geo.csv ({len(bus_stops_df)} rows)")

# 2. Bike Stations CSV
bike_stations_df = pd.DataFrame(bike_stations_export)
bike_stations_df[['lat', 'lon']] = pd.DataFrame(bike_stations_df['loc'].tolist(), index=bike_stations_df.index)
bike_stations_df = bike_stations_df.drop('loc', axis=1)
bike_stations_df.to_csv(output_dir / 'bike_stations_geo.csv', index=False)
print(f"    ✓ Exported bike_stations_geo.csv ({len(bike_stations_df)} rows)")

# 3. Trip Archetypes CSV
archetypes_df = pd.DataFrame(archetypes_export)
archetypes_df.to_csv(output_dir / 'archetypes.csv', index=False)
print(f"    ✓ Exported archetypes.csv ({len(archetypes_df)} rows)")

# 4. Demographics CSV
demographics_df = pd.DataFrame(demographics['data'], index=demographics['stations'])
demographics_df.index.name = 'station'
demographics_df.to_csv(output_dir / 'demographics.csv')
print(f"    ✓ Exported demographics.csv ({len(demographics_df)} rows)")

# 5. Directionality CSV
directionality_df = pd.DataFrame({
    'direction': directionality['labels'],
    'trip_count': directionality['values']
})
directionality_df.to_csv(output_dir / 'directionality.csv', index=False)
print(f"    ✓ Exported directionality.csv ({len(directionality_df)} rows)")

# 6. Duration Distribution CSV
duration_df = pd.DataFrame({
    'duration_min': duration_distribution['bins'],
    'trip_count': duration_distribution['counts']
})
duration_df.to_csv(output_dir / 'duration_distribution.csv', index=False)
print(f"    ✓ Exported duration_distribution.csv ({len(duration_df)} rows)")

# 7. Correlation CSV
correlation_df = pd.DataFrame({
    'stop_name': correlation_scatter['names'],
    'bus_boardings': correlation_scatter['bus_boardings'],
    'integration_index': correlation_scatter['integration_index'],
    'neighborhood': correlation_scatter['neighborhoods']
})
correlation_df.to_csv(output_dir / 'correlation.csv', index=False)
print(f"    ✓ Exported correlation.csv ({len(correlation_df)} rows)")

# 8. Daily Timeseries CSV
daily_ts_df = pd.DataFrame({
    'date': daily_timeseries['dates'],
    'pogoh_trips_total': daily_timeseries['pogoh_trips'],
    'pogoh_trips_campus': daily_timeseries['pogoh_campus_trips'],
    'pogoh_trips_city': daily_timeseries['pogoh_city_trips']
})
daily_ts_df.to_csv(output_dir / 'daily_timeseries.csv', index=False)
print(f"    ✓ Exported daily_timeseries.csv ({len(daily_ts_df)} rows)")

# 9. Monthly Trends CSV
monthly_trends_df = pd.DataFrame({
    'month': monthly_trends['labels'],
    'pogoh_daily_avg': monthly_trends['pogoh_daily_avg'],
    'pogoh_campus_daily': monthly_trends['pogoh_campus_daily'],
    'pogoh_city_daily': monthly_trends['pogoh_city_daily']
})
monthly_trends_df.to_csv(output_dir / 'monthly_trends.csv', index=False)
print(f"    ✓ Exported monthly_trends.csv ({len(monthly_trends_df)} rows)")

# 10. PRT Historical CSV
prt_historical_df = pd.DataFrame({
    'date': monthly_trends['all_prt_labels'],
    'bus_boardings': monthly_trends['all_prt_values']
})
prt_historical_df.to_csv(output_dir / 'prt_historical.csv', index=False)
print(f"    ✓ Exported prt_historical.csv ({len(prt_historical_df)} rows)")

# 11. Top PRT-POGOH Integration CSV
top_prt_df = pd.DataFrame({
    'stop_name': top_prt_pogoh['stops'],
    'bus_boardings': top_prt_pogoh['bus_boardings'],
    'bike_trips_nearby': top_prt_pogoh['bike_trips_nearby']
})
top_prt_df.to_csv(output_dir / 'top_prt_pogoh.csv', index=False)
print(f"    ✓ Exported top_prt_pogoh.csv ({len(top_prt_df)} rows)")

# 12. Heatmap CSV (Hour × Day)
heatmap_df = pd.DataFrame(
    heatmap['data'],
    columns=heatmap['days'],
    index=heatmap['hours']
)
heatmap_df.index.name = 'hour'
heatmap_df.to_csv(output_dir / 'heatmap_hour_day.csv')
print(f"    ✓ Exported heatmap_hour_day.csv ({len(heatmap_df)} rows)")

# 13. Seasonal Heatmap CSV (Hour × Season)
seasonal_heatmap_df = pd.DataFrame(
    seasonal_heatmap_export['data'],
    columns=seasonal_heatmap_export['seasons'],
    index=seasonal_heatmap_export['hours']
)
seasonal_heatmap_df.index.name = 'hour'
seasonal_heatmap_df.to_csv(output_dir / 'heatmap_hour_season.csv')
print(f"    ✓ Exported heatmap_hour_season.csv ({len(seasonal_heatmap_df)} rows)")

# 14. Station Archetypes CSV
station_arch_rows = []
for archetype, data in station_archetype_top.items():
    for i in range(len(data['stations'])):
        station_arch_rows.append({
            'archetype': archetype,
            'station': data['stations'][i],
            'percentage': data['percentages'][i],
            'archetype_trips': data['trip_counts'][i],
            'total_trips': data['total_trips'][i]
        })
station_arch_df = pd.DataFrame(station_arch_rows)
station_arch_df.to_csv(output_dir / 'station_archetypes.csv', index=False)
print(f"    ✓ Exported station_archetypes.csv ({len(station_arch_df)} rows)")

print(f"\n    → Total: 13 JSON + 14 CSV files exported to ./processed_data/")

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETE - SUMMARY STATISTICS")
print("=" * 80)

print(f"\nData Coverage:")
print(f"  • Total POGOH Trips (Nov 2024 - Oct 2025): {len(trips):,}")
print(f"  • Total Bus Stops: {len(bus_stops_clean):,}")
print(f"  • Total Bike Stations: {len(stations)}")

print(f"\nIntegration Metrics:")
print(f"  • Stops with bike activity (>0 trips in 400m): {(bus_stops_clean['bike_trips_400m'] > 0).sum()}")
print(f"  • Stops with NO bike activity: {(bus_stops_clean['bike_trips_400m'] == 0).sum()}")
print(f"  • Average Integration Index: {bus_stops_clean['integration_index'].mean():.4f}")

top_integrated = bus_stops_clean.nlargest(3, 'integration_index')
print(f"\nTop 3 Integrated Stops:")
for idx, (_, stop) in enumerate(top_integrated.iterrows(), 1):
    print(f"  {idx}. {stop['stop_name']} (Index: {stop['integration_index']:.4f})")

print(f"\nTrip Archetypes:")
for arch in archetypes_export:
    pct = (arch['count'] / len(trips_clean)) * 100
    print(f"  • {arch['label']}: {arch['count']:,} trips ({pct:.1f}%)")

print("\n" + "=" * 80)
print("JSON files ready for HTML integration!")
print("=" * 80 + "\n")

# ============================================================================
# 10. BUILD DATA.JS
# ============================================================================

print("\n" + "=" * 80)
print("BUILDING DATA.JS FOR INTERACTIVE DASHBOARD")
print("=" * 80)

# Read all JSON files from processed_data directory (re-reading ensures validity)
# We use the variables we just exported if available, but reading is safer for a robust pipeline step

try:
    with open(output_dir / 'bus_stops_geo.json', 'r') as f: bus_stops = json.load(f)
    with open(output_dir / 'bike_stations_geo.json', 'r') as f: bike_stations = json.load(f)
    with open(output_dir / 'monthly_trends.json', 'r') as f: monthly_trends = json.load(f)
    with open(output_dir / 'archetypes.json', 'r') as f: archetypes = json.load(f)
    with open(output_dir / 'directionality.json', 'r') as f: directionality = json.load(f)
    with open(output_dir / 'demographics.json', 'r') as f: demographics = json.load(f)
    with open(output_dir / 'duration_distribution.json', 'r') as f: duration_dist = json.load(f)
    with open(output_dir / 'correlation.json', 'r') as f: correlation = json.load(f)
    with open(output_dir / 'heatmap.json', 'r') as f: heatmap = json.load(f)
    with open(output_dir / 'seasonal_heatmap.json', 'r') as f: seasonal_heatmap = json.load(f)
    with open(output_dir / 'daily_timeseries.json', 'r') as f: daily_timeseries = json.load(f)
    with open(output_dir / 'top_prt_pogoh.json', 'r') as f: top_prt_pogoh = json.load(f)
    with open(output_dir / 'station_archetypes.json', 'r') as f: station_archetypes = json.load(f)

    # Write data.js
    with open('data.js', 'w') as f:
        f.write('// PITTSBURGH TRANSIT ATLAS - ENHANCED DATA MODULE\n\n')
        f.write(f'const busStopsData = {json.dumps(bus_stops, indent=2)};\n\n')
        f.write(f'const bikeStationsData = {json.dumps(bike_stations, indent=2)};\n\n')
        f.write(f'const monthlyTrendsData = {json.dumps(monthly_trends, indent=2)};\n\n')
        f.write(f'const archetypesData = {json.dumps(archetypes, indent=2)};\n\n')
        f.write(f'const directionalityData = {json.dumps(directionality, indent=2)};\n\n')
        f.write(f'const demographicsData = {json.dumps(demographics, indent=2)};\n\n')
        f.write(f'const durationDistData = {json.dumps(duration_dist, indent=2)};\n\n')
        f.write(f'const correlationData = {json.dumps(correlation, indent=2)};\n\n')
        f.write(f'const heatmapData = {json.dumps(heatmap, indent=2)};\n\n')
        f.write(f'const seasonalHeatmapData = {json.dumps(seasonal_heatmap, indent=2)};\n\n')
        f.write(f'const dailyTimeseriesData = {json.dumps(daily_timeseries, indent=2)};\n\n')
        f.write(f'const topPrtPogohData = {json.dumps(top_prt_pogoh, indent=2)};\n\n')
        f.write(f'const stationArchetypesData = {json.dumps(station_archetypes, indent=2)};\n')

    print("✓ data.js generated successfully!")
    print(f"  • File size: {len(open('data.js').read()) / 1024:.1f} KB")

except Exception as e:
    print(f"❌ Error building data.js: {e}")

print("\nPipeline fully completed.")
