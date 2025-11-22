#!/usr/bin/env python3
"""
Build data.js from individual JSON files
"""

import json

# Read all JSON files
with open('bus_stops_geo.json', 'r') as f:
    bus_stops = json.load(f)

with open('bike_stations_geo.json', 'r') as f:
    bike_stations = json.load(f)

with open('monthly_trends.json', 'r') as f:
    monthly_trends = json.load(f)

with open('archetypes.json', 'r') as f:
    archetypes = json.load(f)

with open('directionality.json', 'r') as f:
    directionality = json.load(f)

with open('demographics.json', 'r') as f:
    demographics = json.load(f)

with open('duration_distribution.json', 'r') as f:
    duration_dist = json.load(f)

with open('correlation.json', 'r') as f:
    correlation = json.load(f)

with open('heatmap.json', 'r') as f:
    heatmap = json.load(f)

with open('seasonal_heatmap.json', 'r') as f:
    seasonal_heatmap = json.load(f)

with open('daily_timeseries.json', 'r') as f:
    daily_timeseries = json.load(f)

with open('top_prt_pogoh.json', 'r') as f:
    top_prt_pogoh = json.load(f)

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
    f.write(f'const topPrtPogohData = {json.dumps(top_prt_pogoh, indent=2)};\n')

print("✓ data.js generated successfully!")
print(f"  • File size: {len(open('data.js').read()) / 1024:.1f} KB")
