#!/usr/bin/env python3
"""
Static Visualization Generator for PGH Transit Atlas
Uses Seaborn and Bokeh to create publication-quality charts
For old-school class submission (static HTML report)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.layouts import column
from bokeh.palettes import Category20
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create output directory for charts
output_dir = Path('./static_viz')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATING STATIC VISUALIZATIONS FOR EDA REPORT")
print("=" * 80)

# Load processed data
data_dir = Path('./processed_data')

print("\n[1/8] Loading processed data...")
with open(data_dir / 'daily_timeseries.json', 'r') as f:
    daily_data = json.load(f)

with open(data_dir / 'archetypes.json', 'r') as f:
    archetypes = json.load(f)

with open(data_dir / 'demographics.json', 'r') as f:
    demographics = json.load(f)

with open(data_dir / 'station_archetypes.json', 'r') as f:
    station_archetypes = json.load(f)

with open(data_dir / 'heatmap.json', 'r') as f:
    heatmap_raw = json.load(f)

# Load daily timeseries for hourly aggregation
df_daily_full = pd.read_csv(data_dir / 'daily_timeseries.csv')

print("✓ Data loaded successfully")

# ============================================================================
# VIZ 1: DAILY TIMESERIES (Seaborn)
# ============================================================================
print("\n[2/8] Creating daily timeseries chart (Seaborn)...")

df_daily = pd.DataFrame({
    'date': pd.to_datetime(daily_data['dates']),
    'Total': daily_data['pogoh_trips'],
    'Campus': daily_data['pogoh_campus_trips'],
    'City': daily_data['pogoh_city_trips']
})

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_daily['date'], df_daily['Total'], linewidth=2, label='Total POGOH', color='#2B4CFF', alpha=0.9)
ax.plot(df_daily['date'], df_daily['Campus'], linewidth=1.5, label='Campus Corridor', color='#FF9500', alpha=0.8)
ax.plot(df_daily['date'], df_daily['City'], linewidth=1.5, label='City-Wide', color='#34C759', alpha=0.8)

ax.set_title('FIG 1: Daily Ridership Timeseries (2024 Full Year)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Daily Trips', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(bottom=0)

# Add annotations for key insights
max_idx = df_daily['Total'].idxmax()
max_date = df_daily.loc[max_idx, 'date']
max_val = df_daily.loc[max_idx, 'Total']
ax.annotate(f'Peak: {max_val} trips\n{max_date.strftime("%b %d")}',
            xy=(max_date, max_val),
            xytext=(20, 30),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black'))

plt.tight_layout()
plt.savefig(output_dir / 'fig1_daily_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved fig1_daily_timeseries.png")

# ============================================================================
# VIZ 2: TRIP ARCHETYPES (Seaborn Horizontal Bar)
# ============================================================================
print("\n[3/8] Creating trip archetypes chart (Seaborn)...")

# archetypes is a list of dicts
df_arch = pd.DataFrame(archetypes)
df_arch = df_arch.rename(columns={'label': 'Archetype', 'count': 'Trips'})

# Calculate percentages
total_trips = df_arch['Trips'].sum()
df_arch['Percentage'] = (df_arch['Trips'] / total_trips) * 100

# Sort by trip count descending
df_arch = df_arch.sort_values('Trips', ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2B4CFF', '#34C759', '#FF9500', '#FF2B8C']
bars = ax.barh(df_arch['Archetype'], df_arch['Trips'], color=colors[:len(df_arch)], edgecolor='black', linewidth=1.5)

ax.set_title('FIG 2: Trip Behavioral Archetypes (K-Means Clustering)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Number of Trips', fontsize=12, fontweight='bold')
ax.set_ylabel('Archetype', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add percentage labels
for i, (idx, row) in enumerate(df_arch.iterrows()):
    ax.text(row['Trips'] + 5000, i, f"{row['Percentage']:.1f}%",
            va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig2_archetypes.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved fig2_archetypes.png")

# ============================================================================
# VIZ 3: RIDER TYPE COMPARISON (Seaborn)
# ============================================================================
print("\n[4/8] Creating rider type comparison (Seaborn)...")

# Calculate totals per rider type
casual_total = sum(demographics['data']['CASUAL'])
member_total = sum(demographics['data']['MEMBER'])

df_demo = pd.DataFrame({
    'Rider Type': ['CASUAL', 'MEMBER'],
    'Trips': [casual_total, member_total]
})

# Calculate percentages
total = df_demo['Trips'].sum()
df_demo['Percentage'] = (df_demo['Trips'] / total) * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df_demo['Rider Type'], df_demo['Trips'],
              color=['#FF9500', '#2B4CFF'], edgecolor='black', linewidth=2, width=0.6)

ax.set_title('FIG 3: Rider Type Distribution', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Number of Trips', fontsize=12, fontweight='bold')
ax.set_xlabel('Rider Type', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for i, (idx, row) in enumerate(df_demo.iterrows()):
    ax.text(i, row['Trips'] + 5000, f"{row['Trips']:,}\n({row['Percentage']:.1f}%)",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig3_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved fig3_demographics.png")

# ============================================================================
# VIZ 4: HOURLY PATTERNS (Bokeh Interactive)
# ============================================================================
print("\n[5/8] Creating hourly patterns chart (Bokeh)...")

# Calculate hourly totals from heatmap data (sum across days)
hourly_trips = [sum(heatmap_raw['data'][h]) for h in range(24)]

df_hourly = pd.DataFrame({
    'hour': list(range(24)),
    'trips': hourly_trips
})

source = ColumnDataSource(df_hourly)

p = figure(
    title="FIG 4: Hourly Trip Distribution (24-Hour Pattern)",
    x_axis_label="Hour of Day",
    y_axis_label="Total Trips",
    width=1000,
    height=400,
    toolbar_location="above"
)

p.line('hour', 'trips', source=source, line_width=3, color='#2B4CFF', alpha=0.8)
p.circle('hour', 'trips', source=source, size=8, color='#2B4CFF', alpha=0.6)

# Add hover tool
hover = HoverTool(tooltips=[
    ("Hour", "@hour:00"),
    ("Trips", "@trips{0,0}")
])
p.add_tools(hover)

# Styling
p.title.text_font_size = '16pt'
p.title.text_font_style = 'bold'
p.xaxis.axis_label_text_font_size = '12pt'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font_style = 'bold'
p.yaxis.axis_label_text_font_style = 'bold'

output_file(output_dir / 'fig4_hourly_bokeh.html')
save(p)
print("✓ Saved fig4_hourly_bokeh.html")

# ============================================================================
# VIZ 5: DAY × HOUR HEATMAP (Seaborn)
# ============================================================================
print("\n[6/8] Creating day×hour heatmap (Seaborn)...")

# Convert heatmap data to matrix (hours × days)
days = heatmap_raw['days']
hours = list(range(24))
matrix = heatmap_raw['data']  # Already 24 rows × 7 cols

df_heatmap = pd.DataFrame(matrix, index=hours, columns=days)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df_heatmap, cmap='YlOrRd', annot=False, fmt='d',
            cbar_kws={'label': 'Trip Count'}, linewidths=0.5, ax=ax)

ax.set_title('FIG 5: Day × Hour Demand Heatmap (Weekly Pattern)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
ax.set_ylabel('Hour of Day', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig5_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved fig5_heatmap.png")

# ============================================================================
# VIZ 6: STATION ARCHETYPES - 4 SEPARATE CHARTS (Seaborn)
# ============================================================================
print("\n[7/8] Creating station archetype charts (Seaborn)...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('FIG 6: Behavioral Hotspots (Top 3 Stations per Archetype)',
             fontsize=18, fontweight='bold', y=0.995)

archetype_configs = [
    ('Commuter', '🚴 COMMUTER HUBS', '#2B4CFF', axes[0, 0]),
    ('Last-Mile', '🔗 LAST-MILE CONNECTORS', '#FF9500', axes[0, 1]),
    ('Errand', '🛒 ERRAND CENTERS', '#34C759', axes[1, 0]),
    ('Leisure', '🎨 LEISURE DESTINATIONS', '#FF2B8C', axes[1, 1])
]

for arch_key, title, color, ax in archetype_configs:
    data = station_archetypes[arch_key]

    # Truncate station names for display
    labels = [s[:35] + '...' if len(s) > 35 else s for s in data['stations']]

    bars = ax.barh(labels, data['percentages'], color=color, edgecolor='black', linewidth=2)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Percentage of Trips (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 75)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Highest at top

    # Add percentage + trip count labels
    for i, (pct, trips, total) in enumerate(zip(data['percentages'], data['trip_counts'], data['total_trips'])):
        ax.text(pct + 1, i, f"{pct:.1f}% ({trips:,}/{total:,})",
                va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig6_station_archetypes.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved fig6_station_archetypes.png")

# ============================================================================
# VIZ 7: ARCHETYPE COMPARISON (Bokeh Multi-Bar)
# ============================================================================
print("\n[8/8] Creating archetype comparison (Bokeh)...")

# Prepare data from DataFrame
archetypes_list = [a['label'] for a in archetypes]
trips_list = [a['count'] for a in archetypes]
percentages = [(a['count'] / sum([x['count'] for x in archetypes])) * 100 for a in archetypes]

colors_list = ['#2B4CFF', '#34C759', '#FF9500', '#FF2B8C']

source = ColumnDataSource(data={
    'archetypes': archetypes_list,
    'trips': trips_list,
    'percentages': percentages,
    'colors': colors_list
})

p = figure(
    x_range=archetypes_list,
    title="FIG 7: Trip Archetype Distribution",
    width=900,
    height=500,
    toolbar_location="above"
)

p.vbar(x='archetypes', top='trips', source=source, width=0.7,
       color='colors',
       line_color='black', line_width=2)

# Add hover tool
hover = HoverTool(tooltips=[
    ("Archetype", "@archetypes"),
    ("Trips", "@trips{0,0}"),
    ("Percentage", "@percentages{0.0}%")
])
p.add_tools(hover)

# Styling
p.title.text_font_size = '16pt'
p.title.text_font_style = 'bold'
p.xaxis.axis_label = 'Trip Archetype'
p.yaxis.axis_label = 'Number of Trips'
p.xaxis.axis_label_text_font_size = '12pt'
p.yaxis.axis_label_text_font_size = '12pt'
p.xaxis.axis_label_text_font_style = 'bold'
p.yaxis.axis_label_text_font_style = 'bold'
p.xaxis.major_label_text_font_size = '11pt'

output_file(output_dir / 'fig7_archetypes_bokeh.html')
save(p)
print("✓ Saved fig7_archetypes_bokeh.html")

# ============================================================================
# VIZ 8: TOP STATIONS BY RIDER TYPE (Seaborn Grouped Bar)
# ============================================================================
print("\n[BONUS] Creating top stations by rider type (Seaborn)...")

# Get top 10 stations by total trips
stations = demographics['stations']
casual = demographics['data']['CASUAL']
member = demographics['data']['MEMBER']

df_stations = pd.DataFrame({
    'Station': stations,
    'CASUAL': casual,
    'MEMBER': member
})
df_stations['Total'] = df_stations['CASUAL'] + df_stations['MEMBER']
df_stations = df_stations.nlargest(10, 'Total').sort_values('Total', ascending=True)

# Truncate station names
df_stations['Station_Short'] = df_stations['Station'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(df_stations))
width = 0.35

bars1 = ax.barh(x - width/2, df_stations['CASUAL'], width, label='Casual', color='#FF9500', edgecolor='black', linewidth=1)
bars2 = ax.barh(x + width/2, df_stations['MEMBER'], width, label='Member', color='#2B4CFF', edgecolor='black', linewidth=1)

ax.set_title('FIG 8: Top 10 Stations by Rider Type', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Number of Trips', fontsize=12, fontweight='bold')
ax.set_ylabel('Station', fontsize=12, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels(df_stations['Station_Short'], fontsize=9)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'fig8_top_stations.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved fig8_top_stations.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  • fig1_daily_timeseries.png (Seaborn line chart)")
print("  • fig2_archetypes.png (Seaborn horizontal bars)")
print("  • fig3_demographics.png (Seaborn bar chart)")
print("  • fig4_hourly_bokeh.html (Bokeh interactive)")
print("  • fig5_heatmap.png (Seaborn heatmap)")
print("  • fig6_station_archetypes.png (Seaborn 2×2 grid)")
print("  • fig7_archetypes_bokeh.html (Bokeh interactive)")
print("  • fig8_top_stations.png (Seaborn grouped bars)")
print("\nReady for HTML assembly!")
