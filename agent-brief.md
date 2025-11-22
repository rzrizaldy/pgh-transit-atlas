PROJECT BRIEF: Pittsburgh First/Last-Mile Atlas (2024-2025)

1. MISSION STATEMENT

Build a single-page HTML experience analyzing the symbiosis between POGOH bikeshare and PRT (bus) networks.
Core Policy Question: Is bikeshare acting as a valid first/last-mile connector, or operating in a silo?

2. DATA SOURCES (Ingestion Scope)

The agent must build a Python pipeline to ingest and merge:

POGOH Trip Data (Nov 2024 – Oct 2025): * Key Fields: Start Station Id, End Station Id, Duration, Start Date, Rider Type.

Action: Clean timestamps, filter outliers (>4hrs), geocode stations via Station Metadata.

PRT Bus Stop Usage (Series):

Key Fields: stop_id, HOOD (Neighborhood), R_W_202409, R_W_202501, R_W_202505 (Ridership Weekday snapshots).

Action: Melt column-wise dates into row-wise time series.

Geospatial Context:

PRT GTFS/Shapefiles (Routes).

Pittsburgh Neighborhoods (HOOD).

3. ANALYTICAL METRICS (Python Implementation)

The following metrics must be calculated in the EDA phase and visualized in the Interactive phase:

A. The "Integration Index" (Stop Level)

For every bus stop $b$:


$$I_b = \frac{\text{Daily Bus Boardings}_b \times \log(\text{Bike Trips within 400m}_b + 1)}{\text{Distance to Nearest Dock}}$$


Goal: Highlight stops with high bus traffic AND high bike activity (Success Stories) vs. High bus traffic / No bike activity (Gaps).

B. Temporal Handoff (Time Machine)

Compare monthly aggregates: POGOH Trips vs. PRT System Ridership.

Hypothesis Check: Does POGOH ridership dip in winter (Nov-Feb) more aggressively than Bus ridership?

C. Trip Archetypes (Clustering)

Use K-Means on POGOH trips based on:

Duration

Displacement (Straight line distance Start -> End)

Hour of Day
Labels: "Commuter" (AM/PM peak, A->B), "Leisure" (Weekend, Loop), "Last-Mile" (Short duration, near busway).

4. UI/UX SPECIFICATIONS

Theme: "Isomorphic Neo-Brutalism × Retro-Schematic"

Layout: Raw CSS Grid, Hard borders (3px solid black), Drop shadows (6px hard).

Palette: * Background: #F2F0E9 (Drafting Paper)

Primary: #2B4CFF (Blueprint Blue)

Accent: #FF4D00 (Safety Orange)

Success: #00B884 (Terminal Green)

Typography: Space Grotesk (Headers), JetBrains Mono (Data).

5. DELIVERABLE STRUCTURE (Single HTML)

The final output must be a self-contained HTML file with two distinct "Modes" toggled via JS:

MODE 1: INTERACTIVE VIZ (The "Reuters" View)

Hero Map: Leaflet. Layers: Bus Routes (Lines), Stops (Circles), Bike Stations (Squares).

Stop Explorer: Click a stop -> Show radial chart of bike arrivals.

Equity Chloropleth: Neighborhoods colored by Bikes_Per_1000_Bus_Riders.

MODE 2: LAB NOTEBOOK (The "Methods" View)

Rendered Python code blocks (syntax highlighted).

Markdown commentary explaining the "Why" behind the cleaning logic.

Static matplotlib outputs for validation (e.g., Histogram of Trip Durations).

6. TECH STACK

Backend: Python (Pandas, Geopandas, Shapely).

Frontend: HTML5, Vanilla JS, Leaflet.js, Chart.js.

Styling: Custom CSS (No heavy frameworks like Bootstrap).