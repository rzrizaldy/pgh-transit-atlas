# Pittsburgh Transit Atlas 🚌 🚴

**Rizaldy Utomo** | Carnegie Mellon University
*Public Policy, Analytics, AI Management*

---

An interactive exploratory data analysis (EDA) of Pittsburgh's multimodal transit network, analyzing 550,000+ bikeshare trips and 7,000+ bus stops to understand how micro-mobility integrates with public transit.

**[👉 View Interactive Dashboard](https://rzrizaldy.github.io/pgh-transit-atlas/)**
**[📄 View Static Analysis Report](https://rzrizaldy.github.io/pgh-transit-atlas/final_project_eda_py_rutomo.html)**

---

## Project Overview

This project investigates the "Last Mile" connectivity problem in Pittsburgh. By combining POGOH bikeshare trip data with PRT bus ridership data, we identify gaps in the network and propose data-driven policy interventions.

### Key Findings

1. **Academic Seasonality:** Ridership fluctuates 9× between semester peaks and winter breaks, requiring dynamic fleet management.
2. **The Connectivity Gap:** Only **11.5%** of bus stops are within a 5-minute walk (400m) of a bikeshare station.
3. **Two Cities:** A massive divergence exists between the "Campus Corridor" (68% of trips, highly seasonal) and the "City Network" (year-round commuter stability).

## Repository Structure

This repo contains the full ETL pipeline and visualization codebase.

* `index.html`: The main interactive dashboard (D3.js + Leaflet).
* `final_project_eda_py_rutomo.ipynb`: The core Python notebook containing all data cleaning, K-Means clustering, and statistical analysis.
* `etl.py`: Python script that processes raw Excel/CSV files into optimized JSON for the web dashboard.
* `dataset/`: Raw data files (POGOH Trip Data Nov '24 - Oct '25, PRT Stop Usage).
* `static_viz/`: Generated charts (Seaborn/Matplotlib) used in the report.

## Technical Stack

* **Python:** Pandas (ETL), Scikit-Learn (K-Means Clustering), Seaborn (Static Viz).
* **Web:** HTML5/CSS3 (Neo-Brutalist Design), D3.js (Charts), Leaflet.js (Mapping).
* **Analysis:** Unsupervised Learning for trip archetyping, Haversine formulas for geospatial distance.

## Reproducibility

1. Clone this repo.
2. Install requirements: `pandas`, `numpy`, `matplotlib`, `seaborn`, `folium`, `scikit-learn`, `openpyxl`.
3. Run `python3 etl.py` to regenerate the `data.js` and `processed_data/` files from raw sources.
4. Open `final_project_eda_py_rutomo.ipynb` to step through the exploratory analysis.

---

*Note: This project was developed as a final submission for (90-800) Exploratory Data Analysis (EDA) and Visualization with Python at Heinz College, CMU.*
