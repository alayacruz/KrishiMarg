# 🚛 KrishiMarg
# 🌾 Problem Statement & Impact

## 📌 Why We Chose This Problem

India faces a critical paradox — while being one of the largest producers of agricultural goods, a significant portion of fruits and vegetables is lost post-harvest due to inefficient transportation, lack of condition-aware routing, and poor cold-chain decisions. At the same time, many regions experience food shortages. This imbalance highlights a major gap in the agricultural supply chain, especially during transit.

## 🚀 Our Solution: KrishiMarg

**KrishiMarg** is designed to bridge this gap by making crop transportation smarter and more reliable. It takes into account key factors such as:

- 🌡️ Temperature  
- 💧 Humidity  
- 🌦️ Climate conditions  
- 🛣️ Road conditions  
- 📍 Distance  

Using these parameters, it computes the **optimal route** that ensures crops are transported in the safest and most efficient way possible.

## 🇮🇳 Impact on Bharat

By enabling condition-aware routing, KrishiMarg significantly reduces spoilage of perishable goods during transit. This leads to:

- 📉 Reduction in food wastage  
- 🚚 Faster and safer delivery of produce  
- 🌍 Better distribution of food across regions  
- 👨‍🌾 Increased income stability for farmers  
- 🛒 Improved availability for consumers  

Ultimately, KrishiMarg contributes to building a more efficient, sustainable, and resilient agricultural ecosystem in Bharat.




## ⚙️ End-to-End Workflow

### 1. User Input Layer

The workflow begins with user-provided inputs collected via a Streamlit interface:

- Source location (origin city)
- Destination location
- Crop name (Tomato, Chili, Garlic, Onion, Orange, Soybean, Mustard, Groundnut, Rice)

---

### 2. Geospatial Processing

- Input locations are converted into latitude–longitude coordinates using OpenStreetMap Nominatim API.
- This enables downstream routing and environmental data mapping.

---

### 3. Route Generation (OSRM)

- The system queries the OSRM routing engine to generate:
  - Primary route
  - Alternative routes

Each route includes:
- GeoJSON geometry
- Distance (meters)
- Duration (seconds)

---

### 4. Waypoint-Based Sampling

- Each route is divided into waypoints at ~10 km intervals.
- For each waypoint:
  - Geographic coordinates are extracted
  - Estimated arrival time is calculated based on route progression

---

### 5. Weather Data Integration

- Weather data is fetched using Open-Meteo API for all waypoints.
- Parameters collected:
  - Temperature (°C)
  - Relative Humidity (%)

This creates a spatio-temporal environmental profile of each route.

---

### 6. Feature Extraction

For each route, the following features are computed:

- Average Temperature
- Maximum Temperature
- Minimum Temperature
- Average Humidity
- Distance (km)
- Travel Time (hours)

---

### 7. Spoilage Risk Computation

#### 7.1 Weighted Regression Formula

A deterministic weighted scoring function is applied:

Spoilage Score =
0.3162 × Max Temp +
0.2195 × Distance +
0.2140 × Avg Humidity +
0.1260 × Min Temp +
0.1243 × Avg Temp

This produces a continuous risk score for each route.

---

### 8. Route Evaluation & Ranking

- All routes are evaluated using computed risk scores.
- The optimal route is selected as the one with the minimum spoilage risk.

---


### 10. AI Recommendation Engine

#### 10.1 LLM-Based Explanation

- Model used: Sarvam-M
- Generates:
  - Natural language explanation
  - Route comparison
  - Justification of optimal route


---

### 11. Multilingual Processing

#### Translation
- Uses Sarvam translation API
- Supports 8 Indian languages:
  - English, Hindi, Marathi, Tamil, Telugu, Kannada, Bengali, Gujarati

#### Text-to-Speech
- Model: Bulbul v3
- Generates audio output of route recommendation

---

### 12. Visualization Layer

#### Map Visualization
- Implemented using Folium
- Displays:
  - All candidate routes
  - Color-coded paths
  - Interactive tooltips with route metrics

#### Comparative Analysis
- Tabular comparison of:
  - Distance
  - Duration
  - Temperature metrics
  - Humidity
  - Risk score
  - Waypoints sampled

---

## 🧠 Key Technical Innovations

- Waypoint-based weather sampling (10 km granularity)
- Spatio-temporal route risk modeling
- Crop-specific spoilage intelligence
- Hybrid scoring (deterministic + ML-based)
- Multilingual AI explainability (text + audio)
- Databricks-native data pipeline

---

## 📊 System Architecture Summary

User Input  
→ Geocoding (OSM)  
→ Route Generation (OSRM)  
→ Waypoint Sampling  
→ Weather Data Fetching  
→ Feature Extraction  
→ Risk Computation (Regression / ML)  
→ Route Ranking  
→ Data Storage (Databricks)  
→ AI Recommendation (Sarvam M)  
→ Translation + Text-to-Speech  
→ Visualization + Output  


## 🚀 Final Output

The system provides:

- Optimal route (minimum spoilage risk)
- Quantitative route comparison metrics
- Visual route mapping
- Multilingual explanation (text + audio)

This ensures efficient, safe, and intelligent transportation of perishable agricultural goods.
# 📊 Databricks Features Documentation for Agri-Logistics System

## 🗂️ Unity Catalog Tables

| Table Name                  | Schema             | Description                                                     | Key Columns                                                                 |
|----------------------------|--------------------|-----------------------------------------------------------------|------------------------------------------------------------------------------|
| route_spoilage_analytics   | workspace.default  | Stores route simulation results for crop transport analysis     | crop, distance, temperatures, humidity, risk_scores, geometry               |
| crop_reference_data        | workspace.default  | Crop metadata for reference lookups                             | crop_type, temperature_base_values                                          |
| final_crops                | workspace.default  | Comprehensive crop profiles for logistics planning              | storage_conditions, ethylene_data, shelf_life, handling_requirements        |
| mp_crops_postharvest_1     | workspace.default  | Post-harvest crop data for cold chain decisions                 | chilling_injury_thresholds, transport_groups                                |

---

## 📁 Unity Catalog Volumes

| Volume Path                                                          | File Name                     | Used In                         | Purpose                                                  |
|----------------------------------------------------------------------|------------------------------|----------------------------------|----------------------------------------------------------|
| /Volumes/workspace/default/crops/                                    | mp_crops_postharvest (1).json | crop_conditions, user_input_step1 | Crop profile data for condition checks                  |
| /Volumes/workspace/default/crops/20crops/                            | final_crops.json              | regression_for_spoilage          | Training data for ML spoilage prediction model           |

---

## ⚙️ SQL Warehouses

| Warehouse ID        | Used In  | Purpose                                                                 |
|---------------------|----------|-------------------------------------------------------------------------|
| 2c3b173ddf7eb7ca    | app.py  | Executes interactive queries on Unity Catalog tables via WorkspaceClient API |

---

## 🔥 PySpark & Data Processing

| Component                  | Type                  | Used In              | Purpose                                                              |
|----------------------------|-----------------------|----------------------|----------------------------------------------------------------------|
| query_lat_lon              | Custom PySpark UDF    | Route planning       | Geocoding location names using OpenStreetMap API                    |
| Spoilage Training Pipeline | Pandas DataFrame      | regression_for_spoilage | ML model training using crop data from Unity Catalog volumes        |

---

## 📒 Key Notebooks & Applications

| Notebook / File           | Databricks Features Used                                      | Purpose                                                                 |
|---------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------|
| app.py                   | Unity Catalog Tables, SQL Warehouse, WorkspaceClient API      | Streamlit app for agri-logistics risk analysis & route recommendation   |
| regression_for_spoilage   | Unity Catalog Volumes, Pandas DataFrames                      | ML pipeline for spoilage prediction model training                      |
| crop_conditions           | Unity Catalog Volumes                                         | Reads crop profiles for threshold and condition checks                  |
| tester_routes             | PySpark UDF, OpenStreetMap API                                | Route planning with geocoding and OSRM integration                      |




## 🚀 Overview

This system leverages Databricks for building a scalable **Agri-Logistics Optimization Platform**:

- 📍 **Geospatial Processing** via PySpark UDFs  
- 🌾 **Crop Intelligence** using Unity Catalog tables & volumes  
- 🤖 **ML-based Spoilage Prediction** using Pandas pipelines  
- 🛣️ **Route Optimization** integrating OSRM & weather conditions  
- 📊 **Real-time Analytics** through SQL Warehouses  
- 🌐 **Application Layer** powered by Streamlit (`app2.py`)  

---

## 🧠 Key Capabilities

- End-to-end **crop transport risk analysis**
- **Dynamic route optimization** based on environmental conditions
- **Cold chain decision support** using post-harvest datasets
- Integration of **ML models for spoilage prediction**
- Centralized data governance using **Unity Catalog**

---
