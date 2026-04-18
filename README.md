# KrishiMarg

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
