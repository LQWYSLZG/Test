---
title: Inventory Assistant for Supply Chain
emoji: 🚛
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🚛 Inventory Assistant for Supply Chain

A demand analytics dashboard for inventory and supply chain professionals.

## Features

- **ABC Inventory Analysis** — rank items by demand contribution with XYZ stability classification
- **Demand Forecast** — adaptive forecasting using Prophet, Holt/ETS, Croston, and TBATS
- **Inventory Alerts & Optimization** — reorder points, stockout risk, and four optimization models
- **Customer Segmentation** — RFM analysis with churn detection and annual value estimation
- **Demand Health** — slow mover and dead stock detection with seasonal adjustment

## Required Columns (Demand History Files)

- `Item Name`
- `Quantity`
- `Invoice Date`
- `Customer Name`

## Optional Columns

- `Unit Price` or `Revenue` — enables revenue-based analysis and inventory value calculations
- `Current Stock` — enables reorder alerts and months-until-zero calculations
- `Warehouse` or `Location` — enables multi-location filtering
