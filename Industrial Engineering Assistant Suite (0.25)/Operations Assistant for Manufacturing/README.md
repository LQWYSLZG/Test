---
title: Operations Assistant for Manufacturing
emoji: 🏭
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🏭 Operations Assistant for Manufacturing

A shop floor execution analytics dashboard for production managers, industrial engineers, and plant operators.

## Features

- **Time & Capacity** — Takt time calculator, cycle time analysis per operation and workstation
- **Line Performance** — Line balancing, bottleneck detection, Value Stream summary (VA vs NVA)
- **OEE & Downtime** — OEE dashboard (Availability × Performance × Quality), downtime Pareto, SMED changeover analysis
- **Throughput & Shifts** — Throughput analysis (units/hr), production rate vs takt, shift planning calculator

## Required Columns

### Time Study File (for Line Balancing & Cycle Time)
- `Workstation` — name or ID of the workstation
- `Operation` — operation or task name
- `Cycle Time (s)` — time per unit in seconds

### Downtime Log File (for OEE & Downtime)
- `Machine` — machine or line name
- `Duration (min)` — downtime duration in minutes
- `Category` *(optional)* — reason/failure category
- `Date` *(optional)* — enables trend analysis
- `Shift` *(optional)*

### Production Log File (for OEE & Throughput)
- `Machine` — machine or line name
- `Planned Hours` — scheduled production hours
- `Run Hours` — actual run hours
- `Good Units` — units passing quality check
- `Total Units` — total units produced
- `Ideal Cycle Time (s)` *(optional)* — enables Performance calculation
- `Date` *(optional)* — enables trend analysis
- `Shift` *(optional)*

## Notes

- Column names are flexible — common synonyms are automatically recognised
- Multiple files and multiple sheets per file are supported
- Sheet type (time study / downtime / production) is auto-detected
- No data is stored or shared — all processing is in-session only
