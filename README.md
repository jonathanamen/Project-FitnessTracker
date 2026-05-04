# Project Fitness Tracker — Exercise Event Classification

A Python-based machine learning project that processes wearable telemetry 
data from a Fitbit device to automatically classify and track workout activity.

## Overview

This project ingests raw sensor telemetry from a Fitbit wearable and applies 
machine learning models to identify and log workout events, including exercise 
type, rep count, set count, and session date — eliminating the need for manual 
workout logging.

## Capabilities

- **Exercise Classification** — Identifies exercise type from raw sensor data
- **Rep Counting** — Detects and counts individual repetitions per exercise
- **Set Tracking** — Groups reps into sets and tracks set count per session
- **Session Logging** — Records exercise events with date and timestamps

## Tech Stack

- **Language:** Python
- **Data Source:** Fitbit telemetry data
- **Approach:** Machine learning on wearable sensor data

## Project Structure

```
Project-FitnessTracker/
├── data/           # Raw and processed Fitbit telemetry data
├── models/         # Trained models and predictions
├── notebooks/      # Exploration and development notebooks
├── src/            # Source code
├── reports/        # Analysis and figures
└── references/     # Data dictionaries and documentation
```

## Status

Active development. Rep counter and exercise classification under refinement.

## Next Improvements

- Improve rep counter accuracy
- Expand exercise type classification coverage
- Add real-time inference capability
- Build visualization dashboard for workout history