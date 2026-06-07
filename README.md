# Airport Traffic Forecasting and Analytics Dashboard

An end-to-end machine learning platform that predicts airport flight traffic,
analyzes operational trends, explains predictions using SHAP,
and provides interactive monitoring and forecasting through Streamlit.

## Business Problem

Airports require accurate traffic forecasts to optimize
resource allocation, staffing, runway utilization,
and air traffic management.

This project predicts total flight traffic using
historical operational data and provides explainable
AI insights for decision-making.

## Dataset

Features:
- Airport ICAO
- Airport Name
- State
- Flight Departures
- Flight Arrivals
- IFR Departures
- IFR Arrivals
- Month
- Date

Target:
- Total Flight Traffic (FLT_TOT_1)

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Plotly
- Streamlit

## ML Pipeline

1. Data Cleaning
2. Feature Engineering
3. Time-Based Train-Test Split
4. One-Hot Encoding
5. XGBoost Training
6. Model Evaluation
7. SHAP Explainability

## Model Performance

| Metric | Value |
|----------|----------|
| MAE | 145 |
| RMSE | 220 |
| R² Score | 0.92 |

## Dashboard Features

- Traffic Analytics
- Airport Benchmarking
- Monthly Trends
- Forecasting
- SHAP Explainability
- Model Monitoring
- Drift Analysis
- Quantitative Analysis

## Key Insights

- Peak traffic occurs during summer months.
- IFR traffic strongly correlates with total traffic.
- Airport traffic exhibits clear monthly seasonality.
- SHAP analysis identified arrivals and departures as major drivers.

## Limitations

- Dataset contains only 2025 data.
- Long-term multi-year trends could not be modeled.
- Forecasts should be interpreted as short-term projections.

## Future Improvements

- Incorporate weather data.
- Add airline-level analysis.
- Train on multi-year historical datasets.
- Deploy real-time monitoring pipeline.

## Links

- Live Dashboard
- GitHub Repository
- Case Study