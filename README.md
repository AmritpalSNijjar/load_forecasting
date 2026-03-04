# Day-Ahead Load Forecasting Under Weather Forecast Uncertainty

This project explores day-ahead electricity load forecasting under realistic operational constraints, with a focus on model robustness when temperature forecasts are imperfect. Rather than evaluating models under observed temperatures, this analysis stress-tests three model classes under simulated temperature forecast uncertainty to identify which models remain reliable when inputs are noisy and how performance shifts during peak load periods.

## Project Report

A full write-up of the methodology, results, and discussion is available in [`report.pdf`](report.pdf).

## Motivation

In operational practice, day-ahead load forecasts rely on weather forecasts rather than realized temperatures. Models evaluated under observed temperatures may overstate real-world performance. This project addresses that gap by:

- Imposing realistic operational feature constraints: no short-horizon load lags, forecasted temperature only
- Applying Monte Carlo temperature perturbations to simulate day-ahead forecast uncertainty
- Evaluating model robustness not just on overall RMSE but specifically on peak-hour performance

## Dataset

Data was obtained from the **PG&E 2025 Energy Analytics Challenge** and consists of hourly electricity load and weather observations for the San Diego, California region spanning 2020–2022. The dataset is publicly available and included in this repository under `/data`.

Features used:
- Cyclical hour encodings, weekend and notable-day indicators
- Averaged temperature across five neighboring sites, 6-hour rolling temperature average, cooling and heating degree terms
- 24-hour and 48-hour lagged load features
- Short-horizon lags (e.g., t-1) were excluded to reflect day-ahead operational constraints

## Models

Three model classes were evaluated, each trained as 24 separate hourly models to capture intraday load dynamics:

- **Linear Regression**: baseline, ordinary least squares per hour
- **XGBoost**: gradient boosted trees with hour-specific hyperparameter tuning
- **Hybrid Linear + XGBoost**: linear model captures primary structure; XGBoost trained on out-of-sample residuals to learn nonlinear corrections

## Methodology

- **Train/test split**: trained on 2020–2021, evaluated on held-out 2022 test set
- **Cross-validation**: 5-split expanding-window time-series cross-validation within training period
- **Temperature uncertainty simulation**: Monte Carlo perturbations applied to temperature inputs at sigma = 1–5°F, with 100 simulations per uncertainty level; features recalculated for each simulation
- **Evaluation**: overall RMSE, top 10% highest load hours, seasonal partitions, hourly heatmaps

## Key Findings

- Under perfect temperature inputs, the hybrid model achieved the lowest RMSE (147.8 MW), outperforming standalone XGBoost (155.9 MW) and Linear Regression (156.9 MW)
- With increasing temperature forecast uncertainty, the hybrid model lost its advantage; at sigma >= 3°F, Linear Regression became the most robust model
- Focusing on the top 10% of highest load hours revealed dramatic shifts: XGBoost's RMSE increased roughly 78%, from 155.9 MW to 277.4 MW, while Linear Regression remained comparatively stable
- These results highlight a trade-off between baseline accuracy and robustness: more flexible models achieve lower error under perfect inputs but are more sensitive to temperature forecast noise

![Model Performance vs Temperature Forecast Uncertainty](figures/baseline_top10_rmse_uncertainty_plot.png)

## Repository Structure

```
load_repo/
├── data/                        # Hourly load and weather data, San Diego 2020-2022
├── notebooks/                   # Analysis notebooks
├── figures/                     # Output figures
├── report.pdf                   # Full project report
├── requirements.txt             # Python dependencies
└── README.md
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Limitations

- Temperature forecast errors are simulated synthetically and not calibrated to real NWP forecast error distributions
- Only temperature uncertainty is modeled; solar irradiance and other exogenous uncertainties are excluded
- The historical load-temperature relationship is assumed stable across the evaluation period
