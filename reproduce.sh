#!/bin/bash
set -e

echo "Creating virtual environment..."
python -m venv env
source env/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Building split indices..."
python -m src.split_construction

echo "Engineering features..."
python -m src.features.build_day_ahead_features

echo "Training models..."
python -m src.models.day_ahead_lin_xgb
python -m src.models.day_ahead_complete_xgb

echo "Evaluating model errors..."
python -m src.results.day_ahead_results

echo "Running Monte Carlo simulations (this will take a while)..."
python -m src.results.day_ahead_temp_uncertainty_filter_none
python -m src.results.day_ahead_temp_uncertainty_filter_top_10_pct_load
python -m src.results.day_ahead_temp_uncertainty_filter_season

echo "Running hourly Monte Carlo simulation (this will take a while)..."
python -m src.results.day_ahead_temp_uncertainty_filter_hour

echo "Generating plots..."
python -m src.results.create_plots

echo "Done. Plots saved to /plots."

echo "Deactivating virtual environment..."
deactivate
