# Nairobi PM2.5 Air Quality Forecasting
## Production-Ready AR Model | Walk-Forward Validation | Interactive Streamlit Dashboard

This repository contains a fully interactive Streamlit dashboard for forecasting PM2.5 air pollution levels in Nairobi using a walk-forward autoregressive (AR) model.
It includes data processing, model evaluation, visualization, and a baseline comparison to demonstrate forecasting improvements.

## Features
  •	Autoregressive Walk-Forward Forecasting
  •	Full Model Evaluation: MAE, RMSE, R², Residual Analysis
	•	Interactive Plotly Visualizations
	•	Clean Dashboard UI (Streamlit)
	•	Baseline Model Comparison
	•	Easy Deployment to Streamlit Cloud / GitHub Pages

## Repository Structure
Air-Quality-Forecasting/
- │
- ├── app.py                             # Streamlit dashboard app
- ├── nairobi_pm25_final_results.csv     # Processed AR model results (test set)
- ├── requirements.txt                   # Python dependencies
- ├── README.md                          # Project documentation
- └── docs/                            

## Clone the repository
git clone https://github.com/Bartho-A/Air-Quality-Forecasting.git
cd Air-Quality-Forecasting

## Install dependencies
pip install -r requirements.txt

## Run Streamlit App
streamlit run app.py


