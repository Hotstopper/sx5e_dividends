# sx5e_dividends

Forecasting EUROSTOXX 50 dividends per share (DPS) using both traditional time-series and machine-learning approaches.  
The project compares a classical **ARIMA** model against a **hybrid residual-learning XGBoost** model trained on firm-level financial fundamentals, with a discussion on implications for EMH.

## Repository Structure
```
sx5e_dividends/
├── results/
│   ├── arima_evaluation_metrics.csv/   # Evaluation metrics of ARIMA model forecasts
│   ├── arima_rolling_by_ticker.csv/    # ARIMA forecast results
|   ├── preds.csv/                      # XGBoost forecast results
|   ├── arima_forecast.png/             # ARIMA model forecast percentage error distribution
│   └── model_forecast.png/             # XGBoost model forecast percentage error distribution
|
├── arima.ipynb                         # ARIMA model forecasting
├── model.ipynb                         # XGBoost model forecasting
|
├── dividends_data.csv                  # EUROSTOXX 50 dividends per share data
├── SX5E.csv                            # Financial fundamentals data of EUROSTOXX 50 Companies
|
├── .gitignore                          # Ignored files
└── README.md                           # Project overview and instructions
```

## Data
**`dividends_data.csv`**  
Contains historical DPS values for EUROSTOXX 50 constituents, structured by ticker and year.

**`SX5E.csv`**  
Contains firm-level accounting and market variables (EPS, FCF per share, revenue growth, etc.), aligned to the same panel. Used as input features for the XGBoost residual-learning model.

## Forecasting
### **ARIMA (`arima.ipynb`)**

- Conducts **rolling one-step-ahead forecasts** of DPS for each ticker using a univariate ARIMA (1,1,2) models trained on the previous 10 years.
- Generates per-ticker and aggregate metrics (MAE, RMSE, MAPE).  
- Saves forecasts to `results/arima_rolling_by_ticker.csv` and evaluation summaries to `results/arima_evaluation_metrics.csv`.  
- Produces `results/arima_forecast.png`, a distribution of ARIMA forecast percentage errors.

---

### **XGBoost Hybrid Model (`model.ipynb`)**

- Implements a **residual-learning hybrid**:
  $ \hat{y} = a\,F + b\,\hat{r}, \quad \hat{r} = f_{\text{XGB}}(X_{\setminus F}) $
  where $ F $ is the dividend-futures baseline and $ f_{\text{XGB}}\ $ learns deviations using firm-level fundamentals.
- Trains on data up to 2022 and tests on 2023–2024.  
- Produces out-of-fold residual predictions to avoid leakage and fits optimal combination weights a and b based on quantile regression.  
- Outputs:
  - `results/preds.csv` (Actual, Futures, XGBoost, and Hybrid predictions)
  - `results/model_forecast.png` (percentage-error comparison between Futures and Hybrid models)

## Results and Discussion
The ARIMA model’s lack of predictive gain supports the weak-form EMH, as past dividend patterns hold no exploitable information.
The pure ML model’s failure to outperform futures aligns with the semi-strong EMH, indicating that public fundamentals are already priced in.
In the hybrid model, the residual coefficient b < 0 reflects noise, while a = 1.03 suggests a slight, systematic underestimation of dividends by futures in 2023–2024.