import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression 
import warnings
from io import BytesIO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. DATA GENERATION AND MODEL TRAINING FUNCTIONS (Integrated) ---

# @st.cache_data is used to run this expensive function only once
@st.cache_data
def generate_and_train_data(seed=42):
    N_RECORDS = 1000
    N_YEARS = 10
    N_MONTHS = N_YEARS * 12
    START_DATE = '2015-01-01'
    np.random.seed(seed)
    
    df = pd.DataFrame()
    
    # Data Generation (Fix for DatetimeIndex error included)
    date_series = pd.to_datetime(pd.date_range(start=START_DATE, periods=N_MONTHS, freq='MS').repeat(8), errors='coerce').dropna().to_series().sample(N_RECORDS, replace=True, random_state=seed)
    df['Date'] = date_series.values
    df['Month'] = df['Date'].dt.month
    
    df['Kharif_Rainfall_mm'] = np.random.normal(loc=700, scale=150, size=N_RECORDS)
    df['Avg_Max_Temp_C'] = 25 + 10 * np.sin(2 * np.pi * (df['Month'] - 3) / 12) + np.random.normal(0, 2, N_RECORDS)
    df['Fertilizer_N_Low'] = np.random.choice([0, 1], N_RECORDS, p=[0.3, 0.7])
    df['Pest_Incidence_Index'] = np.clip(10 * df['Avg_Max_Temp_C'] + np.random.randint(0, 30, N_RECORDS), 10, 100).astype(int)
    # Synthetic Water Req for Module 4
    df['Crop_Water_Req_mm'] = 5 + 0.8 * df['Avg_Max_Temp_C'] - 0.1 * df['Kharif_Rainfall_mm'] + np.random.normal(0, 1, N_RECORDS) 
    
    # YIELD (Target)
    df['Yield_Kg_per_Ha'] = np.clip((1000 + 0.5 * df['Kharif_Rainfall_mm']) + (-300 * df['Fertilizer_N_Low']) + np.random.normal(0, 200, N_RECORDS), 500, 2000).round(0)
    
    # PRICE (Target)
    msp_trend = np.linspace(4000, 7500, N_YEARS)
    # Build the mapping in chronological order so each year gets the
    # correct synthetic MSP value, independent of the random row order.
    years = sorted(df['Date'].dt.year.unique())
    msp_map = dict(zip(years, msp_trend.round(-1)))
    df['MSP_Quintal'] = df['Date'].dt.year.map(msp_map)
    df['Global_Price_USD'] = np.linspace(0.65, 0.95, N_RECORDS) + np.random.normal(0, 0.05, N_RECORDS)
    seasonal_factor = np.where(df['Month'].isin([11, 12, 1]), 0.95, 1.05)
    df['Market_Price_Modal'] = np.clip((df['MSP_Quintal'] * 0.5 + df['Global_Price_USD'] * 3000) * seasonal_factor + np.random.normal(0, 200), df['MSP_Quintal'] * 0.9, df['MSP_Quintal'] * 1.5).round(0)
    
    # Costs and Profit (for Module 3)
    df['Net_Profit_per_Ha'] = 23913 # Base profit from final analysis
    
    # MODEL 1: ARIMA (Price)
    monthly_price_series = df.set_index('Date')['Market_Price_Modal'].resample('MS').mean().dropna()
    arima_model = ARIMA(monthly_price_series, order=(2, 1, 1)).fit()
    
    # MODEL 2: Random Forest (Yield)
    yield_features = ['Kharif_Rainfall_mm', 'Avg_Max_Temp_C', 'Fertilizer_N_Low', 'Pest_Incidence_Index']
    X_yield = df[yield_features]
    Y_yield = df['Yield_Kg_per_Ha']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_model.fit(X_yield, Y_yield)
    
    # MODEL 4: Linear Regression (Water Req)
    water_features = ['Avg_Max_Temp_C', 'Kharif_Rainfall_mm']
    X_water = df[water_features]
    Y_water = df['Crop_Water_Req_mm']
    lr_model = LinearRegression()
    lr_model.fit(X_water, Y_water)
    
    return df, arima_model, rf_model, lr_model, monthly_price_series, yield_features, water_features

# --- 2. DASHBOARD LAYOUT (The Container) ---
def main():
    # Configure the page before making other Streamlit calls.
    st.set_page_config(layout="wide")

    # Load all models and data once
    df, arima_model, rf_model, lr_model, monthly_price_series, yield_features, water_features = generate_and_train_data()
    st.title("🌾 Telangana Cotton Prescriptive Analytics")
    st.markdown("### Integrated Dashboard for Yield, Price, and Operational Optimization")
    
    st.sidebar.header("Navigation")
    
    # Use tabs to organize the four modules
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 M1: Price Forecast", 
        "🌳 M2: Yield Risk & M3: Profit Prescription", 
        "💧 M4: Water Requirement Analysis",
        "⚙️ Simulation Tool"
    ])
