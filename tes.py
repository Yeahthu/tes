import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Initialize session state for data storage and model selection
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None

# Create two tabs
tab1, tab2 = st.tabs(["Generate Data and Choose Model", "Show Data Table and Predictions"])

# Content for Tab 1: Data Generation and Model Selection
with tab1:
    st.header("Generate Data and Choose Model")
    
    num_samples = st.slider("Number of samples", min_value=1, max_value=1000, value=10)
    generate_button = st.button("Generate Data")
    
    if generate_button:
        # Generate random data
        timestamp = pd.date_range(start='2022-01-01', periods=num_samples, freq='D')
        data = {
            'Date': timestamp,
            'Temperature': np.random.uniform(20, 35, num_samples),
            'Humidity': np.random.uniform(30, 90, num_samples),
            'Wind Speed': np.random.uniform(0, 15, num_samples),
        }
        df = pd.DataFrame(data)
        st.session_state['data'] = df
        st.session_state['predictions'] = None
        st.session_state['metrics'] = None
        st.success("Data generated successfully!")

    st.subheader("Choose Machine Learning Model")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Linear Regression"):
            st.session_state['model'] = LinearRegression()
    with col2:
        if st.button("Random Forest"):
            st.session_state['model'] = RandomForestRegressor()
    with col3:
        if st.button("Support Vector Machine"):
            st.session_state['model'] = SVR()
    with col4:
        if st.button("Decision Tree"):
            st.session_state['model'] = DecisionTreeRegressor()
    with col5:
        if st.button("K-Nearest Neighbors"):
            st.session_state['model'] = KNeighborsRegressor()
    
    if st.session_state['model'] is not None:
        st.success(f"{st.session_state['model']._class.name_} selected!")
        
    # Train the model and make predictions
    if st.session_state['data'] is not None and st.session_state['model'] is not None:
        X = st.session_state['data'][['Humidity', 'Wind Speed']]
        y = st.session_state['data']['Temperature']  # Example target
        model = st.session_state['model']
        model.fit(X, y)
        st.session_state['predictions'] = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, st.session_state['predictions'])
        mse = mean_squared_error(y, st.session_state['predictions'])
        st.session_state['metrics'] = {'R² Score': r2, 'Mean Squared Error': mse}
        
        # Predict for the next date
        next_date = st.session_state['data']['Date'].max() + timedelta(days=1)
        next_data = pd.DataFrame({
            'Humidity': [np.random.uniform(30, 90)],
            'Wind Speed': [np.random.uniform(0, 15)]
        })
        next_prediction = model.predict(next_data)
        
        st.subheader(f"Prediction for {next_date.date()}:")
        st.write(f"Predicted Temperature: {next_prediction[0]:.2f} °C")
        st.write(f"R2 Score model: {r2}")
        st.write(f"MSE Score model: {mse}")
        st.success("Model trained and predictions made!")

# Content for Tab 2: Show Data Table with Date Filter and Predictions
with tab2:
    st.header("Data Table and Predictions")
    if st.session_state['data'] is not None:
        start_date = st.date_input("Start date", datetime(2022, 1, 1))
        end_date = st.date_input("End date", datetime(2022, 1, 1) + pd.to_timedelta(num_samples, unit='d'))

        filtered_data = st.session_state['data'][(st.session_state['data']['Date'] >= pd.to_datetime(start_date)) & 
                                                 (st.session_state['data']['Date'] <= pd.to_datetime(end_date))]
        
        st.write("Here is the filtered data:")
        st.dataframe(filtered_data)
        
        
    else:
        st.warning("No data available. Please generate data in the 'Generate Data and Choose Model' tab.")
