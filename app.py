import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
import pickle
import openpyxl
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

@st.cache(allow_output_mutation=True)
def preprocess_data(dataset):
    dataset = dataset.T
    dataset.columns = dataset.iloc[0]
    dataset = dataset.drop(dataset.index[0])
    dataset.replace("-", np.nan, inplace=True)
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset = dataset.fillna(dataset.median())
    return dataset

def create_sequences(data, seq_length=60):
    x = []
    y = []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def plotly_test(dataset, sembako_type):
    fig = px.line(dataset, x=dataset.index, y=sembako_type, title='Grocery Price Prediction')
    return fig

def process_model(model, selection, rawdataset, sembako_type):
    dataset = preprocess_data(rawdataset)
    original = dataset.copy().reset_index().rename(columns={'index': 'Date'})
    
    sembako_data = dataset[sembako_type].values.reshape(-1, 1)
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = sc.fit_transform(sembako_data)
    
    x_test, y_test = create_sequences(scaled_dataset)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    if selection == "SVR":
        x_pred = x_test.reshape(x_test.shape[0], x_test.shape[1])
        predicted_price = model.predict(x_pred)
        predicted_price = sc.inverse_transform(predicted_price.reshape(-1, 1))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=original['Date'], y=sembako_data.flatten(), mode='lines', name='Grocery Price', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=original['Date'], y=predicted_price.flatten(), mode='lines', name='Predicted Grocery Price', line=dict(color='green')))
        
        mse = mean_squared_error(sembako_data[60:], predicted_price)
        r2 = r2_score(sembako_data[60:], predicted_price)
    else:
        predicted_price = model.predict(x_test)
        predicted_price = sc.inverse_transform(predicted_price)
        predicted_price = np.round(predicted_price, 2)
        
        newdataset = sembako_data[60:]
        newchart = pd.DataFrame(index=original.index[60:])
        newchart['Actual'] = newdataset.flatten()
        newchart['Predicted'] = predicted_price.flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=newchart.index, y=newchart['Actual'], mode='lines', name='Grocery Price', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=newchart.index, y=newchart['Predicted'], mode='lines', name='Predicted Grocery Price', line=dict(color='green')))
        
        mse = mean_squared_error(newdataset, predicted_price)
        r2 = r2_score(newdataset, predicted_price)
    
    return fig, mse, r2

def predict_future(model, rawdataset, years, sembako_type, seq_length=60):
    dataset = preprocess_data(rawdataset)
    sembako_data = dataset[sembako_type].values.reshape(-1, 1)
    
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled_dataset = sc.fit_transform(sembako_data)
    
    last_sequence = scaled_dataset[-seq_length:]
    predictions = []
    
    for _ in range(years * 12):
        x_test = np.reshape(last_sequence, (1, last_sequence.shape[0], 1))
        predicted_price = model.predict(x_test)
        predicted_price = sc.inverse_transform(predicted_price)
        predictions.append(predicted_price[0, 0])
        last_sequence = np.append(last_sequence[1:], sc.transform(predicted_price.reshape(-1, 1)), axis=0)
    
    last_date = pd.to_datetime(rawdataset.columns[-1])
    future_dates = [last_date + timedelta(days=30*i) for i in range(1, years * 12 + 1)]
    
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted': predictions})
    
    return future_df

st.title("Grocery Price Prediction")
st.write("Predicting average grocery prices using LSTM, GRU, and SVR models")

uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")
if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file, index_col=None)
    
    st.write("Dataset:")
    st.write(dataset)
    
    st.write("Dataset Preprocessing:")
    preprocessed = preprocess_data(dataset)
    st.write(preprocessed)
    
    sembako_type = st.selectbox("Select commodity type", preprocessed.columns)
    
    st.write("Dataset Visualization:")
    plottest = plotly_test(preprocessed, sembako_type)
    st.plotly_chart(plottest)
    
    selection = st.selectbox("Select Model", ["LSTM", "GRU", "SVR"])
    years = st.number_input("Predict how many years into the future?", min_value=1, max_value=10, value=1)
    
    if st.button("Start Prediction"):
        if selection == "LSTM":
            with open('LSTM.pkl', 'rb') as f:
                model = pickle.load(f)
        elif selection == "GRU":
            with open('GRU.pkl', 'rb') as f:
                model = pickle.load(f)
        elif selection == "SVR":
            with open('SVR.pkl', 'rb') as f:
                model = pickle.load(f)
        
        result, mse, r2 = process_model(model, selection, dataset, sembako_type)
        st.plotly_chart(result)
        st.write("Mean Squared Error (MSE): ", mse)
        st.write("R-squared (R2): ", r2)
        
        st.write(f"Predicting {years} years into the future:")
        future_df = predict_future(model, dataset, years, sembako_type)
        st.write(future_df)
        
        future_fig = px.line(future_df, x='Date', y='Predicted', title='Future Grocery Price Prediction')
        st.plotly_chart(future_fig)
