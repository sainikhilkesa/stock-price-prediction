import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# convert date to numeric
import datetime

def convert_date_to_numeric(date):
  

  # Get the year, month, and day of the date.
  year = date.year
  month = date.month
  day = date.day

  # Convert the year, month, and day to numeric values.
  year_value = year * 365
  month_value = month * 30
  day_value = day

  # Calculate the numeric value of the date.
  date_value = year_value + month_value + day_value

  return date_value



# function for preprocessing
# function for preprocessing
def preprocess(date, open, high, low, volume):
    # Convert the date to a numeric value.
    date_value = convert_date_to_numeric(date)

    # Create a NumPy array containing the data.
    data = np.array([date_value, open, high, low, volume])

    # Create a MinMaxScaler object.
    scaler = MinMaxScaler()

    # Reshape the data to 2D and fit the scaler to the data.
    scaler.fit(data.reshape(1, -1))

    # Transform the data using the scaler.
    preprocessed_data = scaler.transform(data.reshape(1, -1))

    return preprocessed_data


# model training
#def model_train(date, open, high, low, close, adj_close, volume):


# Define a function to predict the price
def predict_price(date, open_price, low_price, high_price, volume):
    # Load the model
    model = pickle.load(open('RF_model.pkl', 'rb'))

    # Create a DataFrame from the input data
    df = pd.DataFrame({
        'date': [date],
        'open': [open_price],
        'low': [low_price],
        'high': [high_price],
        'volume': [volume]
    })

    # Make a prediction
    prediction = model.predict(df)[0]

    return prediction


st.title("Stock Price prediction : ")

# Create a form to collect the user input
with st.form('price_prediction_form'):
    date = st.date_input('Date')
    open_price = st.number_input('Open price')
    low_price = st.number_input('Low price')
    high_price = st.number_input('High price')
    volume = st.number_input('Volume')

    submit_button = st.form_submit_button('Predict price')

# If the user clicks the submit button, then call the model function to predict the price
if submit_button:
    if submit_button:
        inputs = preprocess(date, open_price, low_price, high_price, volume)
        reshape_arr = inputs.reshape(1, -1)
        prediction = predict_price(*reshape_arr)  # Remove the extra brackets
        st.write('Predicted price: {}'.format(prediction))

