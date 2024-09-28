from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import pandas as pd
import sklearn



app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('lstm_model.h5')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

        # Print all form data for debugging
     print(request.form)

     data = {
    'store_nbr': [int(request.form['store_nbr'])],
    'year': [int(request.form['year'])],
    'month': [int(request.form['month'])],
    'day': [int(request.form['day'])],
    'onpromotion': [int(request.form['onpromotion'])],
    'holiday_type': [request.form['holiday_type']],
    'locale': [request.form['locale']],
    'locale_name': [request.form['locale_name']],
    'dcoilwtico': [float(request.form['dcoilwtico'])],
    'transferred': [1 if request.form['transferred'] == 'True' else 0],
    'city': [request.form['city']],
    'state': [request.form['state']],
    'store_type': [request.form['store_type']],
    'day_of_week': [int(request.form['day_of_week'])],
    'transactions': [int(float(request.form['transactions']))],
    'cluster': [int(request.form['cluster'])]
     }
     input_df = pd.DataFrame(data)
     input_df = input_df[['city','cluster', 'day','day_of_week','dcoilwtico','holiday_type','locale','locale_name','month','onpromotion','state','store_nbr','store_type','transactions','transferred','year']]
     features = input_df.columns
     input_data = input_df[features].values
     print(input_data)
     scaler = joblib.load('scaler.pkl')     # change directory according to your device
     input_data = scaler.transform(input_data)
     input_data = input_data.reshape((1, 1, 16))

     # Log the preprocessed input
     print(f"Preprocessed Input: {input_data}")

     # Make the prediction
     prediction = model.predict(input_data)
     scaler_y = joblib.load('scaler_y.pkl')   # change directory according to your device
     prediction_original = scaler_y.inverse_transform(prediction)


     # Log the prediction
     print(f"Prediction: {prediction_original}")

     # Extract and format the prediction
     predicted_sales = prediction_original[0][0]
     prediction_text = f"Predicted Sales: ${predicted_sales:,.2f}"

     # Log the prediction text
     print(f"Prediction Text: {prediction_text}")

     return render_template('index.html', prediction_text=prediction_text)



if __name__ == "__main__":
    print(model.summary())  # Print model summary to verify input shape
    app.run(debug=True)