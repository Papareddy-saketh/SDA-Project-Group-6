from fastapi import FastAPI
from fastapi.responses import FileResponse
import pandas as pd

# Loading precomputed forecasted sales
forecasted_sales = pd.read_csv('Forecasted_Sales.csv')

# Create FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Store Sales Forecast API!"}

# Endpoint to get forecasted sales for a given number of steps
@app.get("/predict/{steps}")
def get_predictions(steps: int):
    try:
        # Get the requested number of steps from the precomputed forecast
        forecast = forecasted_sales.head(steps) 
        return {"forecasted_sales": forecast['forecasted_sales'].tolist()} 
    
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Endpoint to download the forecasted sales as a CSV file
@app.get("/download_forecast")
def download_forecast():
    try:
        # Provide the CSV file for download
        file_path = 'Forecasted_Sales.csv'  # Ensure the file exists in the working directory
        return FileResponse(path=file_path, filename='Forecasted_Sales.csv', media_type='text/csv')
    
    except Exception as e:
        return {"error": f"An error occurred while trying to download the forecast: {e}"}

#use "uvicorn Arima:app --reload" for running
