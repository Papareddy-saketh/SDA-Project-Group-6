# -*- coding: utf-8 -*-
"""SDA_project_code_LSTM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12d-xQoXYL746nMv5SZ6JXERmxWa5r_Gt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('train.csv')
data_train

data_train.shape

data_train.info()

data_train.isnull().sum()

data_train.duplicated().sum()

data_oil = pd.read_csv('oil.csv')
data_oil.head()

data_oil.info()

data_oil.isnull().sum()

data_oil.duplicated().sum()

data_holidays = pd.read_csv('holidays_events.csv')
data_holidays

data_holidays.info()

data_holidays.isnull().sum()

data_holidays.duplicated().sum()

data_stores = pd.read_csv('stores.csv')
data_stores

data_stores['cluster'].unique()

data_stores.info()

data_stores.isnull().sum()

data_stores.duplicated().sum()

data_transactions = pd.read_csv('transactions.csv')
data_transactions.head()

data_transactions.info()

data_transactions.isnull().sum()

data_transactions.duplicated().sum()

#Create columns for year, month, date, day of week for easier aggregation
names = ['data_oil', 'data_holidays', 'data_stores', 'data_train', 'data_transactions']
for name, data in zip(names, [data_oil, data_holidays, data_stores,  data_train, data_transactions]):
    print()
    print('processing', name, '...')
    if 'date' in data.columns:
        #Convert from object to datetime
        data.date = pd.to_datetime(data.date)
        data['year'] = data.date.dt.year
        data['month'] = data.date.dt.month
        data['day'] = data.date.dt.day
        data['day_of_week'] = data.date.dt.dayofweek
        data['day_name'] = data.date.dt.strftime('%A')
    else:
        print('the date column does not exist in', name)

#merge the data

#Clone the df to compare after merging
data_train_original = data_train

#Merge the oil data
data_train = pd.merge(data_train, data_oil[['date', 'dcoilwtico']], on='date', how='left')

#Merge holidays data
data_holidays.rename(columns={'type': 'holiday_type'}, inplace=True)
data_train = pd.merge(data_train, data_holidays, on = ['date', 'day', 'month', 'year', 'day_of_week', 'day_name'], how = 'left')
#Merge the store data
data_stores.rename(columns={'type': 'store_type'}, inplace=True)
data_train = pd.merge(data_train, data_stores, on='store_nbr', how='left')

#Merge the transactions data
data_train = pd.merge(data_train, data_transactions, on = ['date', 'store_nbr', 'day', 'month', 'year', 'day_of_week', 'day_name'], how='left')

data_train['cluster'].unique()

data_train

import matplotlib.pyplot as plt
import seaborn as sns

# Grouping data by 'locale' and summing the 'sales'
locale_sales = data_train.groupby('locale')['sales'].sum().reset_index()

# Creating the barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='locale', y='sales', data=locale_sales)

# Adding labels and title
plt.xlabel('Locale')
plt.ylabel('Total Sales')
plt.title('Total Sales by Locale')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45)

# Displaying the plot
plt.show()

data_train['locale'].unique()

import matplotlib.pyplot as plt
import seaborn as sns

# Grouping data by 'locale' and summing the 'sales'
locale_sales = data_train.groupby('holiday_type')['sales'].sum().reset_index()

# Creating the barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='holiday_type', y='sales', data=locale_sales)

# Adding labels and title
plt.xlabel('holiday_type')
plt.ylabel('Total Sales')
plt.title('Total Sales by holiday')

# Rotating x-axis labels for better readability
plt.xticks(rotation=0)

# Displaying the plot
plt.show()

data_train['holiday_type'].unique()

import matplotlib.pyplot as plt
import seaborn as sns

# Grouping data by 'locale' and summing the 'sales'
locale_sales = data_train.groupby('locale_name')['sales'].sum().reset_index()

# Creating the barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='locale_name', y='sales', data=locale_sales)

# Adding labels and title
plt.xlabel('Locale_name')
plt.ylabel('Total Sales')
plt.title('Total Sales by Locale_name')

# Rotating x-axis labels for better readability
plt.xticks(rotation=90)

# Displaying the plot
plt.show()

data_train['locale_name'].unique()

import matplotlib.pyplot as plt
import seaborn as sns

# Grouping data by 'locale' and summing the 'sales'
locale_sales = data_train.groupby('city')['sales'].sum().reset_index()

# Creating the barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='city', y='sales', data=locale_sales)

# Adding labels and title
plt.xlabel('city')
plt.ylabel('Total Sales')
plt.title('Total Sales by city')

# Rotating x-axis labels for better readability
plt.xticks(rotation=90)

# Displaying the plot
plt.show()

data_train['city'].unique()

# Grouping data by 'locale' and summing the 'sales'
locale_sales = data_train.groupby('state')['sales'].sum().reset_index()

# Creating the barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='state', y='sales', data=locale_sales)

# Adding labels and title
plt.xlabel('state')
plt.ylabel('Total Sales')
plt.title('Total Sales by state')

# Rotating x-axis labels for better readability
plt.xticks(rotation=90)

# Displaying the plot
plt.show()

data_train['state'].unique()

# Grouping data by 'locale' and summing the 'sales'
locale_sales = data_train.groupby('store_type')['sales'].sum().reset_index()

# Creating the barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='store_type', y='sales', data=locale_sales)

# Adding labels and title
plt.xlabel('store_type')
plt.ylabel('Total Sales')
plt.title('Total Sales by Locale')

# Rotating x-axis labels for better readability
plt.xticks(rotation=0)

# Displaying the plot
plt.show()

data_train['store_type'].unique()

data_train

# Grouping the data by 'store_nbr' and 'date', summing 'sales' and aggregating other columns
aggregated_data = data_train.groupby(['store_nbr','year','month','day']).agg({
    'sales': 'sum',              # Sum sales
    'onpromotion': 'sum',        # Sum promotions (or use 'mean' if you want the average)
    'holiday_type': 'first',     # Keep the first value of holiday_type
    'locale': 'first',           # Keep the first value of locale (assuming it's consistent per store/date)
    'locale_name': 'first',      # Keep the first value of locale_name
    'dcoilwtico': 'mean',        # Take the mean of oil prices if they vary within a store/date
    'transferred' : 'first',
    'city': 'first',
    'state': 'first',
    'store_type': 'first',
    'day_of_week': 'first',
    'day_name': 'first',
    'transactions': 'first',
    'cluster': 'first',
}).reset_index()

# Renaming the 'sales' column for clarity
aggregated_data = aggregated_data.rename(columns={'sales': 'total_sales'})

# Display the first few rows of the new dataset
aggregated_data.head()

aggregated_data

cleaned_df = aggregated_data.dropna()

cleaned_df

# Step 1: Save the DataFrame to a CSV file
cleaned_df.to_csv('store_sales.csv', index=False)

# Step 2: Download the file (if you're in Google Colab)
from google.colab import files
files.download('store_sales.csv')

from sklearn.preprocessing import LabelEncoder

# Create a label encoder instance
label_encoder = LabelEncoder()

# Label encode the categorical columns directly
cleaned_df['locale'] = label_encoder.fit_transform(cleaned_df['locale'])
cleaned_df['locale_name'] = label_encoder.fit_transform(cleaned_df['locale_name'])
cleaned_df['city'] = label_encoder.fit_transform(cleaned_df['city'])
cleaned_df['state'] = label_encoder.fit_transform(cleaned_df['state'])
cleaned_df['store_type'] = label_encoder.fit_transform(cleaned_df['store_type'])
cleaned_df['holiday_type'] = label_encoder.fit_transform(cleaned_df['holiday_type'])

cleaned_df.reset_index(drop=True, inplace=True)
cleaned_df

cleaned_df.shape

cleaned_df = cleaned_df.drop('day_name', axis=1)

test_set = cleaned_df[cleaned_df['year'] == 2017]
test_set.reset_index(drop=True, inplace=True)
train_set = cleaned_df[cleaned_df['year'] < 2017]
train_set.reset_index(drop=True, inplace=True)

test_set

train_set

features = train_set.columns
train_x = train_set[features].values
train_y = train_set['total_sales'].values
test_x = test_set[features].values
test_y = test_set['total_sales'].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_log_error

# Normalize the features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Reshape the data for LSTM input
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

from sklearn.preprocessing import MinMaxScaler

# Normalize the target variable
scaler_y = MinMaxScaler()
train_y = scaler_y.fit_transform(train_y.reshape(-1, 1))
test_y = scaler_y.transform(test_y.reshape(-1, 1))

# Build the LSTM model
from keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(60, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))  # Return sequences for the next LSTM
model.add(Dropout(0.2))  # Optional: add dropout for regularization
model.add(LSTM(10))  # This LSTM will receive 2D input (batch_size, 50)
model.add(Dropout(0.2))
model.add(Dense(1))  # Final output layer


model.compile(loss='huber', optimizer=Adam(learning_rate=0.0001))

# Train the model
model.fit(train_x, train_y, epochs=20, batch_size=8, validation_data=(test_x, test_y), verbose=1)

# Make predictions on the validation set
lstm_preds = model.predict(test_x)

predicted_y_original = scaler_y.inverse_transform(lstm_preds)

# You can also inverse transform the actual test_y if you want to compare:
test_y_original = scaler_y.inverse_transform(test_y)

# Flatten predictions and true values for evaluation
lstm_preds_flat = predicted_y_original.flatten()
test_y_flat = test_y_original.flatten()

# Evaluate the LSTM model using RMSLE
rmsle = np.sqrt(mean_squared_log_error(test_y_flat, lstm_preds_flat))
print("LSTM RMSLE:", rmsle)

import numpy as np



# Compute the direction of change for actual and predicted values
actual_direction = np.sign(np.diff(test_y_flat))
predicted_direction = np.sign(np.diff(lstm_preds_flat))

# Calculate the number of correct directional predictions
correct_direction = np.sum(actual_direction == predicted_direction)

# Compute directional accuracy as a percentage
directional_accuracy = correct_direction / len(actual_direction) * 100

print(f"Directional Accuracy: {directional_accuracy:.2f}%")

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'actual' and 'predicted' are your arrays/lists of actual and predicted sales
# You can also use the inverse transformation if your predictions are in log scale

plt.figure(figsize=(10, 6))

plt.plot(lstm_preds_flat, label='Predicted Sales', color='red')
# Adding titles and labels
plt.title('Predicted Sales Over Time')
plt.xlabel('Time (Index or Date)')
plt.ylabel('Sales')
plt.legend()

# Display the plot
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(test_y_flat, label='Actual Sales', color='blue')
plt.show()