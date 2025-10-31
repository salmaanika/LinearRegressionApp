import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Linear Regression Web Application")
st.subheader("Application Built by Anika")

# Sidebar
st.sidebar.header("Upload CSV Data or Use Sample")
use_example = st.sidebar.checkbox("Use Example Dataset")

# Load Data
if use_example: 
  df = sns.load_dataset('tips')
  df = df.dropna()
  st.success("Loaded sample dataset: 'tips'")
else:
  uploaded_file = st.sidebar.file_uploader("Upload Your CSV File", type=['csv'])
  if uploaded_file:
    df = pd.read_csv(uploaded_file)
  else:
    st.warning("Please Upload a CSV File or Use The Example Dataset")
    st.stop()
    
# Show Dataset
st.subheader("Dataset Preview")
st.write(df.head())


# Model Feature Selection and Training
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
  st.error("Need at least Two Numeric Column for Regression")
  st.stop()

target = st.selectbox("Select Target Variable", numeric_cols)
features = st.multiselect("Select input feature columns",[col for col in numeric_cols if col != target], default=([col for col in numeric_cols if col != target]))

if len(features) == 0:
  st.write("Please select at least one feature")
  st.stop()

df = df[features + [target]].dropna()

x = df[features]
y = df[target]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evolution")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R^2 Score: {r2:.2f}")

st.subheader("Make a Prediction")
input_data = {}
valid_input = True
for feature in features:
  st.text_input(f"Enter {feature} (numeric value)")
  try:
    if user_val.strip()=="":
      valid_iput = False
    else:
      input_data[feature] = float(user_val)
  except ValueError:
    valid_input = False

