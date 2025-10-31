import streamlit as st
import pandas
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    
