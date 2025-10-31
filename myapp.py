import streamlit as st
import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Linear Regression Web Application")
st.subheader("Application Built by Anika")

# Sidebar
st.sidebar.header("Upload CSV Data or Use Sample")
use_example = st.sidebar.checkbox("Use Example Dataset")
