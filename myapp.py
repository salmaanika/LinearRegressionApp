import streamlit as st
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklear.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Linear Regression Web Application")
st.subheader("Application Built by Anika")

# Sidebar
st.sidebar.header("Upload CSV Data or Use Sample")
