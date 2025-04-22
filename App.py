# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:38:38 2025

@author: User
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Project2_GroupXX",
    page_icon="🌸",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #26A69A;
    }
    .info-text {
        font-size: 1rem;
    }
    .highlight {
        color: #FF7043;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>Iris Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>A Business Application for Flower Classification</h3>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('iris.csv')
    except:
        data = {
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
            'species': [
                'setosa', 'setosa', 'setosa',
                'versicolor', 'versicolor', 'versicolor',
                'virginica', 'virginica', 'virginica', 'virginica'
            ]
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

# Load models
@st.cache_resource
def load_models():
    models = {}
    scaler = None
    try:
        model_files = os.listdir('models')
        for file in model_files:
            if file.endswith('.pkl'):
                with open(f'models/{file}', 'rb') as f:
                    if file == 'scaler.pkl':
                        scaler = pickle.load(f)
                    else:
                        model_name = file.replace('.pkl', '').replace('_', ' ')
                        models[model_name] = pickle.load(f)
    except:
        st.warning("Pre-trained models not found. Demo mode active.")
    return models, scaler

models, scaler = load_models()

# Fix page to Prediction
page = "Prediction"

# Prediction Page
if page == "Prediction":
    st.markdown("<h2 class='sub-header'>Make Predictions</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Enter Flower Measurements")

        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5, 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        input_df = pd.DataFrame(input_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        if st.button("Predict Species"):
            if not models or not scaler:
                # Demo mode prediction
                if petal_length < 2.5:
                    prediction = "setosa"
                    probabilities = [0.95, 0.04, 0.01]
                elif petal_length < 5.0:
                    prediction = "versicolor"
                    probabilities = [0.01, 0.89, 0.10]
                else:
                    prediction = "virginica"
                    probabilities = [0.00, 0.12, 0.88]
                st.success(f"Predicted species: {prediction.upper()}")

                species_list = ["setosa", "versicolor", "virginica"]
                proba_df = pd.DataFrame({
                    'Species': species_list,
                    'Probability': probabilities
                })
                fig = px.bar(proba_df, x='Species', y='Probability', color='Species', text='Probability')
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig)

                st.info("This is a demonstration prediction. In the actual deployment, predictions would be made using the trained model.")
            else:
                input_scaled = scaler.transform(input_data)
                best_model = models.get('best model', next(iter(models.values())))
                prediction = best_model.predict(input_scaled)[0]
                probabilities = best_model.predict_proba(input_scaled)[0]

                st.success(f"Predicted species: {prediction.upper()}")

                species_list = best_model.classes_
                proba_df = pd.DataFrame({
                    'Species': species_list,
                    'Probability': probabilities
                })
                fig = px.bar(proba_df, x='Species', y='Probability', color='Species')
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig)

    with col2:
        st.markdown("### Visualization of Input")
        fig = px.scatter(df, x='petal_length', y='petal_width', color='species', title='Your Input Compared to Dataset')
        fig.add_scatter(x=[petal_length], y=[petal_width], mode='markers',
                        marker=dict(size=15, symbol='star', color='yellow', line=dict(width=2, color='black')),
                        name='Your Input')
        st.plotly_chart(fig)

        st.markdown("### Your Input Summary")
        st.table(input_df)

        st.markdown("### Reference Ranges by Species")
        reference = pd.DataFrame({
            'Measurement': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Setosa': ['4.3-5.8 cm', '2.3-4.4 cm', '1.0-1.9 cm', '0.1-0.6 cm'],
            'Versicolor': ['4.9-7.0 cm', '2.0-3.4 cm', '3.0-5.1 cm', '1.0-1.8 cm'],
            'Virginica': ['4.9-7.9 cm', '2.2-3.8 cm', '4.5-6.9 cm', '1.4-2.5 cm']
        })
        st.table(reference)

st.markdown("""---""")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Project 2 - Data Modeling | GroupXX | 2025
</div>
""", unsafe_allow_html=True)
