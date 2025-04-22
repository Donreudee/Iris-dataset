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
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

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

# Create sidebar
st.sidebar.image("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv", use_column_width=True)
st.sidebar.markdown("<h2 class='sub-header'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Model Performance", "Prediction"])

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('iris.csv')
    except:
        # Sample data if the file is not found
        data = {
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 
                      'setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        }
        df = pd.DataFrame(data)
        # Add some versicolor and virginica samples
        df = pd.concat([df, pd.DataFrame({
            'sepal_length': [7.0, 6.4, 6.9, 5.5, 6.5],
            'sepal_width': [3.2, 3.2, 3.1, 2.3, 2.8],
            'petal_length': [4.7, 4.5, 4.9, 4.0, 4.6],
            'petal_width': [1.4, 1.5, 1.5, 1.3, 1.5],
            'species': ['versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor']
        })])
        df = pd.concat([df, pd.DataFrame({
            'sepal_length': [6.3, 5.8, 7.1, 6.3, 6.5],
            'sepal_width': [3.3, 2.7, 3.0, 2.9, 3.0],
            'petal_length': [6.0, 5.1, 5.9, 5.6, 5.8],
            'petal_width': [2.5, 1.9, 2.1, 1.8, 2.2],
            'species': ['virginica', 'virginica', 'virginica', 'virginica', 'virginica']
        })])
    return df

df = load_data()

# Load models (with fallback options for demo/development)
@st.cache_resource
def load_models():
    models = {}
    scaler = None
    
    try:
        # Try to load pre-trained models
        model_files = os.listdir('models')
        for file in model_files:
            if file.endswith('.pkl'):
                model_name = file.replace('.pkl', '').replace('_', ' ')
                with open(f'models/{file}', 'rb') as f:
                    if file == 'scaler.pkl':
                        scaler = pickle.load(f)
                    else:
                        models[model_name] = pickle.load(f)
    except:
        # If models don't exist, show dummy message
        st.warning("Pre-trained models not found. Demo mode active.")
        # We'll handle prediction differently in this case
    
    return models, scaler

models, scaler = load_models()

# Home Page
if page == "Home":
    st.markdown("<h2 class='sub-header'>Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='info-text'>
        <p>This application demonstrates a <span class='highlight'>machine learning classification model</span> 
        for identifying iris flower species based on their measurements.</p>
        
        <p>The Iris dataset is a classic in the machine learning community, containing measurements 
        for 150 iris flowers of three different species:</p>
        <ul>
            <li><b>Setosa</b> - Found primarily in eastern Asia</li>
            <li><b>Versicolor</b> - Common in North America</li>
            <li><b>Virginica</b> - Native to eastern North America</li>
        </ul>
        
        <p><b>Business Application:</b> This classifier can help botanists, florists, and 
        researchers quickly identify iris species based on simple measurements, 
        saving time and improving accuracy in field research and commercial applications.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Display example images or plots
        species = df['species'].unique()
        colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
        
        fig, ax = plt.subplots(figsize=(5, 5))
        for species_name in species:
            subset = df[df['species'] == species_name]
            ax.scatter(subset['sepal_length'], subset['petal_length'], 
                      label=species_name, color=colors[species_name], alpha=0.7)
        
        ax.set_xlabel('Sepal Length (cm)')
        ax.set_ylabel('Petal Length (cm)')
        ax.set_title('Iris Species by Sepal and Petal Length')
        ax.legend()
        st.pyplot(fig)
    
    # Feature explanation
    st.markdown("<h2 class='sub-header'>Feature Explanation</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-text'>
    <p>The model uses four key measurements of iris flowers to make predictions:</p>
    <ul>
        <li><b>Sepal Length</b>: The length of the outer parts of the flower (cm)</li>
        <li><b>Sepal Width</b>: The width of the outer parts of the flower (cm)</li>
        <li><b>Petal Length</b>: The length of the inner parts of the flower (cm)</li>
        <li><b>Petal Width</b>: The width of the inner parts of the flower (cm)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Project navigation
    st.markdown("<h2 class='sub-header'>Explore the Application</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #E3F2FD; border-radius: 5px;'>
            <h3>Data Explorer</h3>
            <p>View dataset statistics and visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #E8F5E9; border-radius: 5px;'>
            <h3>Model Performance</h3>
            <p>Check accuracy and other metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #FFF3E0; border-radius: 5px;'>
            <h3>Prediction</h3>
            <p>Try the model with your own measurements</p>
        </div>
        """, unsafe_allow_html=True)

# Data Explorer Page
elif page == "Data Explorer":
    st.markdown("<h2 class='sub-header'>Dataset Exploration</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualizations", "Feature Analysis"])
    
    with tab1:
        st.markdown("### Dataset Head")
        st.dataframe(df.head())
        
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe())
        
        st.markdown("### Species Distribution")
        species_counts = df['species'].value_counts().reset_index()
        species_counts.columns = ['Species', 'Count']
        
        fig = px.bar(species_counts, x='Species', y='Count', color='Species',
                    text='Count', title='Distribution of Iris Species')
        fig.update_layout(xaxis_title='Species', yaxis_title='Count')
        st.plotly_chart(fig)
    
    with tab2:
        st.markdown("### Pairplot of Features")
        fig = sns.pairplot(df, hue='species', height=2.5)
        st.pyplot(fig)
        
        st.markdown("### 3D Scatter Plot")
        fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length',
                         color='species', size='petal_width',
                         title='3D Scatter Plot of Iris Features')
        fig.update_layout(scene=dict(xaxis_title='Sepal Length',
                                  yaxis_title='Sepal Width',
                                  zaxis_title='Petal Length'))
        st.plotly_chart(fig)
    
    with tab3:
        st.markdown("### Feature Distributions by Species")
        
        feature = st.selectbox("Select Feature", 
                              ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        fig = px.box(df, x='species', y=feature, color='species',
                   title=f'Distribution of {feature} by Species')
        st.plotly_chart(fig)
        
        st.markdown("### Feature Correlation Matrix")
        corr = df.drop('species', axis=1).corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
                       title='Feature Correlation Matrix')
        st.plotly_chart(fig)

# Model Performance Page
elif page == "Model Performance":
    st.markdown("<h2 class='sub-header'>Model Performance Analysis</h2>", unsafe_allow_html=True)
    
    if not models:
        st.warning("Pre-trained models not found. This is a demonstration view.")
        
        # Create example performance metrics for demonstration
        example_results = {
            'Logistic Regression': {'accuracy': 0.96, 'cv_mean': 0.95},
            'Decision Tree': {'accuracy': 0.93, 'cv_mean': 0.92},
            'Random Forest': {'accuracy': 0.97, 'cv_mean': 0.96},
            'SVM': {'accuracy': 0.98, 'cv_mean': 0.97}
        }
        
        # Display example metrics
        st.markdown("### Model Comparison")
        
        model_names = list(example_results.keys())
        test_accuracies = [example_results[m]['accuracy'] for m in model_names]
        cv_means = [example_results[m]['cv_mean'] for m in model_names]
        
        fig = go.Figure(data=[
            go.Bar(name='Test Accuracy', x=model_names, y=test_accuracies),
            go.Bar(name='Cross-Validation Mean', x=model_names, y=cv_means)
        ])
        fig.update_layout(barmode='group', title='Model Performance Comparison',
                         xaxis_title='Model', yaxis_title='Accuracy')
        st.plotly_chart(fig)
        
        # Example confusion matrix
        st.markdown("### Confusion Matrix (SVM Model)")
        
        example_cm = np.array([[16, 0, 0], [0, 15, 1], [0, 1, 17]])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(example_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['setosa', 'versicolor', 'virginica'],
                   yticklabels=['setosa', 'versicolor', 'virginica'], ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Example Confusion Matrix')
        st.pyplot(fig)
        
        st.markdown("""
        <div class='info-text'>
        <p><b>Note:</b> These are example metrics for demonstration purposes. 
        In the actual deployment, metrics would be calculated from trained models.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display actual model performance
        best_model_name = "SVM"  # Default, would be determined from actual training
        
        for name, model in models.items():
            if name == 'best model' and best_model_name not in models:
                best_model_name = name
        
        st.markdown(f"### Best Model: {best_model_name}")
        
        # Would use real metrics in actual deployment
        st.markdown("""
        <div class='info-text'>
        <p>Model performance metrics would be displayed here based on the trained models.</p>
        </div>
        """, unsafe_allow_html=True)

# Prediction Page
elif page == "Prediction":
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
                # Demo mode prediction based on simple rules
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
                
                # Show dummy probabilities
                species_list = ["setosa", "versicolor", "virginica"]
                proba_df = pd.DataFrame({
                    'Species': species_list,
                    'Probability': probabilities
                })
                
                fig = px.bar(proba_df, x='Species', y='Probability', color='Species',
                           text='Probability', title='Prediction Probabilities')
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig)
                
                st.info("This is a demonstration prediction. In the actual deployment, predictions would be made using the trained model.")
            else:
                # Scale the input features
                input_scaled = scaler.transform(input_data)
                
                # Make prediction using best model
                best_model = models.get('best model', next(iter(models.values())))
                prediction = best_model.predict(input_scaled)[0]
                probabilities = best_model.predict_proba(input_scaled)[0]
                
                st.success(f"Predicted species: {prediction.upper()}")
                
                # Show probabilities
                species_list = best_model.classes_
                proba_df = pd.DataFrame({
                    'Species': species_list,
                    'Probability': probabilities
                })
                
                fig = px.bar(proba_df, x='Species', y='Probability', color='Species',
                           title='Prediction Probabilities')
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig)
    
    with col2:
        st.markdown("### Visualization of Input")
        
        # Create a visualization of where this input falls relative to the dataset
        fig = px.scatter(df, x='petal_length', y='petal_width', color='species',
                       title='Your Input Compared to Dataset')
        
        # Add marker for input
        fig.add_scatter(x=[petal_length], y=[petal_width], 
                       mode='markers', marker=dict(size=15, symbol='star', 
                                                color='yellow', line=dict(width=2, color='black')),
                       name='Your Input')
        
        st.plotly_chart(fig)
        
        # Show input data as table
        st.markdown("### Your Input Summary")
        st.table(input_df)
        
        # Reference ranges
        st.markdown("### Reference Ranges by Species")
        reference = pd.DataFrame({
            'Measurement': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Setosa': ['4.3-5.8 cm', '2.3-4.4 cm', '1.0-1.9 cm', '0.1-0.6 cm'],
            'Versicolor': ['4.9-7.0 cm', '2.0-3.4 cm', '3.0-5.1 cm', '1.0-1.8 cm'],
            'Virginica': ['4.9-7.9 cm', '2.2-3.8 cm', '4.5-6.9 cm', '1.4-2.5 cm']
        })
        st.table(reference)

# Footer
st.markdown("""---""")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Project 2 - Data Modeling | GroupXX | 2025
</div>
""", unsafe_allow_html=True)
