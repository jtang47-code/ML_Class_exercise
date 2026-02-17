import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(page_title="Iris EDA", layout="wide")

# Title and description
st.title("🌸 Iris Dataset - Exploratory Data Analysis")
st.markdown("A simple analysis tool to explore the iris flower dataset")

# Load the iris dataset
@st.cache_data
def load_data():
    """Load the iris dataset from scikit-learn"""
    iris_data = load_iris()
    iris_dataframe = pd.DataFrame(
        iris_data.data,
        columns=iris_data.feature_names
    )
    iris_dataframe['target'] = iris_data.target
    iris_dataframe['species'] = iris_dataframe['target'].map({
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    })
    return iris_dataframe

# Load data
iris_df = load_data()

# Display dataset information
st.header("📊 Dataset Overview")

# Show first few rows
st.subheader("First Rows of Data")
rows_to_display = st.slider(
    "Number of rows to display:",
    min_value=1,
    max_value=20,
    value=5
)
st.dataframe(iris_df.head(rows_to_display), use_container_width=True)

# Show summary statistics
st.subheader("Summary Statistics")
st.dataframe(iris_df.describe(), use_container_width=True)

# Show dataset shape and info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", iris_df.shape[0])
with col2:
    st.metric("Total Columns", iris_df.shape[1])
with col3:
    st.metric("Missing Values", iris_df.isnull().sum().sum())

# Get numeric columns (excluding target)
numeric_columns = iris_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns = [col for col in numeric_columns if col != 'target']

# Visualization section
st.header("📈 Visualizations")

# Allow user to select columns for analysis
st.subheader("Column Selection")
selected_columns = st.multiselect(
    "Select numeric columns to visualize:",
    options=numeric_columns,
    default=numeric_columns[:2]  # Default to first two columns
)

# Create visualizations only if columns are selected
if selected_columns:
    # Histogram section
    st.subheader("Distribution Analysis (Histogram)")
    histogram_column = st.selectbox(
        "Select a column for histogram:",
        options=selected_columns,
        key="histogram_column"
    )
    
    # Create histogram with species color coding
    histogram = px.histogram(
        iris_df,
        x=histogram_column,
        color='species',
        nbins=20,
        title=f"Distribution of {histogram_column}",
        labels={histogram_column: histogram_column, 'species': 'Iris Species'},
        barmode='overlay'
    )
    histogram.update_layout(height=400)
    st.plotly_chart(histogram, use_container_width=True)
    
    # Scatter plot section
    st.subheader("Relationship Analysis (Scatter Plot)")
    
    # Ensure we have at least 2 columns for scatter plot
    if len(selected_columns) >= 2:
        scatter_col1, scatter_col2 = st.columns(2)
        
        with scatter_col1:
            x_column = st.selectbox(
                "Select X-axis column:",
                options=selected_columns,
                key="scatter_x"
            )
        
        with scatter_col2:
            y_column = st.selectbox(
                "Select Y-axis column:",
                options=selected_columns,
                index=1 if len(selected_columns) > 1 else 0,
                key="scatter_y"
            )
        
        # Create scatter plot with species color coding
        scatter_plot = px.scatter(
            iris_df,
            x=x_column,
            y=y_column,
            color='species',
            title=f"{x_column} vs {y_column}",
            labels={x_column: x_column, y_column: y_column, 'species': 'Iris Species'},
            size_max=8
        )
        scatter_plot.update_layout(height=400)
        st.plotly_chart(scatter_plot, use_container_width=True)
    else:
        st.warning("Please select at least 2 columns to create a scatter plot.")
else:
    st.info("Please select at least one column to visualize the data.")

# Footer with dataset info
st.divider()
st.markdown(
    """
    **About the Iris Dataset:**
    - Total samples: 150
    - Features: Sepal length, Sepal width, Petal length, Petal width
    - Target classes: 3 iris species (setosa, versicolor, virginica)
    - Source: UCI Machine Learning Repository / Scikit-learn
    """
)
