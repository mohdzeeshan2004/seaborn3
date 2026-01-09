import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Seaborn Datasets Explorer", layout="wide")

st.title("📊 Seaborn Datasets Explorer")
st.write("Explore and visualize built-in datasets from the Seaborn library")

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox(
    "Choose a dataset:",
    ["iris", "titanic", "tips", "flights", "diamonds", "penguins"]
)

# Load the selected dataset
@st.cache_data
def load_data(name):
    return sns.load_dataset(name)

try:
    df = load_data(dataset_name)

    # Display dataset information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Display the dataframe
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Display basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Data types information
    st.subheader("Data Types")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes,
            "Non-Null Count": df.count()
        }),
        use_container_width=True
    )

    # Visualization section
    st.subheader("Data Visualization")

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Visualization options
    col1, col2 = st.columns(2)

    with col1:
        chart_type = st.selectbox(
            "Select visualization type:",
            ["Scatter Plot", "Histogram", "Box Plot", "Violin Plot", "Heatmap"]
        )

    selected_col = None
    with col2:
        if chart_type != "Heatmap" and numeric_cols:
            selected_col = st.selectbox("Select column:", numeric_cols)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
            col_y = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
            sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax)
            ax.set_title(f"Scatter Plot: {col_x} vs {col_y}")
        else:
            st.warning("Need at least 2 numeric columns for scatter plot")

    elif chart_type == "Histogram":
        if selected_col:
            sns.histplot(data=df, x=selected_col, kde=True, ax=ax)
            ax.set_title(f"Histogram of {selected_col}")
        else:
            st.warning("No numeric columns available for histogram")

    elif chart_type == "Box Plot":
        if selected_col:
            if categorical_cols:
                cat_col = st.selectbox("Group by:", categorical_cols, key="boxplot_cat")
                sns.boxplot(data=df, x=cat_col, y=selected_col, ax=ax)
                ax.set_title(f"Box Plot of {selected_col} by {cat_col}")
            else:
                sns.boxplot(data=df, y=selected_col, ax=ax)
                ax.set_title(f"Box Plot of {selected_col}")
        else:
            st.warning("No numeric columns available for box plot")

    elif chart_type == "Violin Plot":
        if selected_col:
            if categorical_cols:
                cat_col = st.selectbox("Group by:", categorical_cols, key="violin")
                sns.violinplot(data=df, x=cat_col, y=selected_col, ax=ax)
                ax.set_title(f"Violin Plot of {selected_col} by {cat_col}")
            else:
                sns.violinplot(data=df, y=selected_col, ax=ax)
                ax.set_title(f"Violin Plot of {selected_col}")
        else:
            st.warning("No numeric columns available for violin plot")

    elif chart_type == "Heatmap":
        if numeric_cols:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
        else:
            st.warning("No numeric columns available for heatmap")

    st.pyplot(fig)
    plt.close()

except Exception as e:
    st.error(f"Error loading dataset: {e}")
