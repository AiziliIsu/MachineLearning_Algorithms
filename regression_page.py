import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.data_loader import load_dataset
from algorithms.regression import RegressionAlgorithms
from utils.visualization import plot_regression_results, plot_metrics_comparison, plot_feature_importance


def display_regression_metrics(metrics):
    """Helper function to display regression metrics"""
    st.write("### Model Performance")
    metrics_df = pd.DataFrame({
        'Algorithm': list(metrics.keys()),
        'MSE': [metrics[algo]['MSE'] for algo in metrics],  # <-- added MSE
        'RMSE': [metrics[algo]['RMSE'] for algo in metrics],
        'MAE': [metrics[algo]['MAE'] for algo in metrics],
        'Training Time (s)': [metrics[algo]['Training Time'] for algo in metrics]
    })
    st.dataframe(metrics_df)

    # Visualize metrics
    st.write("#### MSE Comparison")
    fig = plot_metrics_comparison(metrics, 'MSE')  # <-- added MSE plot
    st.pyplot(fig)

    st.write("#### RMSE Comparison")
    fig = plot_metrics_comparison(metrics, 'RMSE')
    st.pyplot(fig)

    st.write("#### MAE Comparison")
    fig = plot_metrics_comparison(metrics, 'MAE')
    st.pyplot(fig)


def regression_page():
    # Load regression dataset
    df, dataset_name = load_dataset("regression")
    
    st.subheader(f"Dataset: {dataset_name}")
    
    st.write("### Sample Data")
    st.dataframe(df.head())
    
    st.write("### Dataset Statistics")
    st.dataframe(df.describe())

    # Feature distribution
    st.write("#### Feature Distributions")
    feature_col = st.selectbox("Select a feature to visualize", list(df.columns[:-1]))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[feature_col], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Target distribution
    st.write("#### Target Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['target'], kde=True, ax=ax)
    ax.set_title('Distribution of Target Variable')
    st.pyplot(fig)
    
    # Model training
    st.write("### Model Training")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    train_button_clicked = st.button("Train Regression Models", key="train_reg_button")
    
    if train_button_clicked:
        with st.spinner("Training models..."):
            reg_algorithms = RegressionAlgorithms(X_train_scaled, X_test_scaled, y_train, y_test)
            metrics = reg_algorithms.run_all_algorithms()
            
            # Store in session state
            st.session_state['reg_algorithms'] = reg_algorithms
            st.session_state['reg_metrics'] = metrics
            
            # Display metrics
            display_regression_metrics(metrics)
    
    # Display results if models have been trained (and train button not just clicked)
    elif 'reg_algorithms' in st.session_state and 'reg_metrics' in st.session_state:
        reg_algorithms = st.session_state['reg_algorithms']
        metrics = st.session_state['reg_metrics']
        
        # Display metrics
        display_regression_metrics(metrics)
        
        # Visualize predictions
        st.write("### Predictions Visualization")
        algorithm = st.selectbox("Select algorithm", list(reg_algorithms.predictions.keys()))
        
        predictions = reg_algorithms.predictions[algorithm]
        fig = plot_regression_results(y_test, predictions, f"{algorithm} Predictions vs Actual Values")
        st.pyplot(fig)
        
        # Feature importance for applicable models
        if algorithm in ["Multiple Linear Regression", "Simple Linear Regression"]:
            st.write("#### Feature Coefficients")
            fig = plot_feature_importance(reg_algorithms.models[algorithm], list(X.columns))
            if fig:
                st.pyplot(fig)


if __name__ == "__main__":
    regression_page()