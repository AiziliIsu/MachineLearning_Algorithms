import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.visualization import plot_metrics_comparison, plot_feature_importance
from algorithms.classification import ClassificationAlgorithms
from utils.data_loader import load_dataset


def display_classification_metrics(metrics):
    """Helper function to display classification metrics"""
    st.write("### Model Performance")

    # Create the main metrics table
    metrics_df = pd.DataFrame({
        'Algorithm': list(metrics.keys()),
        'Accuracy': [metrics[algo]['Accuracy'] for algo in metrics],
        'Precision': [metrics[algo]['Precision'] for algo in metrics],
        'Recall': [metrics[algo]['Recall'] for algo in metrics],
        'F1-Score': [metrics[algo].get('F1-Score', 0.0) for algo in metrics],
        'Specificity': [metrics[algo].get('Specificity', 0.0) for algo in metrics],
        'Training Time (s)': [metrics[algo]['Training Time'] for algo in metrics]
    })

    st.dataframe(metrics_df)

    # Visualize metrics
    st.write("#### Accuracy Comparison")
    fig = plot_metrics_comparison(metrics, 'Accuracy')
    st.pyplot(fig)

    st.write("#### F1-Score Comparison")
    fig = plot_metrics_comparison(metrics, 'F1-Score')
    st.pyplot(fig)

    # Show Confusion Matrices
    st.write("### Confusion Matrices")
    for algo in metrics:
        st.write(f"#### {algo}")
        cm = metrics[algo]['Confusion Matrix']

        # Option 1: simple table
        st.table(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

        # Option 2 (nicer): heatmap
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - {algo}')
        st.pyplot(fig)


def classification_page():
    # Load classification dataset
    df, dataset_name = load_dataset("classification")
    
    st.subheader(f"Dataset: {dataset_name}")
    
    st.write("### Sample Data")
    st.dataframe(df.head())
    
    st.write("### Dataset Statistics")
    st.dataframe(df.describe())

    # Correlation heatmap
    st.write("#### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


    # Feature pairs plot
    st.write("#### Feature Pairs Plot")
    # Ensure 'target' is in df
    if 'target' not in df.columns:
        st.error("'target' column is missing in the dataset.")
    else:
        feature_columns = [col for col in df.columns if col != 'target']
        default_features = feature_columns[:2] if len(feature_columns) >= 2 else feature_columns

        selected_features = st.multiselect(
            "Select features for pairplot (max 3)",
            options=feature_columns,
            default=default_features
        )

        st.write("Selected Features:", selected_features)

        if len(selected_features) > 0:
            if len(selected_features) <= 3:
                # Ensure pairplot_data is a DataFrame
                pairplot_data = df[selected_features + ['target']].copy()

                # Convert categorical columns to numeric if needed
                pairplot_data = pd.get_dummies(pairplot_data, drop_first=True)

                st.write("Pairplot Data Shape:", pairplot_data.shape)
                st.write("Pairplot Data Types:", pairplot_data.dtypes)

                # Create the pairplot
                try:
                    fig = sns.pairplot(pairplot_data, hue='target')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating pairplot: {e}")
            else:
                st.warning("Please select at most 3 features for the pairplot.")





    # Model training
    st.write("### Model Training")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Get class names
    class_names = [str(i) for i in sorted(y.unique())]
    
    # Split data
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, key="classification_test_size")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    train_button_clicked = st.button("Train Classification Models", key="train_cls_button")
    
    if train_button_clicked:
        with st.spinner("Training models..."):
            cls_algorithms = ClassificationAlgorithms(X_train_scaled, X_test_scaled, y_train, y_test)
            metrics = cls_algorithms.run_all_algorithms()
            
            # Store in session state
            st.session_state['cls_algorithms'] = cls_algorithms
            st.session_state['cls_metrics'] = metrics
            
            # Display metrics
            display_classification_metrics(metrics)
    
    # Display results if models have been trained (and train button not just clicked)
    elif 'cls_algorithms' in st.session_state and 'cls_metrics' in st.session_state:
        cls_algorithms = st.session_state['cls_algorithms']
        metrics = st.session_state['cls_metrics']
        
        # Display metrics
        display_classification_metrics(metrics)
        
        # Visualize confusion matrix
        st.write("### Confusion Matrix")
        algorithm = st.selectbox("Select algorithm", list(cls_algorithms.predictions.keys()))

        
        # Feature importance for applicable models
        if algorithm in ["Random Forest", "AdaBoost", "XGBoost"]:
            st.write("#### Feature Importance")
            fig = plot_feature_importance(cls_algorithms.models[algorithm], list(X.columns))
            if fig:
                st.pyplot(fig)

if __name__ == "__main__":
    classification_page()