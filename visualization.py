import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_regression_results(y_test, predictions, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, predictions)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    return fig

# Update the plot_feature_importance function to handle linear regression coefficients

def plot_feature_importance(model, feature_names):
    """Plot feature importance for different model types"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For models with feature_importances_ attribute (like tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        
    # For linear models with coefficients
    elif hasattr(model, 'coef_'):
        # For simple linear regression, coef_ might be a single value or a 1D array
        if len(model.coef_.shape) == 1:
            coeffs = model.coef_
        else:  # For more complex models
            coeffs = model.coef_[0]
            
        # Get absolute values and sorted indices
        abs_coeffs = np.abs(coeffs)
        indices = np.argsort(abs_coeffs)[::-1]
        
        plt.title('Feature Coefficients (Absolute Values)')
        plt.bar(range(len(coeffs)), abs_coeffs[indices], align='center')
        plt.xticks(range(len(coeffs)), [feature_names[i] for i in indices], rotation=90)
    
    else:
        return None
        
    plt.tight_layout()
    return fig



def plot_metrics_comparison(metrics_dict, metric_name):
    algorithms = list(metrics_dict.keys())
    values = [metrics_dict[algo][metric_name] for algo in algorithms]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(algorithms, values)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Comparison of {metric_name} across algorithms')
    plt.tight_layout()
    return fig



