# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# import xgboost as xgb
# from utils.data_loader import load_dataset
#
#
# def apply_cross_validation(X, y, model, cv=5, scoring='neg_mean_squared_error'):
#     cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
#     return cv_scores
#
#
# def cross_validation_page():
#     # Choose problem type
#     problem_type = st.radio("Select problem type", ["Regression", "Classification"])
#
#     # Load appropriate dataset
#     dataset_type = "regression" if problem_type == "Regression" else "classification"
#     df, dataset_name = load_dataset(dataset_type)
#
#     st.subheader(f"Dataset: {dataset_name}")
#
#     # Prepare data
#     X = df.drop('target', axis=1)
#     y = df['target']
#
#     # Choose algorithm
#     if problem_type == "Regression":
#         algorithm = st.selectbox("Select regression algorithm",
#                                  ["Simple Linear Regression", "Multiple Linear Regression",
#                                   "Decision Tree", "Random Forest", "AdaBoost", "XGBoost"])
#
#         if algorithm == "Simple Linear Regression":
#             model = LinearRegression()
#             scoring = 'neg_mean_squared_error'
#
#         elif algorithm == "Multiple Linear Regression":
#             model = LinearRegression()
#             scoring = 'neg_mean_squared_error'
#
#         elif algorithm == "Decision Tree":
#             max_depth = st.slider("Max depth", 1, 20, 5)
#             model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
#             scoring = 'neg_mean_squared_error'
#
#         elif algorithm == "Random Forest":
#             n_estimators = st.slider("Number of trees", 10, 200, 50)
#             model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
#             scoring = 'neg_mean_squared_error'
#
#         elif algorithm == "AdaBoost":
#             n_estimators = st.slider("Number of estimators", 10, 200, 50)
#             model = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)
#             scoring = 'neg_mean_squared_error'
#
#         else:  # XGBoost
#             n_estimators = st.slider("Number of estimators", 10, 200, 50)
#             model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
#             scoring = 'neg_mean_squared_error'
#
#
#     else:  # Classification
#         algorithm = st.selectbox("Select classification algorithm",
#                                  ["Naive Bayes", "K-Nearest Neighbors", "Random Forest",
#                                   "AdaBoost", "XGBoost", "Decision Tree"])
#
#         if algorithm == "Naive Bayes":
#             model = GaussianNB()
#
#         elif algorithm == "K-Nearest Neighbors":
#             n_neighbors = st.slider("Number of neighbors", 1, 20, 5)
#             model = KNeighborsClassifier(n_neighbors=n_neighbors)
#
#         elif algorithm == "Random Forest":
#             n_estimators = st.slider("Number of trees", 10, 200, 50)
#             model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#
#         elif algorithm == "AdaBoost":
#             n_estimators = st.slider("Number of estimators", 10, 200, 50)
#             model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
#
#         elif algorithm == "XGBoost":
#             n_estimators = st.slider("Number of estimators", 10, 200, 50)
#             model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42)
#
#         else:  # Decision Tree
#             max_depth = st.slider("Max depth", 1, 20, 5)
#             model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
#
#         scoring = 'accuracy'
#
#     # Set up cross-validation
#     cv_folds = st.slider("Number of CV folds", 2, 10, 5)
#
#     # Perform cross-validation
#     if st.button("Perform Cross Validation"):
#         with st.spinner("Running cross-validation..."):
#             # Scale features
#             scaler = StandardScaler()
#             X_scaled = scaler.fit_transform(X)
#
#             # Perform cross-validation
#             cv_scores = apply_cross_validation(X_scaled, y, model, cv=cv_folds, scoring=scoring)
#
#             # Store in session state
#             st.session_state['cv_scores'] = cv_scores
#             st.session_state['problem_type'] = problem_type
#             st.session_state['cv_folds'] = cv_folds
#
#             # Display results
#             st.subheader("Cross Validation Results")
#
#             if problem_type == "Regression":
#                 # Convert negative MSE to RMSE for easier interpretation
#                 rmse_scores = np.sqrt(-cv_scores)
#                 st.write(f"RMSE scores for each fold: {rmse_scores}")
#                 st.write(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
#                 st.write(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")
#
#                 # Plot RMSE for each fold
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 ax.bar(range(1, cv_folds + 1), rmse_scores)
#                 ax.axhline(np.mean(rmse_scores), color='red', linestyle='--',
#                            label=f'Mean RMSE: {np.mean(rmse_scores):.4f}')
#                 ax.set_xlabel('Fold')
#                 ax.set_ylabel('RMSE')
#                 ax.set_title('RMSE across CV folds')
#                 ax.set_xticks(range(1, cv_folds + 1))
#                 ax.legend()
#                 st.pyplot(fig)
#
#             else:  # Classification
#                 st.write(f"Accuracy scores for each fold: {cv_scores}")
#                 st.write(f"Mean accuracy: {np.mean(cv_scores):.4f}")
#                 st.write(f"Standard deviation of accuracy: {np.std(cv_scores):.4f}")
#
#                 # Plot accuracy for each fold
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 ax.bar(range(1, cv_folds + 1), cv_scores)
#                 ax.axhline(np.mean(cv_scores), color='red', linestyle='--',
#                            label=f'Mean Accuracy: {np.mean(cv_scores):.4f}')
#                 ax.set_xlabel('Fold')
#                 ax.set_ylabel('Accuracy')
#                 ax.set_title('Accuracy across CV folds')
#                 ax.set_xticks(range(1, cv_folds + 1))
#                 ax.legend()
#                 st.pyplot(fig)
#
#     # Display results if CV has been performed (using session state)
#     elif 'cv_scores' in st.session_state and 'problem_type' in st.session_state:
#         cv_scores = st.session_state['cv_scores']
#         stored_problem_type = st.session_state['problem_type']
#         cv_folds = st.session_state['cv_folds']
#
#         # Only display if the stored problem type matches the current selection
#         if stored_problem_type == problem_type:
#             st.subheader("Cross Validation Results")
#
#             if problem_type == "Regression":
#                 # Convert negative MSE to RMSE for easier interpretation
#                 rmse_scores = np.sqrt(-cv_scores)
#                 st.write(f"RMSE scores for each fold: {rmse_scores}")
#                 st.write(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
#                 st.write(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")
#
#                 # Plot RMSE for each fold
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 ax.bar(range(1, cv_folds + 1), rmse_scores)
#                 ax.axhline(np.mean(rmse_scores), color='red', linestyle='--',
#                            label=f'Mean RMSE: {np.mean(rmse_scores):.4f}')
#                 ax.set_xlabel('Fold')
#                 ax.set_ylabel('RMSE')
#                 ax.set_title('RMSE across CV folds')
#                 ax.set_xticks(range(1, cv_folds + 1))
#                 ax.legend()
#                 st.pyplot(fig)
#
#             else:  # Classification
#                 st.write(f"Accuracy scores for each fold: {cv_scores}")
#                 st.write(f"Mean accuracy: {np.mean(cv_scores):.4f}")
#                 st.write(f"Standard deviation of accuracy: {np.std(cv_scores):.4f}")
#
#                 # Plot accuracy for each fold
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 ax.bar(range(1, cv_folds + 1), cv_scores)
#                 ax.axhline(np.mean(cv_scores), color='red', linestyle='--',
#                            label=f'Mean Accuracy: {np.mean(cv_scores):.4f}')
#                 ax.set_xlabel('Fold')
#                 ax.set_ylabel('Accuracy')
#                 ax.set_title('Accuracy across CV folds')
#                 ax.set_xticks(range(1, cv_folds + 1))
#                 ax.legend()
#                 st.pyplot(fig)
#
#
# if __name__ == "__main__":
#     cross_validation_page()


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
from utils.data_loader import load_dataset

def apply_cross_validation(X, y, model, cv, scoring='neg_mean_squared_error'):
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return cv_scores

def cross_validation_page():
    # Choose problem type
    problem_type = st.radio("Select problem type", ["Regression", "Classification"])

    # Load appropriate dataset
    dataset_type = "regression" if problem_type == "Regression" else "classification"
    df, dataset_name = load_dataset(dataset_type)

    st.subheader(f"Dataset: {dataset_name}")

    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']

    # Choose algorithm
    if problem_type == "Regression":
        algorithm = st.selectbox("Select regression algorithm",
                                 ["Simple Linear Regression", "Multiple Linear Regression",
                                  "Decision Tree", "Random Forest", "AdaBoost", "XGBoost"])

        if algorithm == "Simple Linear Regression":
            model = LinearRegression()
            scoring = 'neg_mean_squared_error'

        elif algorithm == "Multiple Linear Regression":
            model = LinearRegression()
            scoring = 'neg_mean_squared_error'

        elif algorithm == "Decision Tree":
            max_depth = st.slider("Max depth", 1, 20, 5)
            model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            scoring = 'neg_mean_squared_error'

        elif algorithm == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 200, 50)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            scoring = 'neg_mean_squared_error'

        elif algorithm == "AdaBoost":
            n_estimators = st.slider("Number of estimators", 10, 200, 50)
            model = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)
            scoring = 'neg_mean_squared_error'

        else:  # XGBoost
            n_estimators = st.slider("Number of estimators", 10, 200, 50)
            model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
            scoring = 'neg_mean_squared_error'

    else:  # Classification
        algorithm = st.selectbox("Select classification algorithm",
                                 ["Naive Bayes", "K-Nearest Neighbors", "Random Forest",
                                  "AdaBoost", "XGBoost", "Decision Tree"])

        if algorithm == "Naive Bayes":
            model = GaussianNB()

        elif algorithm == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of neighbors", 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        elif algorithm == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 200, 50)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        elif algorithm == "AdaBoost":
            n_estimators = st.slider("Number of estimators", 10, 200, 50)
            model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)

        elif algorithm == "XGBoost":
            n_estimators = st.slider("Number of estimators", 10, 200, 50)
            model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42)

        else:  # Decision Tree
            max_depth = st.slider("Max depth", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        scoring = 'accuracy'

    # Choose cross-validation type
    cv_type = st.radio("Select cross-validation type", ["K-Fold", "Leave-One-Out (LOO)"])

    if cv_type == "K-Fold":
        cv_folds = st.slider("Number of CV folds", 2, 10, 5)
        cv_strategy = cv_folds
    else:
        st.info("Leave-One-Out Cross-Validation will be used. (Each sample will be used once as a test set)")
        cv_strategy = LeaveOneOut()

    # Perform cross-validation
    if st.button("Perform Cross Validation"):
        with st.spinner("Running cross-validation..."):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform cross-validation
            cv_scores = apply_cross_validation(X_scaled, y, model, cv=cv_strategy, scoring=scoring)

            # Store in session state
            st.session_state['cv_scores'] = cv_scores
            st.session_state['problem_type'] = problem_type
            st.session_state['cv_type'] = cv_type
            st.session_state['cv_folds'] = cv_folds if cv_type == "K-Fold" else None

            # Display results
            st.subheader("Cross Validation Results")

            if problem_type == "Regression":
                rmse_scores = np.sqrt(-cv_scores)
                st.write(f"RMSE scores: {rmse_scores}")
                st.write(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
                st.write(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(1, len(rmse_scores) + 1), rmse_scores)
                ax.axhline(np.mean(rmse_scores), color='red', linestyle='--',
                           label=f'Mean RMSE: {np.mean(rmse_scores):.4f}')
                ax.set_xlabel('Fold')
                ax.set_ylabel('RMSE')
                ax.set_title('RMSE across CV folds')
                ax.set_xticks(range(1, len(rmse_scores) + 1))
                ax.legend()
                st.pyplot(fig)

            else:
                st.write(f"Accuracy scores: {cv_scores}")
                st.write(f"Mean accuracy: {np.mean(cv_scores):.4f}")
                st.write(f"Standard deviation of accuracy: {np.std(cv_scores):.4f}")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(1, len(cv_scores) + 1), cv_scores)
                ax.axhline(np.mean(cv_scores), color='red', linestyle='--',
                           label=f'Mean Accuracy: {np.mean(cv_scores):.4f}')
                ax.set_xlabel('Fold')
                ax.set_ylabel('Accuracy')
                ax.set_title('Accuracy across CV folds')
                ax.set_xticks(range(1, len(cv_scores) + 1))
                ax.legend()
                st.pyplot(fig)

    elif 'cv_scores' in st.session_state and 'problem_type' in st.session_state:
        cv_scores = st.session_state['cv_scores']
        stored_problem_type = st.session_state['problem_type']
        stored_cv_type = st.session_state['cv_type']

        if stored_problem_type == problem_type and stored_cv_type == cv_type:
            st.subheader("Cross Validation Results")

            if problem_type == "Regression":
                rmse_scores = np.sqrt(-cv_scores)
                st.write(f"RMSE scores: {rmse_scores}")
                st.write(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
                st.write(f"Standard deviation of RMSE: {np.std(rmse_scores):.4f}")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(1, len(rmse_scores) + 1), rmse_scores)
                ax.axhline(np.mean(rmse_scores), color='red', linestyle='--',
                           label=f'Mean RMSE: {np.mean(rmse_scores):.4f}')
                ax.set_xlabel('Fold')
                ax.set_ylabel('RMSE')
                ax.set_title('RMSE across CV folds')
                ax.set_xticks(range(1, len(rmse_scores) + 1))
                ax.legend()
                st.pyplot(fig)

            else:
                st.write(f"Accuracy scores: {cv_scores}")
                st.write(f"Mean accuracy: {np.mean(cv_scores):.4f}")
                st.write(f"Standard deviation of accuracy: {np.std(cv_scores):.4f}")

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(1, len(cv_scores) + 1), cv_scores)
                ax.axhline(np.mean(cv_scores), color='red', linestyle='--',
                           label=f'Mean Accuracy: {np.mean(cv_scores):.4f}')
                ax.set_xlabel('Fold')
                ax.set_ylabel('Accuracy')
                ax.set_title('Accuracy across CV folds')
                ax.set_xticks(range(1, len(cv_scores) + 1))
                ax.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    cross_validation_page()



