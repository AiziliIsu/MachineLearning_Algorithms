import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionAlgorithms:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def decision_tree_regressor(self, max_depth=None):
        """Decision Tree Regressor using MSE as a criterion"""
        start_time = time.time()
        model = DecisionTreeRegressor(criterion='squared_error', max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time

        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)

        self.models['Decision Tree'] = model
        self.predictions['Decision Tree'] = predictions
        self.metrics['Decision Tree'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Training Time': training_time
        }

        return mse, rmse, mae

    def random_forest_regressor(self, n_estimators=50):
        """Random Forest Regressor"""
        start_time = time.time()
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time

        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)

        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = predictions
        self.metrics['Random Forest'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Training Time': training_time
        }

        return mse, rmse, mae

    def adaboost_regressor(self, n_estimators=50):
        """AdaBoost Regressor"""
        start_time = time.time()
        model = AdaBoostRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time

        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)

        self.models['AdaBoost'] = model
        self.predictions['AdaBoost'] = predictions
        self.metrics['AdaBoost'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Training Time': training_time
        }

        return mse, rmse, mae

    def xgboost_regressor(self, n_estimators=50):
        """XGBoost Regressor"""
        start_time = time.time()
        model = XGBRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time

        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)

        self.models['XGBoost'] = model
        self.predictions['XGBoost'] = predictions
        self.metrics['XGBoost'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Training Time': training_time
        }

        return mse, rmse, mae

    def simple_linear_regression(self):
        """
        Simple Linear Regression using a single feature
        """
        start_time = time.time()
        # Use only the first feature for simple linear regression
        X_train_single = self.X_train[:, 0].reshape(-1, 1)
        X_test_single = self.X_test[:, 0].reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_train_single, self.y_train)
        predictions = model.predict(X_test_single)
        training_time = time.time() - start_time

        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)

        self.models['Simple Linear Regression'] = model
        self.predictions['Simple Linear Regression'] = predictions
        self.metrics['Simple Linear Regression'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Training Time': training_time
        }

        return mse, rmse, mae

    def multiple_linear_regression(self):
        """
        Multiple Linear Regression using all features
        """
        start_time = time.time()
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time

        mse = mean_squared_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)

        self.models['Multiple Linear Regression'] = model
        self.predictions['Multiple Linear Regression'] = predictions
        self.metrics['Multiple Linear Regression'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Training Time': training_time
        }

        return mse, rmse, mae


    def run_all_algorithms(self):
        self.simple_linear_regression()
        self.multiple_linear_regression()
        self.adaboost_regressor()
        self.decision_tree_regressor()
        self.random_forest_regressor()
        self.xgboost_regressor()

        return self.metrics