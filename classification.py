import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

class ClassificationAlgorithms:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def naive_bayes(self):
        start_time = time.time()
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='macro')
        recall = recall_score(self.y_test, predictions, average='macro')
        f1 = f1_score(self.y_test, predictions, average='macro')

        # NEW: confusion matrix
        cm = confusion_matrix(self.y_test, predictions)

        # NEW: specificity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = None
        
        self.models['Naive Bayes'] = model
        self.predictions['Naive Bayes'] = predictions
        self.metrics['Naive Bayes'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time,
            'Confusion Matrix': cm.tolist(),
            'Specificity': specificity
        }

        return accuracy, precision, recall, f1, cm, specificity

    def k_nearest_neighbors(self, n_neighbors=5):
        start_time = time.time()
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='macro')
        recall = recall_score(self.y_test, predictions, average='macro')
        f1 = f1_score(self.y_test, predictions, average='macro')

        # NEW: confusion matrix
        cm = confusion_matrix(self.y_test, predictions)

        # NEW: specificity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = None
        
        self.models['K-Nearest Neighbors'] = model
        self.predictions['K-Nearest Neighbors'] = predictions
        self.metrics['K-Nearest Neighbors'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time,
            'Confusion Matrix': cm.tolist(),
            'Specificity': specificity
        }

        return accuracy, precision, recall, f1, cm, specificity

    def random_forest(self, n_estimators=50):
        start_time = time.time()
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='macro')
        recall = recall_score(self.y_test, predictions, average='macro')
        f1 = f1_score(self.y_test, predictions, average='macro')

        # NEW: confusion matrix
        cm = confusion_matrix(self.y_test, predictions)

        # NEW: specificity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = None
        
        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = predictions
        self.metrics['Random Forest'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time,
            'Confusion Matrix': cm.tolist(),
            'Specificity': specificity
        }

        return accuracy, precision, recall, f1, cm, specificity

    def adaboost(self, n_estimators=50):
        start_time = time.time()
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='macro')
        recall = recall_score(self.y_test, predictions, average='macro')
        f1 = f1_score(self.y_test, predictions, average='macro')

        # NEW: confusion matrix
        cm = confusion_matrix(self.y_test, predictions)

        # NEW: specificity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = None
        
        self.models['AdaBoost'] = model
        self.predictions['AdaBoost'] = predictions
        self.metrics['AdaBoost'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time,
            'Confusion Matrix': cm.tolist(),
            'Specificity': specificity
        }

        return accuracy, precision, recall, f1, cm, specificity

    def xgboost(self, n_estimators=50):
        start_time = time.time()
        model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='macro')
        recall = recall_score(self.y_test, predictions, average='macro')
        f1 = f1_score(self.y_test, predictions, average='macro')

        # NEW: confusion matrix
        cm = confusion_matrix(self.y_test, predictions)

        # NEW: specificity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = None
        
        self.models['XGBoost'] = model
        self.predictions['XGBoost'] = predictions
        self.metrics['XGBoost'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time,
            'Confusion Matrix': cm.tolist(),
            'Specificity': specificity
        }

        return accuracy, precision, recall, f1, cm, specificity

    from sklearn.metrics import confusion_matrix

    def decision_tree(self, max_depth=None):
        """Decision Tree Classifier using Entropy (Information Gain)"""
        start_time = time.time()
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time

        # Metrics calculation
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='macro', zero_division=0)
        recall = recall_score(self.y_test, predictions, average='macro', zero_division=0)
        f1 = f1_score(self.y_test, predictions, average='macro', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions)

        # Specificity calculation
        if cm.shape == (2, 2):  # binary classification
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = None  # Not applicable for multiclass classification

        self.models['Decision Tree'] = model
        self.predictions['Decision Tree'] = predictions
        self.metrics['Decision Tree'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time,
            'Confusion Matrix': cm.tolist(),
            'Specificity': specificity
        }

        return accuracy, precision, recall, f1, cm, specificity


    def logistic_regression(self):
        """
        Logistic Regression for classification tasks
        """
        start_time = time.time()
        model = LogisticRegression(max_iter=1000)  # increased max_iter to ensure convergence
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        training_time = time.time() - start_time

        # Standard classification metrics
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, predictions, average='weighted', zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, predictions)

        # Specificity calculation
        if cm.shape == (2, 2):  # binary classification
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            specificity = None  # Not applicable for multiclass classification

        self.models['Logistic Regression'] = model
        self.predictions['Logistic Regression'] = predictions
        self.metrics['Logistic Regression'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Training Time': training_time,
            'Confusion Matrix': cm.tolist(),  # converted to list to make it JSON serializable if needed
            'Specificity': specificity
        }

        return accuracy, precision, recall, f1, cm, specificity

    def run_all_algorithms(self):
        self.naive_bayes()
        self.k_nearest_neighbors()
        self.decision_tree()
        self.random_forest()
        self.adaboost()
        self.xgboost()
        self.logistic_regression()
        
        return self.metrics