import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle as pk

class NN1(nn.Module):
    """
    Simple 1 nn linear layer classifier from input features
    """
    def __init__(self, num_features):
        super(NN1, self).__init__()
        # singe Linear layer 
        self.classifier = nn.Linear(num_features, 1)
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        probs = self.sigmoid(self.classifier(x))  # Pass through the classifier
        return probs


class NN2(nn.Module):
    """
    Simple nn classifier with 2 linear hidden layers from input features
    """
    def __init__(self, num_features):
        super(NN2, self).__init__()
        self.l1 = nn.Linear(num_features, 128)
        self.l2 = nn.Linear(128, 32)
        self.classifier = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        probs = self.sigmoid(self.classifier(x))  # Pass through the classifier
        return probs


class CreativeEffectivenessModel:
    """
    Wrapper class for any kind of model. This helps testing and exploring models but is a valid
    approach for the training pipeline and model deployment (just that once a model is chosen it
    makes no sense to load all libraries so we could move to lazy imports)
    """
    def __init__(self, model_class=None, features=None, target=None, path=None):
        if path is not None:
            self.load(path)
        else: 
            if model_class == 'lgbm':
                params = {
                    'objective': 'binary',  # Binary classification
                    'metric': 'binary_logloss',#'auc',#
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'max_depth': 8,
                    'n_estimators': 500,
                    'verbose': -1,
                    'is_unbalance': True,
                    'random_state': 42
                }
                self.model = lgb.LGBMClassifier(**params)
            elif model_class == 'rf':
                params = {
                    "n_estimators": 500,
                    "criterion": 'logloss', #entroy, gini
                    "max_depth": 8,
                    "max_leaf_nodes": 31,
                    "random_state": 42
                }
                self.model = RandomForestClassifier(**params)
            elif model_class == 'nb':
                self.model = GaussianNB()
            elif model_class == 'nn1':
                self.model = NN1(len(features))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
            elif model_class == 'nn2':
                self.model = NN2(len(features))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
            else:
                self.model = None
            self.model_class = model_class
            self.features = features
            self.target = target
        
    def train(self, train_data: pd.DataFrame, verbose: bool = True):
        """
        Train model with train_data
        Args:
            train_data (pd.DataFrame): Input data used for training containing the
                        features required by the model.
            verbose (bool): Print training metrics (if set/available by the model)
                    and test metrics. Default is True
        """
        X_train = self.preprocess_fit(train_data[self.features].copy())
        y_train = train_data[self.target].values
        # Train the model
        if self.model_class in ['lgbm', 'nb']:
            self.model.fit(X_train, y_train)
        elif self.model_class in ['nn1', 'nn2']:
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            n_epochs = 40
            self.model.train()
            
            for epoch in range(n_epochs):
                running_loss = 0.0
                for batch_X, batch_y in dataloader:
                    
                    optimizer.zero_grad()
            
                    outputs = self.model(batch_X)
                    
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
            
                    running_loss += loss.item()
                if verbose:
                    print(f'Epoch [{epoch+1}/{n_epochs}],' +
                            f'Loss: {running_loss/len(dataloader):.4f}')
        
            
    def eval(self, test_data: pd.DataFrame, threshold: float = 0.5,
             verbose=True) -> dict:
        """
        Evaluates the trained mode to the test_data
        Args:
            test_data (pd.DataFrame): Input data containing the required features
            threshold (float, optional): The threshold for classifying a positive 
                                     class. Defaults to 0.5.
            verbose (bool): Print training metrics (if set/available by the model)
                    and test metrics. Default is True
        Returns:
            dict: The test error/performance classification metrics
        """
        test_data = test_data.copy()
        y_pred, y_pred_proba = self.predict(test_data, threshold=threshold)
        y_test = test_data[self.target]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)  # Use probabilities for AUC
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        if verbose:
            print(f"Accuracy: {accuracy:.2f}")
            print(f"ROC AUC: {roc_auc:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1 Score: {f1:.3f}")
        return metrics
        
    def train_eval(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                   verbose: bool = True) -> dict:
        """
        Train model with train_data and evaluate with test_data
        Args:
            train_data (pd.DataFrame): Input data used for training containing the
                        features required by the model.
            test_data (pd.DataFrame): Input data used for evaluation containing the
                    features required by the model.
            verbose (bool): Print training metrics (if set/available by the model)
                    and test metrics. Default is True
        Returns:
            dict: The test error/performance classification metrics
        """
        self.train(train_data, verbose=verbose)
        return self.eval(test_data, verbose=verbose)
        
    def predict(self, data: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions and prediction probabilities for the input data.
        Args:
            data (pd.DataFrame): Input data containing the features required by the model.
                             It must include all the features specified in `self.features`.
            threshold (float, optional): The threshold for classifying a positive 
                                     class. Defaults to 0.5.

        Returns:
            Tuple: 
            - np.ndarray: Binary predictions (0 or 1) based on the threshold.
            - np.ndarray: The predicted probabilities for the positive class.
        """
        X = self.preprocess(data[self.features].copy())
        if self.model_class in ['lgbm', 'nb']:
            y_pred_proba = self.model.predict_proba(X)
            y_pred_proba = y_pred_proba[:, 1]
        elif self.model_class in ['nn1', 'nn2']:
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_pred_proba = self.model(X_tensor).numpy()
        y_pred = (y_pred_proba >= threshold).astype(int)
        return y_pred, y_pred_proba

    def preprocess_fit(self, train_data: pd.DataFrame) -> np.array:
        """
        Fit the preprocessing components and transform the input training data
        (label encoding and scaling (if necessary)).

        Args:
            train_data (pd.DataFrame): The input data to preprocess
    
        Returns:
            np.ndarray: The preprocessed data.
        """
        label_encoded = train_data.select_dtypes(include='object').columns
        self.label_encoder = {}
        for feat in label_encoded:
            self.label_encoder[feat] = LabelEncoder()
            self.label_encoder[feat].fit(train_data[feat])
            train_data[feat] = self.label_encoder[feat].transform(train_data[feat])
      
        if self.model_class in ['nn1', 'nn2']:
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            return self.scaler.transform(train_data)
        else:
            return train_data.values
        
    def preprocess(self, data: pd.DataFrame) -> np.array:
        """
        Preprocess the input data by applying label encoding and scaling (if necessary).

        Args:
            data (pd.DataFrame): The input data to preprocess
    
        Returns:
            np.ndarray: The preprocessed data.
        """
        for feat in self.label_encoder.keys():
            # note that this will rise error if a category was not present when fitting
            data[feat] = self.label_encoder[feat].transform(data[feat])
            
        if self.model_class in ['nn1', 'nn2']:
            return self.scaler.transform(data)
        else:
            return data.values

    def save(self, path: str):
        """
        Save the model (serializes attributes in dict and pickles them).
        We assume this is a local path but with the proper connector this could be cloud storage.
        Args:
            path (str): The file path where the model's attributes will be saved.
        """
        with open(path, mode='wb') as pfile:
            pk.dump(self.__dict__, pfile)
            
    def load(self, path: str):
        """
        Load the model (from pickle, sets the attributes of the class).
        We assume this is a local path but with the proper connector this could be cloud storage.
        Args:
            path (str): The file path where the model's attributes is saved.
        """
        with open(path, mode='rb') as pfile:
            attributes = pk.load(pfile)
            self.__dict__.update(attributes)
            
        