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
        
    def train(self, train_data, verbose=True):
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
        
            
    def eval(self, test_data, threshold=0.5, verbose=True):
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
        
    def train_eval(self, train_data, test_data, verbose=True):
        self.train(train_data, verbose=verbose)
        return self.eval(test_data, verbose=verbose)
        
    def predict(self, data, threshold=0.5):
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

    def preprocess_fit(self, train_data):
        if ('country' in train_data.columns):
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(train_data['country'])
            train_data['country'] = self.label_encoder.transform(train_data['country'])
        if self.model_class in ['nn1', 'nn2']:
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            return self.scaler.transform(train_data)
        else:
            return train_data.values
        
    def preprocess(self, data):
        if ('country' in data.columns):
            data['country'] = self.label_encoder.transform(data['country'])
        if self.model_class in ['nn1', 'nn2']:
            return self.scaler.transform(data)
        else:
            return data.values

    def save(self, path):
        with open(path, mode='wb') as pfile:
            pk.dump(self.__dict__, pfile)
            
    def load(self, path):
        with open(path, mode='rb') as pfile:
            attributes = pk.load(pfile)
            self.__dict__.update(attributes)
            
        