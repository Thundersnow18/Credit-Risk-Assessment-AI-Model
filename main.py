import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Function to generate sample data
def generate_sample_data(size=1000):
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, size=size),
        'income': np.random.randint(20000, 100000, size=size),
        'credit_history': np.random.choice([1, 0], size=size),  # 1 = Good, 0 = Bad
        'social_media_activity': np.random.choice([1, 0], size=size),  # 1 = Active, 0 = Inactive
        'transaction_frequency': np.random.randint(1, 50, size=size),  # Transactions per month
        'loan_amount': np.random.randint(5000, 50000, size=size),
        'label': np.random.choice([0, 1], size=size)  # 0 = Low Risk, 1 = High Risk
    })
    return data

# Function to preprocess data
def preprocess_data(data):
    X = data.drop(columns=['label'])
    y = data['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Function to train XGBoost model
def train_model(X_train, y_train):
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# Function to evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

# Function to explain model with SHAP
def explain_model(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

# Function to implement unique feature - custom weights for alternate data
def adjust_feature_weights(X, custom_weights):
    adjusted_X = X.copy()
    for feature, weight in custom_weights.items():
        feature_idx = X.columns.get_loc(feature)
        adjusted_X.iloc[:, feature_idx] *= weight
    return adjusted_X

# Main function to run the data pipeline
def main():
    # Generate sample data
    data = generate_sample_data()

    # Preprocess data
    X_scaled, y, scaler = preprocess_data(data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Explain model
    explain_model(model, X_test)

    # Adjust feature weights
    custom_weights = {'social_media_activity': 1.5, 'income': 0.8}
    X_train_adjusted = adjust_feature_weights(pd.DataFrame(X_train, columns=data.drop(columns=['label']).columns), custom_weights)
    X_test_adjusted = adjust_feature_weights(pd.DataFrame(X_test, columns=data.drop(columns=['label']).columns), custom_weights)

    # Retrain model with adjusted features
    model_adjusted = train_model(X_train_adjusted, y_train)

    # Evaluate model with adjusted weights
    accuracy_adjusted, y_pred_adjusted = evaluate_model(model_adjusted, X_test_adjusted, y_test)
    print(f"Accuracy with adjusted weights: {accuracy_adjusted:.4f}")

if __name__ == "__main__":
    main()

