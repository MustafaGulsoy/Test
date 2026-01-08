"""
Alzheimer's Disease Prediction Model using XGBoost
Dataset: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis."""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    print("\n--- Dataset Info ---")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")

    print("\n--- Target Distribution ---")
    if 'Diagnosis' in df.columns:
        print(df['Diagnosis'].value_counts())
        print(f"\nClass balance: {df['Diagnosis'].value_counts(normalize=True).round(3).to_dict()}")

    print("\n--- Numerical Features Statistics ---")
    print(df.describe().round(2))


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess the dataset for model training."""
    df_processed = df.copy()

    # Drop PatientID if exists (not useful for prediction)
    if 'PatientID' in df_processed.columns:
        df_processed = df_processed.drop('PatientID', axis=1)

    # Drop DoctorInCharge if exists (not useful for prediction)
    if 'DoctorInCharge' in df_processed.columns:
        df_processed = df_processed.drop('DoctorInCharge', axis=1)

    # Separate features and target
    target_col = 'Diagnosis'
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    print(f"\nPreprocessed features shape: {X.shape}")
    print(f"Feature names: {list(X.columns)}")

    return X, y, scaler, label_encoders


def train_xgboost_model(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """Train XGBoost model with optimized hyperparameters."""
    print("\n" + "="*50)
    print("TRAINING XGBOOST MODEL")
    print("="*50)

    # XGBoost parameters optimized for medical data
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
        n_jobs=-1
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"Best iteration: {model.best_iteration}")

    return model


def evaluate_model(model, X_test, y_test, feature_names: list) -> dict:
    """Evaluate model performance and return metrics."""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    print("\n--- Performance Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Alzheimer', 'Alzheimer']))

    return metrics


def plot_results(model, X_test, y_test, feature_names: list, save_path: str = "results") -> None:
    """Generate and save visualization plots."""
    os.makedirs(save_path, exist_ok=True)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['No Alzheimer', 'Alzheimer'],
                yticklabels=['No Alzheimer', 'Alzheimer'])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14)
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve', fontsize=14)
    axes[0, 1].legend(loc='lower right')

    # 3. Feature Importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]  # Top 15 features
    axes[1, 0].barh(range(len(indices)), importance[indices], color='steelblue')
    axes[1, 0].set_yticks(range(len(indices)))
    axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
    axes[1, 0].set_title('Top 15 Feature Importance', fontsize=14)
    axes[1, 0].set_xlabel('Importance Score')

    # 4. Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Alzheimer', color='green')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Alzheimer', color='red')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/model_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {save_path}/model_results.png")


def cross_validate_model(model, X, y) -> None:
    """Perform cross-validation."""
    print("\n" + "="*50)
    print("CROSS-VALIDATION (5-Fold)")
    print("="*50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")

    cv_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std() * 2:.4f})")


def save_model(model, scaler, label_encoders, save_path: str = "model") -> None:
    """Save the trained model and preprocessors."""
    os.makedirs(save_path, exist_ok=True)

    joblib.dump(model, f"{save_path}/xgboost_alzheimer_model.pkl")
    joblib.dump(scaler, f"{save_path}/scaler.pkl")
    joblib.dump(label_encoders, f"{save_path}/label_encoders.pkl")

    print(f"\nModel saved to {save_path}/")


def main():
    """Main function to run the complete pipeline."""
    print("="*50)
    print("ALZHEIMER'S DISEASE PREDICTION MODEL")
    print("Using XGBoost Classifier")
    print("="*50)

    # Check if dataset exists
    data_path = "data/alzheimers_disease_data.csv"

    if not os.path.exists(data_path):
        print(f"\nDataset not found at '{data_path}'")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset")
        print(f"\nThen place it in: {data_path}")
        print("\nOr use Kaggle API:")
        print("  kaggle datasets download -d rabieelkharoua/alzheimers-disease-dataset")
        print("  unzip alzheimers-disease-dataset.zip -d data/")
        return

    # Load data
    df = load_data(data_path)

    # Explore data
    explore_data(df)

    # Preprocess data
    X, y, scaler, label_encoders = preprocess_data(df)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train model
    model = train_xgboost_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    feature_names = list(X.columns)
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    # Cross-validation
    cross_validate_model(model, X, y)

    # Plot results
    plot_results(model, X_test, y_test, feature_names)

    # Save model
    save_model(model, scaler, label_encoders)

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
