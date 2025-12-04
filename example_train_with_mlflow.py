"""
Example: Training a fraud detection model with MLflow tracking
This script demonstrates how to integrate MLflow into your existing training pipeline
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from datetime import datetime

# Add mlflow directory to path
sys.path.append('./mlflow')

from mlflow_utils import FraudMLflowTracker
from mlflow_config import setup_mlflow


def generate_sample_data(n_samples=10000):
    """
    Generate synthetic fraud detection data
    Replace this with your actual data loading logic
    """
    np.random.seed(42)
    
    # Generate features
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (10% fraud rate)
    y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Make fraud cases slightly more separable
    X[y == 1] += np.random.randn(sum(y == 1), n_features) * 0.5
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return X, y, feature_names


def train_fraud_model(
    X_train, X_test, y_train, y_test,
    feature_names,
    model_params,
    run_name=None,
    experiment_name="fraud_detection"
):
    """
    Train a fraud detection model with MLflow tracking
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        feature_names: Names of features
        model_params: Model hyperparameters
        run_name: Name for this training run
        experiment_name: MLflow experiment name
    
    Returns:
        Trained model and run_id
    """
    print("=" * 60)
    print("Training Fraud Detection Model")
    print("=" * 60)
    
    # Initialize MLflow tracker
    tracker = FraudMLflowTracker(experiment_name=experiment_name)
    
    # Start MLflow run
    start_time = datetime.now()
    
    with tracker.start_run(run_name=run_name) as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"Experiment: {experiment_name}")
        
        # Set tags
        tracker.set_tags({
            'model_type': 'RandomForest',
            'task': 'fraud_detection',
            'training_date': start_time.isoformat(),
            'environment': 'development'
        })
        
        # Log model parameters
        print("\nLogging model parameters...")
        tracker.log_model_params(model_params)
        
        # Log dataset information
        print("Logging dataset information...")
        fraud_rate_train = np.mean(y_train)
        fraud_rate_test = np.mean(y_test)
        
        tracker.log_dataset_info({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'train_fraud_rate': float(fraud_rate_train),
            'test_fraud_rate': float(fraud_rate_test),
            'train_normal_samples': int(np.sum(y_train == 0)),
            'train_fraud_samples': int(np.sum(y_train == 1)),
            'test_normal_samples': int(np.sum(y_test == 0)),
            'test_fraud_samples': int(np.sum(y_test == 1))
        })
        
        # Train the model
        print("\nTraining model...")
        model = RandomForestClassifier(**model_params, random_state=42)
        model.fit(X_train, y_train)
        print("✓ Training complete")
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        print("Calculating metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Log metrics
        print("Logging metrics to MLflow...")
        tracker.log_fraud_detection_metrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
        
        # Log additional metrics
        tracker.log_model_metrics({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        })
        
        # Log confusion matrix
        print("Logging confusion matrix...")
        tracker.log_confusion_matrix(cm)
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            print("Logging feature importance...")
            tracker.log_feature_importance(
                feature_names=feature_names,
                importance_values=model.feature_importances_.tolist()
            )
        
        # Log training metadata
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        tracker.log_training_metadata({
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'training_duration_seconds': training_duration,
            'n_estimators_trained': model.n_estimators,
            'n_classes': len(np.unique(y_train))
        })
        
        # Log the model
        print("Logging model to MLflow...")
        tracker.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud_detection_model"
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("Training Results")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc_roc:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"FNR:       {fnr:.4f}")
        print("\nConfusion Matrix:")
        print(f"TN: {tn:5d}  |  FP: {fp:5d}")
        print(f"FN: {fn:5d}  |  TP: {tp:5d}")
        print("\nTraining Duration: {:.2f} seconds".format(training_duration))
        print("=" * 60)
        
        print(f"\n✓ Run logged successfully!")
        print(f"Run ID: {run.info.run_id}")
        print(f"View in MLflow UI: {tracker.tracking_uri}")
        
        return model, run.info.run_id


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("Fraud Detection Model Training with MLflow")
    print("=" * 60)
    
    # Setup MLflow configuration
    print("\nSetting up MLflow...")
    config = setup_mlflow()
    
    # Generate or load data
    print("\nLoading data...")
    X, y, feature_names = generate_sample_data(n_samples=10000)
    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"  Fraud rate: {np.mean(y):.2%}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Test:  {len(X_test)} samples")
    
    # Define model parameters
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'class_weight': 'balanced'
    }
    
    # Train model
    model, run_id = train_fraud_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        model_params=model_params,
        run_name="random_forest_baseline",
        experiment_name="fraud_detection"
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. View results in MLflow UI: {config.mlflow_tracking_uri}")
    print(f"2. Compare with other runs")
    print(f"3. Deploy best model to production")
    print(f"\nTo load this model later:")
    print(f"  from mlflow_utils import FraudMLflowTracker")
    print(f"  tracker = FraudMLflowTracker()")
    print(f"  model = tracker.load_model('{run_id}')")
    print()


if __name__ == "__main__":
    main()