import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_drift_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate comprehensive drift metrics including AUC and derived scores

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fpr = fp / (fp + tn)  # False Positive Rate

    # Calculate AUC and drift score
    auc = roc_auc_score(y_true, y_pred_proba)
    drift_score = 2 * auc - 1

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'auc': auc,
        'drift_score': drift_score,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }


def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Actual 0', 'Actual 1']
    )
    plt.title(title)
    return plt


def evaluate_drift_model(X_train, X_test, y_train, y_test, params=None):
    """
    Train and evaluate drift detection model with cross-validation

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        params: GradientBoostingClassifier parameters

    Returns:
        Trained model and evaluation metrics
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        }

    # Initialize and train model
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    # Get predictions on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate comprehensive metrics
    metrics = calculate_drift_metrics(y_test, y_pred_proba)

    # Add cross-validation results
    metrics['cv_scores'] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores.tolist()
    }

    return model, metrics


# Example usage function
def run_drift_analysis(X_train, X_test, y_train, y_test):
    """
    Run complete drift analysis and print results
    """
    # Train and evaluate model
    model, metrics = evaluate_drift_model(X_train, X_test, y_train, y_test)

    # Print results
    print("\nDrift Detection Results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"False Positive Rate: {metrics['fpr']:.3f}")
    print(f"AUC: {metrics['auc']:.3f}")
    print(f"Drift Score: {metrics['drift_score']:.3f}")

    print("\nCross-validation Results:")
    print(f"Mean CV Score: {metrics['cv_scores']['mean']:.3f} ± {metrics['cv_scores']['std']:.3f}")

    # Plot confusion matrix
    conf_matrix = np.array([
        [metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
        [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]
    ])
    plot_confusion_matrix(conf_matrix)

    return model, metrics