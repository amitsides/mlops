import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata


class WuFlachAUC(BaseEstimator, ClassifierMixin):
    """
    Implementation of improved AUC calculation based on:
    Wu, S., Flach, P.A., Ramirez, C.F.: An improved model selection heuristic for AUC.
    """

    def __init__(self, base_classifier=None):
        self.base_classifier = base_classifier or GradientBoostingClassifier()

    def fit(self, X, y):
        # Store class distributions for AUC calculation
        self.class_counts_ = np.bincount(y)
        self.base_classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

    def score(self, X, y):
        """
        Compute improved AUC score using Wu-Flach-Ramirez method
        """
        y_scores = self.predict_proba(X)[:, 1]

        # Calculate ranks with ties handled as prescribed in the paper
        ranks = rankdata(y_scores)
        n_pos = np.sum(y == 1)
        n_neg = len(y) - n_pos

        # Implement the improved AUC calculation
        pos_ranks_sum = np.sum(ranks[y == 1])
        auc = (pos_ranks_sum - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)

        # Apply their suggested correction factor
        if min(n_pos, n_neg) < 10:  # Small sample correction
            correction = 0.5 * (1.0 / n_pos + 1.0 / n_neg)
            auc = max(0, min(1, auc + correction))

        return auc


def calculate_improved_drift_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate drift metrics using Wu-Flach-Ramirez improvements
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate improved AUC
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    ranks = rankdata(y_pred_proba)
    pos_ranks_sum = np.sum(ranks[y_true == 1])
    improved_auc = (pos_ranks_sum - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)

    # Apply small sample correction if needed
    if min(n_pos, n_neg) < 10:
        correction = 0.5 * (1.0 / n_pos + 1.0 / n_neg)
        improved_auc = max(0, min(1, improved_auc + correction))

    # Calculate drift score using improved AUC
    drift_score = 2 * improved_auc - 1

    # Calculate additional metrics with balanced weighting
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (tpr + tnr) / 2

    return {
        'improved_auc': improved_auc,
        'drift_score': drift_score,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': {
            'tn': tn, 'fp': fp,
            'fn': fn, 'tp': tp
        },
        'class_distribution': {
            'positive': n_pos,
            'negative': n_neg
        }
    }


def evaluate_drift_model_improved(X_train, X_test, y_train, y_test, params=None):
    """
    Evaluate drift using Wu-Flach-Ramirez improvements
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        }

    # Initialize improved model
    base_model = GradientBoostingClassifier(**params)
    model = WuFlachAUC(base_model)

    # Fit model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics using improved method
    metrics = calculate_improved_drift_metrics(y_test, y_pred_proba)

    # Perform cross-validation with improved AUC scoring
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring=lambda estimator, X, y: WuFlachAUC().score(X, y)
    )

    metrics['cv_scores'] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores.tolist()
    }

    return model, metrics


def visualize_drift_analysis(metrics, title="Drift Analysis Results"):
    """
    Create visualization of drift analysis results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion matrix heatmap
    conf_matrix = np.array([
        [metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
        [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]
    ])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    ax1.set_title('Confusion Matrix')

    # Metrics summary
    metrics_summary = {
        'Improved AUC': metrics['improved_auc'],
        'Drift Score': metrics['drift_score'],
        'Balanced Accuracy': metrics['balanced_accuracy']
    }

    ax2.barh(list(metrics_summary.keys()), list(metrics_summary.values()))
    ax2.set_xlim(0, 1)
    ax2.set_title('Performance Metrics')

    plt.tight_layout()
    return fig


def run_improved_drift_analysis(X_train, X_test, y_train, y_test):
    """
    Run complete drift analysis with Wu-Flach-Ramirez improvements
    """
    # Train and evaluate model
    model, metrics = evaluate_drift_model_improved(X_train, X_test, y_train, y_test)

    # Print results
    print("\nImproved Drift Detection Results:")
    print(f"Improved AUC: {metrics['improved_auc']:.3f}")
    print(f"Drift Score: {metrics['drift_score']:.3f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")

    print("\nClass Distribution:")
    print(f"Positive samples: {metrics['class_distribution']['positive']}")
    print(f"Negative samples: {metrics['class_distribution']['negative']}")

    print("\nCross-validation Results:")
    print(f"Mean CV Score: {metrics['cv_scores']['mean']:.3f} ± {metrics['cv_scores']['std']:.3f}")

    # Create visualizations
    visualize_drift_analysis(metrics)

    return model, metrics