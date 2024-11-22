import numpy as np
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim


class StochasticAUCMaximization:
    """
    Implements stochastic AUC maximization as described in the paper:
    'Optimization Methods for Large-Scale AUC Maximization'
    """

    def __init__(self, learning_rate=0.01, batch_size=256):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def pairwise_hinge_loss(self, pos_scores, neg_scores):
        """
        Compute pairwise hinge loss for AUC optimization
        """
        pos_scores = pos_scores.unsqueeze(1)
        neg_scores = neg_scores.unsqueeze(0)
        loss = torch.clamp(1 - (pos_scores - neg_scores), min=0)
        return loss.mean()

    def fit(self, X, y, epochs=100):
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        # Initialize model
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(epochs):
            # Stochastic batch selection
            pos_idx = torch.where(y == 1)[0]
            neg_idx = torch.where(y == 0)[0]

            pos_batch = pos_idx[torch.randint(len(pos_idx), (self.batch_size,))]
            neg_batch = neg_idx[torch.randint(len(neg_idx), (self.batch_size,))]

            pos_scores = self.model(X[pos_batch]).squeeze()
            neg_scores = self.model(X[neg_batch]).squeeze()

            loss = self.pairwise_hinge_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_proba(self, X):
        X = torch.FloatTensor(X)
        with torch.no_grad():
            return self.model(X).numpy()


class DriftDetector:
    """
    Comprehensive drift detector implementing multiple statistical tests
    """

    def __init__(self):
        self.categorical_encoder = LabelEncoder()

    def cramers_v(self, x, y):
        """
        Calculate Cramér's V statistic between two categorical variables
        """
        confusion_matrix = np.histogram2d(x, y)[0]
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        min_dim = min(confusion_matrix.shape) - 1

        return np.sqrt(chi2 / (n * min_dim))

    def calculate_categorical_drift(self, source, target):
        """
        Calculate categorical drift using Cramér's V
        """
        # Encode categorical variables
        encoded_source = self.categorical_encoder.fit_transform(source)
        encoded_target = self.categorical_encoder.transform(target)

        return self.cramers_v(encoded_source, encoded_target)

    def calculate_continuous_drift(self, source, target):
        """
        Calculate continuous distribution drift using Kolmogorov-Smirnov test
        """
        ks_statistic, p_value = ks_2samp(source, target)
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value
        }

    def detect_feature_drift(self, source_data, target_data, categorical_features=None):
        """
        Detect drift for each feature using appropriate test
        """
        if categorical_features is None:
            categorical_features = []

        drift_scores = {}

        for feature_idx in range(source_data.shape[1]):
            feature_name = f"feature_{feature_idx}"
            source_feature = source_data[:, feature_idx]
            target_feature = target_data[:, feature_idx]

            if feature_idx in categorical_features:
                drift_score = self.calculate_categorical_drift(
                    source_feature, target_feature
                )
                drift_scores[feature_name] = {
                    'type': 'categorical',
                    'cramers_v': drift_score
                }
            else:
                drift_score = self.calculate_continuous_drift(
                    source_feature, target_feature
                )
                drift_scores[feature_name] = {
                    'type': 'continuous',
                    'ks_test': drift_score
                }

        return drift_scores


def evaluate_comprehensive_drift(source_X, target_X, source_y, target_y,
                                 categorical_features=None, threshold=0.05):
    """
    Comprehensive drift evaluation combining multiple approaches
    """
    # Initialize detectors
    drift_detector = DriftDetector()
    auc_optimizer = StochasticAUCMaximization()

    # Detect feature-level drift
    feature_drift = drift_detector.detect_feature_drift(
        source_X, target_X, categorical_features
    )

    # Train AUC-optimized classifier
    auc_optimizer.fit(source_X, source_y)

    # Get predictions on target data
    target_pred = auc_optimizer.predict_proba(target_X)

    # Calculate optimized AUC score
    auc_score = roc_auc_score(target_y, target_pred)

    # Aggregate results
    drift_summary = {
        'feature_drift': feature_drift,
        'auc_score': auc_score,
        'drift_detected': False,
        'drift_features': []
    }

    # Identify drifted features
    for feature, scores in feature_drift.items():
        if scores['type'] == 'categorical':
            if scores['cramers_v'] > 0.3:  # Common threshold for Cramér's V
                drift_summary['drift_features'].append(feature)
                drift_summary['drift_detected'] = True
        else:
            if scores['ks_test']['p_value'] < threshold:
                drift_summary['drift_features'].append(feature)
                drift_summary['drift_detected'] = True

    return drift_summary


def plot_drift_analysis(drift_summary):
    """
    Visualize drift analysis results
    """
    plt.figure(figsize=(12, 6))

    # Plot feature drift scores
    drift_scores = []
    feature_names = []

    for feature, scores in drift_summary['feature_drift'].items():
        feature_names.append(feature)
        if scores['type'] == 'categorical':
            drift_scores.append(scores['cramers_v'])
        else:
            drift_scores.append(scores['ks_test']['ks_statistic'])

    plt.subplot(1, 2, 1)
    plt.barh(feature_names, drift_scores)
    plt.title('Feature Drift Scores')

    # Plot AUC score
    plt.subplot(1, 2, 2)
    plt.pie([drift_summary['auc_score'], 1 - drift_summary['auc_score']],
            labels=['AUC', '1-AUC'],
            autopct='%1.1f%%')
    plt.title('AUC Score Distribution')

    plt.tight_layout()
    return plt


# Example usage
def run_drift_analysis(source_X, target_X, source_y, target_y, categorical_features=None):
    """
    Run complete drift analysis with all implemented methods
    """
    results = evaluate_comprehensive_drift(
        source_X, target_X, source_y, target_y, categorical_features
    )

    print("\nDrift Analysis Results:")
    print(f"AUC Score: {results['auc_score']:.3f}")
    print(f"Drift Detected: {results['drift_detected']}")

    if results['drift_detected']:
        print("\nDrifted Features:")
        for feature in results['drift_features']:
            feature_scores = results['feature_drift'][feature]
            if feature_scores['type'] == 'categorical':
                print(f"{feature}: Cramér's V = {feature_scores['cramers_v']:.3f}")
            else:
                print(f"{feature}: KS statistic = {feature_scores['ks_test']['ks_statistic']:.3f} "
                      f"(p-value: {feature_scores['ks_test']['p_value']:.3f})")

    # Visualize results
    plot_drift_analysis(results)

    return results