"""
Label Noise Detection via Confident Learning

This module implements a robust label noise detection framework based on
Confident Learning (Northcutt et al., 2021) with extensions for:
- Multi-classifier ensemble probability estimation
- Noise transition matrix estimation
- Noise type categorization (uniform, class-conditional, instance-dependent)
- Statistical significance testing via permutation tests

References:
    - Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident Learning:
      Estimating Uncertainty in Dataset Labels. JAIR, 70, 1373-1411.
    - Frénay, B., & Verleysen, M. (2014). Classification in the presence of
      label noise: a survey. IEEE TNNLS, 25(5), 845-869.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from scipy.special import rel_entr

from report_components.base_component import ReportComponent
from utils.consts import NUM_EXAMPLES_LLM


class NoiseType(Enum):
    """Categorization of label noise patterns."""
    UNIFORM = "uniform"  # Random noise, independent of true class
    CLASS_CONDITIONAL = "class_conditional"  # Noise depends on true class
    INSTANCE_DEPENDENT = "instance_dependent"  # Noise depends on features


@dataclass
class NoiseAnalysisResult:
    """Structured container for noise analysis results."""
    noise_indices: List[int]
    noise_scores: np.ndarray
    noise_ratio: float
    noise_transition_matrix: np.ndarray
    estimated_joint_distribution: np.ndarray
    class_thresholds: Dict[str, float]
    noise_type: NoiseType
    noise_type_confidence: float
    per_class_noise_rates: Dict[str, float]
    statistical_significance: Dict[str, Any]
    ensemble_agreement: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LabelNoiseDetectionComponent(ReportComponent):
    """
    Advanced Label Noise Detection using Confident Learning.

    This component implements a comprehensive framework for detecting and
    characterizing label noise in supervised learning datasets. It combines
    multiple theoretical foundations:

    1. **Confident Learning**: Estimates the joint distribution Q(ŷ, y*) of
       noisy labels ŷ and latent true labels y* using out-of-fold predictions.

    2. **Noise Transition Matrix**: Estimates T where T[i,j] = P(ŷ=j | y*=i),
       characterizing how true labels flip to observed labels.

    3. **Multi-Classifier Ensemble**: Uses calibrated probability estimates
       from diverse classifiers to improve robustness.

    4. **Statistical Validation**: Employs permutation testing to assess
       whether detected noise is statistically significant.

    Attributes:
        target_column: Name of the label column
        cv_folds: Number of cross-validation folds for out-of-fold prediction
        n_permutations: Number of permutations for significance testing
        confidence_threshold: Minimum confidence for noise detection
        use_ensemble: Whether to use multi-classifier ensemble
    """

    def __init__(
        self,
        context,
        target_column: str,
        cv_folds: int = 5,
        n_permutations: int = 100,
        confidence_threshold: float = 0.5,
        use_ensemble: bool = True,
        calibrate_probabilities: bool = True,
        use_llm_explanations: bool = True  # Enable LLM-powered explanations
    ):
        super().__init__(context, use_llm_explanations)
        self.target_column = target_column
        self.cv_folds = cv_folds
        self.n_permutations = n_permutations
        self.confidence_threshold = confidence_threshold
        self.use_ensemble = use_ensemble
        self.calibrate_probabilities = calibrate_probabilities
        self._result: Optional[NoiseAnalysisResult] = None

    def _get_classifier_ensemble(self) -> List[Tuple[str, Any]]:
        """
        Returns a diverse ensemble of classifiers for robust probability estimation.

        Diversity is achieved through:
        - Different inductive biases (tree-based, linear, neural)
        - Different regularization strategies
        - Different decision boundaries
        """
        classifiers = [
            ("RandomForest", RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            )),
            ("GradientBoosting", GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42
            )),
            ("LogisticRegression", LogisticRegression(
                max_iter=1000, C=1.0, random_state=42, n_jobs=-1
            )),
            ("MLP", MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500,
                early_stopping=True, random_state=42
            ))
        ]
        return classifiers

    def _compute_out_of_fold_probabilities(
        self, X: np.ndarray, y: np.ndarray, n_classes: int
    ) -> Tuple[np.ndarray, float]:
        """
        Compute out-of-fold predicted probabilities using cross-validation.

        This is critical for Confident Learning as it prevents information
        leakage - each sample's probability is estimated by a model that
        never saw that sample during training.

        Args:
            X: Feature matrix
            y: Label vector
            n_classes: Number of unique classes

        Returns:
            Tuple of (probability matrix, ensemble agreement score)
        """
        kf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=42
        )

        if self.use_ensemble:
            classifiers = self._get_classifier_ensemble()
        else:
            classifiers = [("RandomForest", RandomForestClassifier(
                n_estimators=100, random_state=42
            ))]

        # Store predictions from each classifier
        all_predictions = []

        for clf_name, clf in classifiers:
            pred_probs = np.zeros((len(y), n_classes))

            try:
                for train_idx, val_idx in kf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]

                    # Scale features for non-tree-based models
                    if clf_name in ["LogisticRegression", "MLP"]:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_val = scaler.transform(X_val)

                    # Clone classifier for each fold
                    from sklearn.base import clone
                    fold_clf = clone(clf)

                    # Optionally calibrate probabilities
                    if self.calibrate_probabilities and clf_name != "LogisticRegression":
                        fold_clf = CalibratedClassifierCV(
                            fold_clf, method='isotonic', cv=3
                        )

                    fold_clf.fit(X_train, y_train)
                    fold_probs = fold_clf.predict_proba(X_val)

                    # Handle missing classes in fold
                    if fold_probs.shape[1] < n_classes:
                        full_probs = np.zeros((len(val_idx), n_classes))
                        classes_in_fold = fold_clf.classes_
                        for idx, c in enumerate(classes_in_fold):
                            full_probs[:, c] = fold_probs[:, idx]
                        fold_probs = full_probs

                    pred_probs[val_idx] = fold_probs

                all_predictions.append(pred_probs)

            except Exception as e:
                warnings.warn(f"Classifier {clf_name} failed: {str(e)}")
                continue

        if not all_predictions:
            raise RuntimeError("All classifiers failed during probability estimation")

        # Ensemble aggregation via averaging
        ensemble_probs = np.mean(all_predictions, axis=0)

        # Compute ensemble agreement (measures classifier consensus)
        if len(all_predictions) > 1:
            pred_labels = [np.argmax(p, axis=1) for p in all_predictions]
            agreement_matrix = np.zeros((len(y), len(all_predictions)))
            for i, preds in enumerate(pred_labels):
                agreement_matrix[:, i] = preds

            # Compute modal agreement
            from scipy.stats import mode
            modes, counts = mode(agreement_matrix, axis=1)
            ensemble_agreement = float(np.mean(counts / len(all_predictions)))
        else:
            ensemble_agreement = 1.0

        return ensemble_probs, ensemble_agreement

    def _estimate_confident_joint(
        self, y: np.ndarray, pred_probs: np.ndarray, n_classes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the confident joint distribution C_ŷ,y*.

        The confident joint counts the number of examples with noisy label ŷ=i
        that are confidently predicted to have true label y*=j.

        This follows the Confident Learning framework (Northcutt et al., 2021).

        Returns:
            Tuple of (confident joint C, thresholds t, noise scores)
        """
        # Step 1: Estimate thresholds t_j = E[P(y*=j | x) | ŷ=j]
        # Average predicted probability for class j among samples labeled as j
        thresholds = np.zeros(n_classes)
        for j in range(n_classes):
            mask = y == j
            if np.sum(mask) > 0:
                thresholds[j] = np.mean(pred_probs[mask, j])
            else:
                thresholds[j] = 0.5  # Default threshold for empty classes

        # Step 2: Build the confident joint
        confident_joint = np.zeros((n_classes, n_classes), dtype=int)
        noise_scores = np.zeros(len(y))

        for i, (label, probs) in enumerate(zip(y, pred_probs)):
            # Find confident predictions (prob >= threshold)
            confident_mask = probs >= thresholds
            confident_classes = np.where(confident_mask)[0]

            if len(confident_classes) == 0:
                # No confident prediction, use argmax
                predicted_true_label = np.argmax(probs)
            elif len(confident_classes) == 1:
                predicted_true_label = confident_classes[0]
            else:
                # Multiple confident predictions, use highest probability
                predicted_true_label = confident_classes[np.argmax(probs[confident_classes])]

            confident_joint[label, predicted_true_label] += 1

            # Compute noise score: how likely is this label wrong?
            # High score = high confidence in different class than labeled
            max_other_prob = np.max(np.delete(probs, label))
            self_prob = probs[label]
            noise_scores[i] = max_other_prob - self_prob + (1 - self_prob)

        return confident_joint, thresholds, noise_scores

    def _estimate_noise_transition_matrix(
        self, confident_joint: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the noise transition matrix T from the confident joint.

        T[i,j] = P(ŷ=j | y*=i) represents the probability that a sample
        with true label i is observed with noisy label j.

        For clean data: T ≈ I (identity matrix)
        """
        # Normalize columns (latent true labels) to get conditional probabilities
        column_sums = confident_joint.sum(axis=0, keepdims=True)
        column_sums = np.maximum(column_sums, 1)  # Avoid division by zero

        # T[i,j] = P(ŷ=i | y*=j) - note the transposition
        T = confident_joint / column_sums

        return T.T  # Transpose to get T[i,j] = P(ŷ=j | y*=i)

    def _identify_noise_type(
        self, transition_matrix: np.ndarray, per_class_noise: np.ndarray
    ) -> Tuple[NoiseType, float]:
        """
        Categorize the type of label noise based on transition matrix patterns.

        - Uniform: All off-diagonal elements are similar
        - Class-conditional: Off-diagonal varies significantly by row
        - Instance-dependent: Requires additional analysis (default if others don't fit)
        """
        n_classes = transition_matrix.shape[0]
        off_diagonal_mask = ~np.eye(n_classes, dtype=bool)
        off_diagonal = transition_matrix[off_diagonal_mask]

        if len(off_diagonal) == 0 or np.all(off_diagonal < 0.01):
            return NoiseType.UNIFORM, 0.95  # Very clean data

        # Test for uniform noise: all off-diagonal elements similar
        off_diag_std = np.std(off_diagonal)
        off_diag_mean = np.mean(off_diagonal)

        coefficient_of_variation = off_diag_std / (off_diag_mean + 1e-10)

        if coefficient_of_variation < 0.3:
            return NoiseType.UNIFORM, 1.0 - coefficient_of_variation

        # Test for class-conditional: significant variance in per-class noise rates
        per_class_std = np.std(per_class_noise)
        per_class_mean = np.mean(per_class_noise)
        class_cv = per_class_std / (per_class_mean + 1e-10)

        if class_cv > 0.5:
            return NoiseType.CLASS_CONDITIONAL, min(1.0, class_cv)

        # Default to instance-dependent
        return NoiseType.INSTANCE_DEPENDENT, 0.6

    def _permutation_test(
        self, X: np.ndarray, y: np.ndarray, observed_noise_ratio: float,
        n_classes: int
    ) -> Dict[str, Any]:
        """
        Perform permutation test to assess statistical significance.

        Null hypothesis: The observed noise ratio could occur by chance
        in a dataset with random label-feature associations.
        """
        if self.n_permutations < 10:
            return {"p_value": None, "significant": None, "message": "Skipped"}

        permuted_ratios = []
        rng = np.random.RandomState(42)

        for _ in range(min(self.n_permutations, 50)):  # Limit for performance
            y_perm = rng.permutation(y)
            try:
                probs_perm, _ = self._compute_out_of_fold_probabilities(
                    X, y_perm, n_classes
                )
                joint_perm, _, scores_perm = self._estimate_confident_joint(
                    y_perm, probs_perm, n_classes
                )
                noise_mask_perm = scores_perm > self.confidence_threshold
                permuted_ratios.append(np.mean(noise_mask_perm))
            except Exception:
                continue

        if len(permuted_ratios) < 5:
            return {"p_value": None, "significant": None, "message": "Insufficient permutations"}

        # One-sided p-value: probability of observing lower noise ratio under null
        p_value = np.mean(np.array(permuted_ratios) <= observed_noise_ratio)

        return {
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "null_distribution_mean": float(np.mean(permuted_ratios)),
            "null_distribution_std": float(np.std(permuted_ratios)),
            "effect_size": float(
                (observed_noise_ratio - np.mean(permuted_ratios)) /
                (np.std(permuted_ratios) + 1e-10)
            )
        }

    def analyze(self) -> None:
        """
        Execute the complete label noise detection pipeline.

        Pipeline stages:
        1. Data preparation and validation
        2. Out-of-fold probability estimation
        3. Confident joint estimation
        4. Noise transition matrix computation
        5. Noise type categorization
        6. Statistical significance testing
        """
        df = self.context.dataset.df

        # Validate inputs
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        # Prepare features and labels
        X = df.drop(columns=[self.target_column]).select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            raise ValueError("No numeric features available for analysis")

        # Handle missing values
        X = X.fillna(X.median())
        X_array = X.values

        # Encode labels
        y_series = df[self.target_column].astype('category')
        labels = y_series.cat.categories.tolist()
        y = y_series.cat.codes.values
        n_classes = len(labels)

        if n_classes < 2:
            raise ValueError("At least 2 classes required for noise detection")

        # Stage 1: Out-of-fold probability estimation
        pred_probs, ensemble_agreement = self._compute_out_of_fold_probabilities(
            X_array, y, n_classes
        )

        # Stage 2: Confident joint estimation
        confident_joint, thresholds, noise_scores = self._estimate_confident_joint(
            y, pred_probs, n_classes
        )

        # Stage 3: Noise transition matrix
        transition_matrix = self._estimate_noise_transition_matrix(confident_joint)

        # Stage 4: Identify noisy samples
        noise_mask = noise_scores > self.confidence_threshold
        noise_indices = np.where(noise_mask)[0].tolist()
        noise_ratio = float(np.mean(noise_mask))

        # Stage 5: Per-class noise rates
        per_class_noise = {}
        per_class_noise_array = np.zeros(n_classes)
        for i, label in enumerate(labels):
            class_mask = y == i
            if np.sum(class_mask) > 0:
                rate = float(np.mean(noise_mask[class_mask]))
            else:
                rate = 0.0
            per_class_noise[str(label)] = rate
            per_class_noise_array[i] = rate

        # Stage 6: Noise type categorization
        noise_type, type_confidence = self._identify_noise_type(
            transition_matrix, per_class_noise_array
        )

        # Stage 7: Statistical significance (optional, expensive)
        if self.n_permutations > 0 and len(noise_indices) > 0:
            stat_significance = self._permutation_test(
                X_array, y, noise_ratio, n_classes
            )
        else:
            stat_significance = {"p_value": None, "significant": None}

        # Normalize joint distribution
        joint_sum = confident_joint.sum()
        if joint_sum > 0:
            estimated_joint = confident_joint / joint_sum
        else:
            estimated_joint = confident_joint.astype(float)

        # Compile results
        self._result = NoiseAnalysisResult(
            noise_indices=noise_indices,
            noise_scores=noise_scores,
            noise_ratio=noise_ratio,
            noise_transition_matrix=transition_matrix,
            estimated_joint_distribution=estimated_joint,
            class_thresholds={str(labels[i]): float(thresholds[i]) for i in range(n_classes)},
            noise_type=noise_type,
            noise_type_confidence=type_confidence,
            per_class_noise_rates=per_class_noise,
            statistical_significance=stat_significance,
            ensemble_agreement=ensemble_agreement,
            metadata={
                "n_samples": len(y),
                "n_classes": n_classes,
                "n_features": X_array.shape[1],
                "cv_folds": self.cv_folds,
                "confidence_threshold": self.confidence_threshold
            }
        )

        # Store in shared artifacts for other components
        self.context.shared_artifacts["label_noise_mask"] = noise_mask
        self.context.shared_artifacts["label_noise_ratio"] = noise_ratio
        self.context.shared_artifacts["label_noise_scores"] = noise_scores
        self.context.shared_artifacts["noise_transition_matrix"] = transition_matrix

        # Legacy format for compatibility
        self.result = {
            "noise_indices": noise_indices,
            "noise_ratio": noise_ratio,
            "class_thresholds": self._result.class_thresholds
        }

    def summarize(self) -> dict:
        if self._result is None:
            return {"error": "Analysis not yet performed"}

        llm_explanations = []
        if self.llm and len(self._result.noise_indices) > 0:
            df = self.context.dataset.df
            for idx in self._result.noise_indices[:NUM_EXAMPLES_LLM]:
                try:
                    row_data = df.iloc[idx].to_dict()
                    current_label = row_data.get(self.target_column)
                    pred_probs = self.context.shared_artifacts.get("label_noise_scores", None)
                    suggested_label = "uncertain"
                    confidence = self._result.noise_scores[idx]

                    explanation = self.llm.explain_label_noise(
                        row_data=row_data,
                        current_label=current_label,
                        suggested_label=suggested_label,
                        confidence=confidence
                    )
                    llm_explanations.append({
                        "row_index": idx,
                        "current_label": current_label,
                        "noise_score": round(float(self._result.noise_scores[idx]), 4),
                        "llm_explanation": explanation
                    })
                except Exception:
                    pass

        summary = {
            "noise_ratio": round(self._result.noise_ratio, 4),
            "suspicious_sample_count": len(self._result.noise_indices),
            "noise_type": self._result.noise_type.value,
            "noise_type_confidence": round(self._result.noise_type_confidence, 3),
            "ensemble_agreement": round(self._result.ensemble_agreement, 3),
            "per_class_noise_rates": {
                k: round(v, 4) for k, v in self._result.per_class_noise_rates.items()
            },
            "statistical_significance": self._result.statistical_significance.get("significant"),
            "p_value": self._result.statistical_significance.get("p_value"),
            "top_suspicious_indices": self._result.noise_indices[:20],
            "transition_matrix_diagonal_mean": round(
                float(np.diag(self._result.noise_transition_matrix).mean()), 4
            )
        }

        if llm_explanations:
            print(f"\n{'='*80}")
            print("🤖 LLM EXPLANATIONS")
            print(f"{'='*80}")
            for i, expl in enumerate(llm_explanations, 1):
                print(f"\n{i}. Row {expl['row_index']} - Label: {expl['current_label']}")
                print(f"   Noise Score: {expl['noise_score']}")
                print(f"   {expl['llm_explanation']}")
            print(f"{'='*80}\n")

        return summary

    def justify(self) -> str:
        """Provide theoretical justification for the methodology."""
        return (
            "This component implements the Confident Learning framework (Northcutt et al., 2021) "
            "for principled label noise detection. The methodology is grounded in the following "
            "theoretical foundations:\n\n"
            "1. **Joint Distribution Estimation**: We estimate Q(ŷ, y*), the joint distribution "
            "of noisy observed labels ŷ and latent true labels y*, using out-of-fold predictions "
            "to prevent information leakage.\n\n"
            "2. **Noise Transition Matrix**: The matrix T where T[i,j] = P(ŷ=j | y*=i) characterizes "
            "the label corruption process, enabling noise type categorization (uniform, class-conditional, "
            "or instance-dependent).\n\n"
            "3. **Calibrated Ensemble Probabilities**: We use a diverse ensemble of classifiers "
            "(Random Forest, Gradient Boosting, Logistic Regression, MLP) with isotonic calibration "
            "to obtain robust probability estimates.\n\n"
            "4. **Statistical Validation**: Permutation testing provides p-values assessing whether "
            "detected noise exceeds what would occur by chance, following the framework of "
            "Ojala & Garriga (2010).\n\n"
            "This approach decouples epistemic uncertainty (model uncertainty) from aleatoric noise "
            "(inherent label ambiguity), providing actionable insights for data cleaning."
        )

    def get_detailed_results(self) -> Optional[NoiseAnalysisResult]:
        """Return the full structured results object."""
        return self._result

