import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import html

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from report_components.base_component import ReportComponent
from utils.consts import (
    NUM_EXAMPLES_LLM, LABEL_NOISE_CV_FOLDS, LABEL_NOISE_N_PERMUTATIONS,
    LABEL_NOISE_CONFIDENCE_THRESHOLD, LABEL_NOISE_MIN_ROWS,
    LABEL_NOISE_MAX_SAMPLE_SIZE, LABEL_NOISE_LARGE_DATASET_CV_FOLDS,
    LABEL_NOISE_LARGE_DATASET_PERMUTATIONS, LARGE_DATASET_ROW_THRESHOLD
)


class NoiseType(Enum):
    UNIFORM = "uniform"
    CLASS_CONDITIONAL = "class_conditional"
    INSTANCE_DEPENDENT = "instance_dependent"


@dataclass
class NoiseAnalysisResult:
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
    def __init__(
        self,
        context,
        target_column: str,
        cv_folds: int = LABEL_NOISE_CV_FOLDS,
        n_permutations: int = LABEL_NOISE_N_PERMUTATIONS,
        confidence_threshold: float = LABEL_NOISE_CONFIDENCE_THRESHOLD,
        use_ensemble: bool = True,
        calibrate_probabilities: bool = True,
        use_llm_explanations: bool = True,
        max_sample_size: int = LABEL_NOISE_MAX_SAMPLE_SIZE
    ):
        super().__init__(context, use_llm_explanations)
        self.target_column = target_column
        self.cv_folds = cv_folds
        self.n_permutations = n_permutations
        self.confidence_threshold = confidence_threshold
        self.use_ensemble = use_ensemble
        self.calibrate_probabilities = calibrate_probabilities
        self.max_sample_size = max_sample_size
        self._result: Optional[NoiseAnalysisResult] = None
        self._is_sampled = False
        self._original_size = 0
        self._sample_indices = None

    def _get_classifier_ensemble(self, is_large_dataset: bool = False) -> List[Tuple[str, Any]]:
        if is_large_dataset:
            classifiers = [
                ("RandomForest", RandomForestClassifier(
                    n_estimators=50, max_depth=8, min_samples_leaf=10,
                    random_state=42, n_jobs=-1
                )),
                ("LogisticRegression", LogisticRegression(
                    max_iter=500, C=1.0, random_state=42, n_jobs=-1
                ))
            ]
        else:
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

    def _stratified_sample(self, X: np.ndarray, y: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split
        n_total = len(y)
        if n_total <= n_samples:
            return X, y, np.arange(n_total)
        sample_ratio = n_samples / n_total
        try:
            _, X_sample, _, y_sample, _, sample_indices = train_test_split(
                X, y, np.arange(n_total),
                test_size=sample_ratio,
                stratify=y,
                random_state=42
            )
        except ValueError:
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(n_total, size=n_samples, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
        return X_sample, y_sample, sample_indices

    def _compute_out_of_fold_probabilities(
        self, X: np.ndarray, y: np.ndarray, n_classes: int, is_large_dataset: bool = False
    ) -> Tuple[np.ndarray, float]:
        kf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=42
        )

        if self.use_ensemble:
            classifiers = self._get_classifier_ensemble(is_large_dataset)
        else:
            classifiers = [("RandomForest", RandomForestClassifier(
                n_estimators=50 if is_large_dataset else 100, random_state=42
            ))]

        all_predictions = []

        for clf_name, clf in classifiers:
            pred_probs = np.zeros((len(y), n_classes))

            try:
                for train_idx, val_idx in kf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]

                    if clf_name in ["LogisticRegression", "MLP"]:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_val = scaler.transform(X_val)

                    from sklearn.base import clone
                    fold_clf = clone(clf)

                    if self.calibrate_probabilities and clf_name != "LogisticRegression" and not is_large_dataset:
                        fold_clf = CalibratedClassifierCV(
                            fold_clf, method='isotonic', cv=3
                        )

                    fold_clf.fit(X_train, y_train)
                    fold_probs = fold_clf.predict_proba(X_val)

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

        off_diag_std = np.std(off_diagonal)
        off_diag_mean = np.mean(off_diagonal)

        coefficient_of_variation = off_diag_std / (off_diag_mean + 1e-10)

        if coefficient_of_variation < 0.3:
            return NoiseType.UNIFORM, 1.0 - coefficient_of_variation

        per_class_std = np.std(per_class_noise)
        per_class_mean = np.mean(per_class_noise)
        class_cv = per_class_std / (per_class_mean + 1e-10)

        if class_cv > 0.5:
            return NoiseType.CLASS_CONDITIONAL, min(1.0, class_cv)

        return NoiseType.INSTANCE_DEPENDENT, 0.6

    def _permutation_test(
        self, X: np.ndarray, y: np.ndarray, observed_noise_ratio: float,
        n_classes: int, is_large_dataset: bool = False
    ) -> Dict[str, Any]:
        if self.n_permutations < 5:
            return {"p_value": None, "significant": None, "message": "Skipped"}

        permuted_ratios = []
        rng = np.random.RandomState(42)

        max_perms = 5 if is_large_dataset else min(self.n_permutations, 20)

        for _ in range(max_perms):
            y_perm = rng.permutation(y)
            try:
                probs_perm, _ = self._compute_out_of_fold_probabilities(
                    X, y_perm, n_classes, is_large_dataset
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
        df = self.context.dataset.df

        if self.target_column not in df.columns:
            self._set_empty_result(f"Target column '{self.target_column}' not found")
            return

        X = df.drop(columns=[self.target_column]).select_dtypes(include=[np.number])
        if X.shape[1] == 0:
            self._set_empty_result("No numeric features available for analysis")
            return

        if len(df) < 10:
            self._set_empty_result("Dataset too small for label noise detection (need at least 10 rows)")
            return

        X = X.fillna(X.median())
        X_array = X.values

        y_series = df[self.target_column].astype('category')
        labels = y_series.cat.categories.tolist()
        y = y_series.cat.codes.values
        n_classes = len(labels)

        if n_classes < 2:
            self._set_empty_result("At least 2 classes required for noise detection")
            return

        self._original_size = len(y)
        is_large_dataset = len(y) > LARGE_DATASET_ROW_THRESHOLD

        if is_large_dataset:
            self.cv_folds = min(self.cv_folds, LABEL_NOISE_LARGE_DATASET_CV_FOLDS)
            self.n_permutations = min(self.n_permutations, LABEL_NOISE_LARGE_DATASET_PERMUTATIONS)
            self.calibrate_probabilities = False

        if len(y) > self.max_sample_size:
            X_array, y, self._sample_indices = self._stratified_sample(X_array, y, self.max_sample_size)
            self._is_sampled = True

        try:
            pred_probs, ensemble_agreement = self._compute_out_of_fold_probabilities(
                X_array, y, n_classes, is_large_dataset
            )

            confident_joint, thresholds, noise_scores = self._estimate_confident_joint(
                y, pred_probs, n_classes
            )

            transition_matrix = self._estimate_noise_transition_matrix(confident_joint)

            noise_mask = noise_scores > self.confidence_threshold
            noise_indices_in_sample = np.where(noise_mask)[0].tolist()

            if self._is_sampled and self._sample_indices is not None:
                noise_indices = [int(self._sample_indices[i]) for i in noise_indices_in_sample]
            else:
                noise_indices = noise_indices_in_sample

            noise_ratio = float(np.mean(noise_mask))

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

            noise_type, type_confidence = self._identify_noise_type(
                transition_matrix, per_class_noise_array
            )

            if self.n_permutations > 0 and len(noise_indices) > 0 and not is_large_dataset:
                stat_significance = self._permutation_test(
                    X_array, y, noise_ratio, n_classes, is_large_dataset
                )
            else:
                stat_significance = {"p_value": None, "significant": None}

            joint_sum = confident_joint.sum()
            if joint_sum > 0:
                estimated_joint = confident_joint / joint_sum
            else:
                estimated_joint = confident_joint.astype(float)

            metadata = {
                "n_samples": len(y),
                "n_classes": n_classes,
                "n_features": X_array.shape[1],
                "cv_folds": self.cv_folds,
                "confidence_threshold": self.confidence_threshold,
                "is_sampled": self._is_sampled,
                "original_size": self._original_size,
                "sample_size": len(y) if self._is_sampled else None,
                "is_large_dataset": is_large_dataset
            }

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
                metadata=metadata
            )

            self.context.shared_artifacts["label_noise_mask"] = noise_mask
            self.context.shared_artifacts["label_noise_ratio"] = noise_ratio
            self.context.shared_artifacts["label_noise_scores"] = noise_scores
            self.context.shared_artifacts["noise_transition_matrix"] = transition_matrix
            self.context.shared_artifacts["label_noise_pred_probs"] = pred_probs
            self.context.shared_artifacts["label_noise_labels"] = labels

            self.result = {
                "noise_indices": noise_indices,
                "noise_ratio": noise_ratio,
                "class_thresholds": self._result.class_thresholds
            }

        except Exception as e:
            self._set_empty_result(f"Analysis failed: {str(e)}")

    def _set_empty_result(self, reason: str):
        self._result = NoiseAnalysisResult(
            noise_indices=[],
            noise_scores=np.array([]),
            noise_ratio=0.0,
            noise_transition_matrix=np.array([[]]),
            estimated_joint_distribution=np.array([[]]),
            class_thresholds={},
            noise_type=NoiseType.UNIFORM,
            noise_type_confidence=0.0,
            per_class_noise_rates={},
            statistical_significance={"p_value": None, "significant": None},
            ensemble_agreement=0.0,
            metadata={"skipped_reason": reason}
        )
        self.result = {"skipped": True, "reason": reason}

    def summarize(self) -> dict:
        if self._result is None:
            return {"error": "Analysis not yet performed"}

        if "skipped_reason" in self._result.metadata:
            return {
                "skipped": True,
                "reason": self._result.metadata["skipped_reason"],
                "noise_ratio": 0.0
            }

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
            ) if self._result.noise_transition_matrix.size > 0 else 0.0
        }

        return summary

    def get_full_summary(self) -> str:
        if self._result is None:
            return "No analysis performed."

        if "skipped_reason" in self._result.metadata:
            return f"⚠️ Analysis skipped: {self._result.metadata['skipped_reason']}"

        lines = []
        llm_explanations = []

        if self.llm and len(self._result.noise_indices) > 0:
            df = self.context.dataset.df
            pred_probs = self.context.shared_artifacts.get("label_noise_pred_probs")
            labels = list(self._result.per_class_noise_rates.keys())
            transition_matrix = self._result.noise_transition_matrix

            for idx in self._result.noise_indices[:NUM_EXAMPLES_LLM]:
                try:
                    row_data = df.iloc[idx].to_dict()
                    current_label = row_data.get(self.target_column)
                    current_label_idx = labels.index(str(current_label)) if str(current_label) in labels else None

                    # Get model's predicted label and probabilities from learned data
                    model_prediction = None
                    model_confidence = None
                    current_label_prob = None
                    if pred_probs is not None and idx < len(pred_probs):
                        probs = pred_probs[idx]
                        predicted_idx = int(np.argmax(probs))
                        if predicted_idx < len(labels):
                            model_prediction = labels[predicted_idx]
                            model_confidence = float(probs[predicted_idx])
                        # Get probability for current label
                        if current_label_idx is not None and current_label_idx < len(probs):
                            current_label_prob = float(probs[current_label_idx])

                    # Get class noise rate from learned per-class rates
                    class_noise_rate = self._result.per_class_noise_rates.get(str(current_label))

                    # Get confused classes from transition matrix
                    confused_with = None
                    if current_label_idx is not None and transition_matrix.size > 0:
                        row = transition_matrix[current_label_idx] if current_label_idx < len(transition_matrix) else None
                        if row is not None:
                            confused_indices = np.argsort(row)[::-1][1:3]  # Top 2 classes it's confused with
                            confused_with = [labels[i] for i in confused_indices if i < len(labels) and row[i] > 0.05]

                    explanation = self.llm.explain_label_noise(
                        row_data=row_data,
                        current_label=current_label,
                        suggested_label=model_prediction or "uncertain",
                        confidence=self._result.noise_scores[idx],
                        model_prediction=model_prediction,
                        model_confidence=model_confidence,
                        current_label_prob=current_label_prob,
                        class_noise_rate=class_noise_rate,
                        confused_with=confused_with
                    )

                    # Capture row features for display (exclude likely identifiers dynamically)
                    row_features = {}
                    for col in df.columns:
                        # Skip columns that are likely identifiers:
                        # - Very high cardinality (>90% unique values)
                        # - Column name contains common identifier patterns
                        col_lower = col.lower()
                        is_likely_id = (
                            'id' in col_lower or
                            'name' in col_lower or
                            'ticket' in col_lower or
                            df[col].nunique() / len(df) > 0.9  # >90% unique = likely identifier
                        )
                        if not is_likely_id:
                            val = row_data.get(col)
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                row_features[col] = val

                    llm_explanations.append({
                        "row_index": idx,
                        "current_label": current_label,
                        "noise_score": round(float(self._result.noise_scores[idx]), 4),
                        "row_features": row_features,
                        "llm_explanation": explanation
                    })
                except Exception:
                    pass

        if llm_explanations:
            # Generate HTML cards for each suspicious row
            cards_html = []
            for expl in llm_explanations:
                # Build feature table rows
                feature_rows = []
                for col, val in expl.get('row_features', {}).items():
                    escaped_col = html.escape(str(col))
                    escaped_val = html.escape(str(val))
                    feature_rows.append(f"<tr><td class='ln-key'>{escaped_col}</td><td class='ln-val'>{escaped_val}</td></tr>")

                features_table = ''.join(feature_rows)
                escaped_label = html.escape(str(expl['current_label']))
                escaped_explanation = html.escape(str(expl['llm_explanation']))

                card = f"""<div class='ln-card'>
                    <div class='ln-title'>Row {expl['row_index']} · Label: {escaped_label}</div>
                    <div class='ln-meta'>Noise Score: {expl['noise_score']}</div>
                    <table class='ln-table'>{features_table}</table>
                    <div class='ln-exp'>{escaped_explanation}</div>
                </div>"""
                cards_html.append(card)

            grid_html = "<div class='ln-grid'>" + ''.join(cards_html) + "</div>"
            lines.append("🤖 LLM EXAMPLE EXPLANATIONS")
            lines.append(grid_html)

        if self.llm:
            try:
                summary_data = self.summarize()
                component_summary = self.llm.generate_component_summary(
                    component_name="Label Noise Detection",
                    metrics={"noise_ratio": summary_data["noise_ratio"], "noise_type": summary_data["noise_type"]},
                    findings=f"Found {summary_data['suspicious_sample_count']} suspicious labels ({summary_data['noise_ratio']:.1%} of data)"
                )
                lines.append(f"{'='*80}")
                lines.append("📋 COMPONENT SUMMARY")
                lines.append(f"{'='*80}")
                lines.append(component_summary)
                lines.append(f"{'='*80}\n")
            except Exception:
                pass

        return "\n".join(lines)

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

