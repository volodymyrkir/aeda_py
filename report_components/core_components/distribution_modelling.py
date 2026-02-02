import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from report_components.base_component import ReportComponent, AnalysisContext
from utils.consts import NUM_EXAMPLES_LLM


class Autoencoder(nn.Module):
    """
    Flexible autoencoder with configurable hidden layer factors.
    """

    def __init__(self, input_dim: int, hidden_factors: List[float] = [0.5, 0.25]):
        super().__init__()
        layers = [input_dim]
        for factor in hidden_factors:
            hidden_dim = max(4, int(layers[-1] * factor))  # Min 4 to avoid zero-dim
            layers.append(hidden_dim)

        encoder_layers = []
        for i in range(len(layers) - 1):
            encoder_layers.extend([nn.Linear(layers[i], layers[i + 1]), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(layers) - 2, -1, -1):  # Reverse for decoder
            decoder_layers.extend([nn.Linear(layers[i + 1], layers[i]), nn.ReLU()])
        self.decoder = nn.Sequential(*decoder_layers[:-1])  # No ReLU on output

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class DistributionModelingComponent(ReportComponent):
    """
    Learns joint feature distributions using a configurable autoencoder and
    detects distributional deviations via reconstruction error. Handles missing values,
    skips high-cardinality columns, and provides local explanations with AI narratives.
    """

    def __init__(
            self,
            context: AnalysisContext,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            hidden_factors: List[float] = [0.5, 0.25],
            early_stop_patience: int = 10,
            validation_split: float = 0.2,
            threshold_percentile: float = 95.0,
            impute_missing: bool = True,
            skip_high_cardinality: bool = True,
            cardinality_threshold: float = 0.85,
            min_unique_for_id_heuristic: int = 50,
            max_explain_rows: int = 5,
            max_explain_features: int = 5,
            use_llm_explanations: bool = True  # Enable LLM-powered explanations
    ):
        super().__init__(context, use_llm_explanations)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factors = hidden_factors
        self.early_stop_patience = early_stop_patience
        self.validation_split = validation_split
        self.threshold_percentile = threshold_percentile
        self.impute_missing = impute_missing
        self.skip_high_cardinality = skip_high_cardinality
        self.cardinality_threshold = cardinality_threshold
        self.min_unique_for_id_heuristic = min_unique_for_id_heuristic
        self.max_explain_rows = max_explain_rows
        self.max_explain_features = max_explain_features
        self.llm_explanations = []
        self.feature_means = {}

    def analyze(self):
        df = self.context.dataset.df
        if df is None or df.empty:
            self.result = self._empty_result("Dataset is empty or not provided")
            return

        candidate_numeric_cols = self.context.shared_artifacts.get(
            "numeric_columns",
            list(df.select_dtypes(include=[np.number]).columns)
        )
        if not candidate_numeric_cols:
            self.result = self._empty_result("No numeric columns available for distribution modeling")
            return

        n_rows = len(df)
        if n_rows < 10:
            self.result = self._empty_result("Dataset too small for autoencoder training (need at least 10 rows)")
            return

        # Skip high-cardinality columns
        numeric_cols = []
        skipped_cols = []
        for col in candidate_numeric_cols:
            non_null = df[col].dropna()
            if len(non_null) < 2:
                continue
            unique_count = non_null.nunique()
            unique_ratio = unique_count / len(non_null)
            if (
                    self.skip_high_cardinality and unique_ratio > self.cardinality_threshold and unique_count >= self.min_unique_for_id_heuristic):
                skipped_cols.append(
                    {"column": col, "unique_ratio": round(unique_ratio, 4), "unique_count": int(unique_count)})
                continue
            numeric_cols.append(col)

        if len(numeric_cols) < 2:
            self.result = self._empty_result("Insufficient suitable numeric columns for modeling (need at least 2)")
            return

        X = df[numeric_cols].copy()

        # Impute missing values
        if self.impute_missing:
            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)
        else:
            X = X.dropna()  # Fallback, but loses rows

        if len(X) < 10:
            self.result = self._empty_result("Too few complete rows after handling missing values")
            return

        # Store feature means for LLM explanations (before scaling)
        self.feature_means = {col: float(X[col].mean()) for col in numeric_cols}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Autoencoder(input_dim=X_tensor.shape[1], hidden_factors=self.hidden_factors).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch[0].to(device)
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(train_dataset)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch[0].to(device)
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    val_loss += loss.item() * len(batch)
            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break

        # Full inference
        model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(device)
            reconstructed = model(X_tensor)
            per_feature_errors = (X_tensor - reconstructed) ** 2  # Shape: (n_rows, n_features)
            reconstruction_error = torch.mean(per_feature_errors, dim=1).cpu().numpy()  # Row-wise mean
            per_feature_errors = per_feature_errors.cpu().numpy()  # For explanations

        reconstruction_error = np.nan_to_num(reconstruction_error, nan=0.0)  # Safety
        threshold = np.percentile(reconstruction_error, self.threshold_percentile)

        explanations = self._explain_high_errors(
            reconstruction_error, per_feature_errors, threshold, numeric_cols, self.feature_means
        )

        self.result = {
            "summary": {
                "mean_reconstruction_error": float(np.mean(reconstruction_error)),
                "threshold": float(threshold),
                "final_train_loss": float(train_losses[-1]),
                "skipped_columns": skipped_cols,
                "used_columns": numeric_cols
            },
            "row_level_errors": reconstruction_error.tolist(),
            "high_error_explanations": explanations
        }
        self.context.shared_artifacts["reconstruction_error"] = reconstruction_error

    def _explain_high_errors(
            self,
            errors: np.ndarray,
            per_feature_errors: np.ndarray,
            threshold: float,
            feature_names: List[str],
            feature_means: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        df = self.context.dataset.df
        high_error_indices = np.where(errors >= threshold)[0][:self.max_explain_rows]
        explanations = []
        for i, idx in enumerate(high_error_indices):
            feature_devs = per_feature_errors[idx]
            top_indices = np.argsort(feature_devs)[::-1][:self.max_explain_features]
            contributions = {feature_names[j]: round(float(feature_devs[j]), 4) for j in top_indices}
            narrative = self._generate_narrative(errors[idx], contributions)

            explanation_entry = {
                "row_index": int(idx),
                "error": round(float(errors[idx]), 5),
                "top_contributing_features": contributions,
                "narrative": narrative
            }

            if i < NUM_EXAMPLES_LLM and self.llm:
                try:
                    row_data = df.iloc[idx][feature_names].to_dict()
                    llm_explanation = self.llm.explain_distribution_anomaly(
                        column_name="multivariate_distribution",
                        detected_distribution="autoencoder_reconstruction",
                        anomaly_details={
                            "reconstruction_error": float(errors[idx]),
                            "threshold": float(threshold),
                            "contributing_features": contributions,
                            "row_data": row_data,
                            "feature_means": feature_means
                        }
                    )
                    explanation_entry["llm_explanation"] = llm_explanation
                    self.llm_explanations.append({
                        "row_index": int(idx),
                        "explanation": llm_explanation
                    })
                except Exception:
                    pass

            explanations.append(explanation_entry)
        return explanations

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "summary": {
                "mean_reconstruction_error": 0.0,
                "threshold": 0.0,
                "final_train_loss": 0.0,
                "skipped_columns": [],
                "used_columns": [],
                "skipped_reason": reason
            },
            "row_level_errors": [],
            "high_error_explanations": []
        }

    def _generate_narrative(
            self,
            error: float,
            contributions: Dict[str, float]
    ) -> str:
        top_features = list(contributions.keys())
        narrative = f"This row has a high reconstruction error of {error:.5f}, suggesting deviation from the learned distribution. "
        narrative += "Major contributors: " + ", ".join(
            [f"{feat} (error {contributions[feat]:.4f})" for feat in top_features])
        narrative += ". Possible causes include outliers, noise, or novel patterns in these features."
        return narrative

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        if "skipped_reason" in self.result["summary"]:
            return {
                "skipped": True,
                "reason": self.result["summary"]["skipped_reason"],
                "mean_reconstruction_error": 0.0,
                "high_error_ratio": 0.0
            }

        errors = np.array(self.result["row_level_errors"])

        example_explanations = []
        for expl in self.result["high_error_explanations"][:3]:
            clean_expl = {k: v for k, v in expl.items() if k != 'llm_explanation'}
            example_explanations.append(clean_expl)

        summary = {
            "mean_reconstruction_error": self.result["summary"]["mean_reconstruction_error"],
            "high_error_ratio": float(np.mean(errors >= self.result["summary"]["threshold"])) if len(errors) > 0 else 0.0,
            "example_explanations": example_explanations,
            "skipped_columns": self.result["summary"]["skipped_columns"]
        }

        return summary

    def get_full_summary(self) -> str:
        if self.result is None:
            raise RuntimeError("analyze() must be called before get_full_summary()")

        if "skipped_reason" in self.result["summary"]:
            return f"Analysis skipped: {self.result['summary']['skipped_reason']}"

        lines = []

        if self.llm_explanations:
            lines.append(f"\n{'='*80}")
            lines.append("🤖 LLM EXAMPLE EXPLANATIONS")
            lines.append(f"{'='*80}")
            for i, expl in enumerate(self.llm_explanations, 1):
                lines.append(f"\n{i}. Row {expl['row_index']}")
                lines.append(f"   {expl['explanation']}")
            lines.append("")

        if self.llm:
            try:
                summary_data = self.summarize()
                component_summary = self.llm.generate_component_summary(
                    component_name="Distribution Modeling",
                    metrics={"high_error_ratio": summary_data["high_error_ratio"]},
                    findings=f"Found {summary_data['high_error_ratio']:.1%} of data with high reconstruction error"
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
        return (
            "Autoencoder-based distribution modeling for detecting multivariate anomalies. "
            "Captures non-linear dependencies via reconstruction error."
        )
