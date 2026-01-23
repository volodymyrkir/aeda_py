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
            max_explain_features: int = 5
    ):
        super().__init__(context)
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

    def analyze(self):
        df = self.context.dataset.df
        if df is None or df.empty:
            raise ValueError("Dataset is empty or not provided")

        candidate_numeric_cols = self.context.shared_artifacts.get(
            "numeric_columns",
            list(df.select_dtypes(include=[np.number]).columns)
        )
        if not candidate_numeric_cols:
            raise ValueError("No numeric columns available for distribution modeling")

        n_rows = len(df)
        if n_rows < 10:  # Arbitrary min for training
            raise ValueError("Dataset too small for autoencoder training")

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
            raise ValueError("Insufficient suitable numeric columns for modeling")

        X = df[numeric_cols].copy()

        # Impute missing values
        if self.impute_missing:
            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=numeric_cols)
        else:
            X = X.dropna()  # Fallback, but loses rows

        if len(X) < 10:
            raise ValueError("Too few complete rows after handling missing values")

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
            reconstruction_error, per_feature_errors, threshold, numeric_cols
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
            feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        high_error_indices = np.where(errors >= threshold)[0][:self.max_explain_rows]
        explanations = []
        for idx in high_error_indices:
            feature_devs = per_feature_errors[idx]
            top_indices = np.argsort(feature_devs)[::-1][:self.max_explain_features]
            contributions = {feature_names[j]: round(float(feature_devs[j]), 4) for j in top_indices}
            narrative = self._generate_narrative(errors[idx], contributions)
            explanations.append({
                "row_index": int(idx),
                "error": round(float(errors[idx]), 5),
                "top_contributing_features": contributions,
                "narrative": narrative
            })
        return explanations

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
        errors = np.array(self.result["row_level_errors"])
        return {
            "mean_reconstruction_error": self.result["summary"]["mean_reconstruction_error"],
            "high_error_ratio": float(np.mean(errors >= self.result["summary"]["threshold"])),
            "example_explanations": self.result["high_error_explanations"][:3],
            "skipped_columns": self.result["summary"]["skipped_columns"]
        }

    def justify(self) -> str:
        return (
            "This component leverages a neural autoencoder to model the joint distribution of numeric features, "
            "capturing non-linear dependencies and enabling detection of distributional anomalies via reconstruction error. "
            "It enhances flexibility with configurable architecture, training parameters, and early stopping for robustness. "
            "Missing values are handled via imputation, high-cardinality columns are skipped to focus on meaningful features, "
            "and local explanations with AI-generated narratives provide interpretable insights. This method advances beyond "
            "traditional statistics, supporting thesis-level analysis of data manifolds and potential extensions to VAEs."
        )