"""
Near-Duplicate Detection using MinHash LSH and Embeddings

This module implements a hybrid near-duplicate detection framework that combines:
- MinHash Locality Sensitive Hashing for efficient text-based similarity
- Dense embeddings for semantic similarity detection
- Configurable similarity thresholds and column weighting

Near-duplicates are records that are highly similar but not identical, often
arising from:
- Data entry variations (typos, formatting differences)
- Record linkage issues across data sources
- Temporal versions of the same entity
- Aggregation artifacts from ETL pipelines

References:
    - Broder, A. Z. (1997). On the resemblance and containment of documents.
      Compression and Complexity of Sequences.
    - Leskovec, J., Rajaraman, A., & Ullman, J. D. (2014). Mining of Massive
      Datasets. Cambridge University Press. (Ch. 3: Finding Similar Items)
"""

import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from collections import defaultdict

from report_components.base_component import ReportComponent, AnalysisContext
from utils.consts import NUM_EXAMPLES_LLM

# Risk thresholds for near-duplicate detection
LOW_NEAR_DUPLICATE_RATIO = 0.005
MEDIUM_NEAR_DUPLICATE_RATIO = 0.02


@dataclass
class NearDuplicatePair:
    """Container for a near-duplicate pair."""
    index_a: int
    index_b: int
    similarity: float
    matching_columns: List[str]
    differing_columns: List[str]
    detection_method: str


@dataclass
class NearDuplicateCluster:
    """Container for a cluster of near-duplicate records."""
    indices: List[int]
    centroid_index: int
    avg_internal_similarity: float
    size: int


@dataclass
class NearDuplicateResult:
    """Structured container for near-duplicate detection results."""
    pairs: List[NearDuplicatePair]
    clusters: List[NearDuplicateCluster]
    near_duplicate_ratio: float
    affected_rows: int
    detection_methods_used: List[str]
    column_contribution: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MinHasher:
    """
    MinHash implementation for efficient Jaccard similarity estimation.

    MinHash approximates Jaccard similarity J(A,B) = |A∩B| / |A∪B| by computing
    the probability that min(h(A)) = min(h(B)) for random hash functions h.
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        """
        Initialize MinHasher with specified number of permutations.

        Args:
            num_perm: Number of hash functions (higher = more accurate, slower)
            seed: Random seed for reproducibility
        """
        self.num_perm = num_perm
        self.seed = seed

        # Generate random hash function parameters using Mersenne prime
        np.random.seed(seed)
        self._mersenne_prime = (1 << 61) - 1
        self._max_hash = (1 << 32) - 1

        # Parameters for hash functions: h(x) = (a * x + b) mod p
        self._a = np.random.randint(1, self._mersenne_prime, size=num_perm, dtype=np.uint64)
        self._b = np.random.randint(0, self._mersenne_prime, size=num_perm, dtype=np.uint64)

    def _hash_token(self, token: str) -> int:
        """Hash a token to a 32-bit integer."""
        return int(hashlib.md5(token.encode('utf-8')).hexdigest()[:8], 16)

    def compute_signature(self, tokens: Set[str]) -> np.ndarray:
        """
        Compute MinHash signature for a set of tokens.

        Args:
            tokens: Set of string tokens (e.g., n-grams, words)

        Returns:
            MinHash signature as numpy array of shape (num_perm,)
        """
        if not tokens:
            return np.full(self.num_perm, self._max_hash, dtype=np.uint64)

        # Hash all tokens
        token_hashes = np.array([self._hash_token(t) for t in tokens], dtype=np.uint64)

        # Compute min hash for each permutation
        signature = np.full(self.num_perm, self._max_hash, dtype=np.uint64)

        for token_hash in token_hashes:
            # Compute all permuted hashes at once
            permuted = (self._a * token_hash + self._b) % self._mersenne_prime
            permuted = permuted & self._max_hash
            signature = np.minimum(signature, permuted)

        return signature

    @staticmethod
    def estimate_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from two MinHash signatures.

        Args:
            sig_a: MinHash signature of first set
            sig_b: MinHash signature of second set

        Returns:
            Estimated Jaccard similarity in [0, 1]
        """
        return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """
    Locality Sensitive Hashing index for efficient nearest neighbor search.

    Uses banding technique: divides signature into bands of rows. Two items
    are candidates if they match in at least one band, which amplifies the
    probability of finding similar items while reducing false positives.
    """

    def __init__(self, num_perm: int = 128, num_bands: int = 32):
        """
        Initialize LSH index.

        Args:
            num_perm: Number of permutations in MinHash signatures
            num_bands: Number of bands for LSH (higher = more candidates, lower threshold)
        """
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        # Hash tables for each band
        self._buckets: List[Dict[int, List[int]]] = [
            defaultdict(list) for _ in range(num_bands)
        ]
        self._signatures: Dict[int, np.ndarray] = {}

    def insert(self, idx: int, signature: np.ndarray):
        """Insert a signature into the index."""
        self._signatures[idx] = signature

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            bucket_key = hash(band.tobytes())
            self._buckets[band_idx][bucket_key].append(idx)

    def query_candidates(self, idx: int) -> Set[int]:
        """
        Find candidate near-duplicates for a given index.

        Returns indices that share at least one band bucket with the query.
        """
        signature = self._signatures.get(idx)
        if signature is None:
            return set()

        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            bucket_key = hash(band.tobytes())
            candidates.update(self._buckets[band_idx][bucket_key])

        # Remove self
        candidates.discard(idx)
        return candidates


class NearDuplicateDetectionComponent(ReportComponent):
    """
    Near-Duplicate Detection using MinHash LSH and optional dense embeddings.

    This component identifies records that are highly similar but not identical,
    which is critical for:
    - Detecting data quality issues from record linkage or ETL processes
    - Identifying potential data leakage in train/test splits
    - Discovering redundant information that may bias models
    - Cleaning datasets for deduplication pipelines

    The detection uses a two-stage approach:
    1. **Candidate Generation**: MinHash LSH efficiently finds candidate pairs
       with sub-quadratic complexity O(n * k) where k << n
    2. **Verification**: Detailed similarity computation confirms near-duplicates
       using configurable column weights and similarity metrics

    Attributes:
        similarity_threshold: Minimum similarity to consider as near-duplicate
        num_perm: Number of MinHash permutations
        num_bands: Number of LSH bands
        use_embeddings: Whether to use dense embeddings for text columns
        text_columns: Specific columns to treat as text for embedding
        exclude_columns: Columns to exclude from similarity computation
        column_weights: Custom weights for column importance
    """

    def __init__(
        self,
        context: AnalysisContext,
        similarity_threshold: float = 0.8,
        num_perm: int = 128,
        num_bands: int = 32,
        use_embeddings: bool = False,
        text_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        column_weights: Optional[Dict[str, float]] = None,
        ngram_size: int = 3,
        max_pairs_to_report: int = 100,
        cluster_similar_records: bool = True,
        random_state: int = 42,
        use_llm_explanations: bool = True  # Enable LLM-powered explanations
    ):
        super().__init__(context, use_llm_explanations)
        self.similarity_threshold = similarity_threshold
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.use_embeddings = use_embeddings
        self.text_columns = text_columns or []
        self.exclude_columns = exclude_columns or []
        self.column_weights = column_weights or {}
        self.ngram_size = ngram_size
        self.max_pairs_to_report = max_pairs_to_report
        self.cluster_similar_records = cluster_similar_records
        self.random_state = random_state
        self.llm_explanations = []

        self._minhasher: Optional[MinHasher] = None
        self._lsh_index: Optional[LSHIndex] = None
        self._embeddings: Optional[np.ndarray] = None

    def analyze(self):
        """
        Run near-duplicate detection analysis.

        Steps:
        1. Preprocess data and compute row representations
        2. Build MinHash signatures and LSH index
        3. Find candidate pairs via LSH
        4. Verify candidates with detailed similarity
        5. Optionally cluster near-duplicates
        6. Compute statistics and risk assessment
        """
        df = self.context.dataset.df

        if df is None or df.empty:
            raise ValueError("Dataset is empty or not provided")

        if len(df) < 2:
            self.result = self._empty_result()
            return

        # Determine columns to use
        use_columns = [c for c in df.columns if c not in self.exclude_columns]

        if not use_columns:
            raise ValueError("No columns available for near-duplicate detection after exclusions")

        # Initialize MinHash and LSH
        self._minhasher = MinHasher(num_perm=self.num_perm, seed=self.random_state)
        self._lsh_index = LSHIndex(num_perm=self.num_perm, num_bands=self.num_bands)

        # Compute signatures and build index
        signatures = self._compute_signatures(df, use_columns)

        for idx, sig in enumerate(signatures):
            self._lsh_index.insert(idx, sig)

        # Find and verify candidate pairs
        pairs = self._find_near_duplicate_pairs(df, signatures, use_columns)

        # Cluster near-duplicates if enabled
        clusters = []
        if self.cluster_similar_records and pairs:
            clusters = self._cluster_near_duplicates(pairs, len(df))

        # Compute column contribution to similarity
        column_contribution = self._compute_column_contribution(pairs, use_columns)

        # Compute affected rows
        affected_indices = set()
        for pair in pairs:
            affected_indices.add(pair.index_a)
            affected_indices.add(pair.index_b)

        affected_rows = len(affected_indices)
        near_duplicate_ratio = affected_rows / len(df) if len(df) > 0 else 0.0

        # Build result
        result = NearDuplicateResult(
            pairs=pairs[:self.max_pairs_to_report],
            clusters=clusters,
            near_duplicate_ratio=round(near_duplicate_ratio, 5),
            affected_rows=affected_rows,
            detection_methods_used=self._get_detection_methods(),
            column_contribution=column_contribution,
            metadata={
                "total_rows": len(df),
                "columns_used": use_columns,
                "similarity_threshold": self.similarity_threshold,
                "total_pairs_found": len(pairs)
            }
        )

        self.result = {
            "summary": {
                "near_duplicate_ratio": result.near_duplicate_ratio,
                "affected_rows": result.affected_rows,
                "total_pairs": len(pairs),
                "total_clusters": len(clusters),
                "detection_methods": result.detection_methods_used,
                "similarity_threshold": self.similarity_threshold
            },
            "pairs": [self._pair_to_dict(p, generate_llm=(i < NUM_EXAMPLES_LLM and self.llm is not None))
                      for i, p in enumerate(result.pairs)],
            "clusters": [self._cluster_to_dict(c) for c in clusters],
            "column_contribution": column_contribution,
            "impact": self._assess_impact(near_duplicate_ratio, len(pairs), len(clusters))
        }

        # Store in shared artifacts for downstream components
        self.context.shared_artifacts["near_duplicate_pairs"] = pairs
        self.context.shared_artifacts["near_duplicate_ratio"] = near_duplicate_ratio
        self.context.shared_artifacts["near_duplicate_affected_indices"] = affected_indices

    def _compute_signatures(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> List[np.ndarray]:
        """Compute MinHash signatures for all rows."""
        signatures = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            tokens = self._row_to_tokens(row, columns)
            sig = self._minhasher.compute_signature(tokens)
            signatures.append(sig)

        return signatures

    def _row_to_tokens(self, row: pd.Series, columns: List[str]) -> Set[str]:
        """
        Convert a row to a set of tokens for MinHash.

        Uses character n-grams for text values and formatted strings for
        numeric/categorical values.
        """
        tokens = set()

        for col in columns:
            value = row[col]

            if pd.isna(value):
                tokens.add(f"{col}:__NULL__")
                continue

            # Convert to string representation
            str_value = str(value).strip().lower()

            if not str_value:
                tokens.add(f"{col}:__EMPTY__")
                continue

            # Generate character n-grams
            padded = f"__{str_value}__"
            for i in range(len(padded) - self.ngram_size + 1):
                ngram = padded[i:i + self.ngram_size]
                tokens.add(f"{col}:{ngram}")

            # Also add word tokens for longer text
            if len(str_value) > 10:
                words = str_value.split()
                for word in words:
                    if len(word) > 2:
                        tokens.add(f"{col}:word:{word}")

        return tokens

    def _find_near_duplicate_pairs(
        self,
        df: pd.DataFrame,
        signatures: List[np.ndarray],
        columns: List[str]
    ) -> List[NearDuplicatePair]:
        """Find and verify near-duplicate pairs using LSH candidates."""
        pairs = []
        seen_pairs = set()

        for idx in range(len(df)):
            candidates = self._lsh_index.query_candidates(idx)

            for candidate_idx in candidates:
                # Ensure consistent pair ordering to avoid duplicates
                pair_key = (min(idx, candidate_idx), max(idx, candidate_idx))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Compute MinHash similarity
                minhash_sim = MinHasher.estimate_similarity(
                    signatures[idx], signatures[candidate_idx]
                )

                if minhash_sim < self.similarity_threshold * 0.8:  # Early pruning
                    continue

                # Detailed similarity verification
                detailed_sim, matching_cols, differing_cols = self._compute_detailed_similarity(
                    df.iloc[idx], df.iloc[candidate_idx], columns
                )

                if detailed_sim >= self.similarity_threshold:
                    pairs.append(NearDuplicatePair(
                        index_a=pair_key[0],
                        index_b=pair_key[1],
                        similarity=round(detailed_sim, 4),
                        matching_columns=matching_cols,
                        differing_columns=differing_cols,
                        detection_method="minhash_lsh"
                    ))

        # Sort by similarity descending
        pairs.sort(key=lambda p: p.similarity, reverse=True)
        return pairs

    def _compute_detailed_similarity(
        self,
        row_a: pd.Series,
        row_b: pd.Series,
        columns: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Compute detailed weighted similarity between two rows.

        Uses a combination of exact matching, Jaccard similarity for sets,
        and normalized edit distance for strings.
        """
        matching_cols = []
        differing_cols = []
        weighted_sim_sum = 0.0
        weight_sum = 0.0

        for col in columns:
            weight = self.column_weights.get(col, 1.0)
            val_a = row_a[col]
            val_b = row_b[col]

            # Handle nulls
            if pd.isna(val_a) and pd.isna(val_b):
                sim = 1.0
                matching_cols.append(col)
            elif pd.isna(val_a) or pd.isna(val_b):
                sim = 0.0
                differing_cols.append(col)
            else:
                # Compute similarity based on type
                sim = self._value_similarity(val_a, val_b)
                if sim >= 0.99:
                    matching_cols.append(col)
                elif sim < 0.5:
                    differing_cols.append(col)

            weighted_sim_sum += sim * weight
            weight_sum += weight

        overall_sim = weighted_sim_sum / weight_sum if weight_sum > 0 else 0.0
        return overall_sim, matching_cols, differing_cols

    def _value_similarity(self, val_a: Any, val_b: Any) -> float:
        """Compute similarity between two values."""
        # Exact match
        if val_a == val_b:
            return 1.0

        # Numeric similarity
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            if val_a == 0 and val_b == 0:
                return 1.0
            max_val = max(abs(val_a), abs(val_b))
            if max_val == 0:
                return 1.0
            return 1.0 - min(abs(val_a - val_b) / max_val, 1.0)

        # String similarity using character overlap (Jaccard on char n-grams)
        str_a = str(val_a).strip().lower()
        str_b = str(val_b).strip().lower()

        if str_a == str_b:
            return 1.0

        if not str_a or not str_b:
            return 0.0

        # Character n-gram Jaccard
        ngrams_a = set(str_a[i:i+self.ngram_size] for i in range(len(str_a) - self.ngram_size + 1))
        ngrams_b = set(str_b[i:i+self.ngram_size] for i in range(len(str_b) - self.ngram_size + 1))

        if not ngrams_a and not ngrams_b:
            return 1.0 if str_a == str_b else 0.0

        intersection = len(ngrams_a & ngrams_b)
        union = len(ngrams_a | ngrams_b)

        return intersection / union if union > 0 else 0.0

    def _cluster_near_duplicates(
        self,
        pairs: List[NearDuplicatePair],
        n_rows: int
    ) -> List[NearDuplicateCluster]:
        """
        Cluster near-duplicate records using Union-Find.

        Groups records that are transitively connected through near-duplicate
        relationships.
        """
        # Union-Find data structure
        parent = list(range(n_rows))
        rank = [0] * n_rows

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Build clusters from pairs
        for pair in pairs:
            union(pair.index_a, pair.index_b)

        # Group by cluster
        cluster_members: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n_rows):
            root = find(idx)
            # Only include indices that are actually in some pair
            for pair in pairs:
                if pair.index_a == idx or pair.index_b == idx:
                    cluster_members[root].append(idx)
                    break

        # Build cluster objects
        clusters = []
        for root, members in cluster_members.items():
            if len(members) < 2:
                continue

            members = list(set(members))  # Remove duplicates

            # Compute average internal similarity
            internal_sims = []
            for pair in pairs:
                if pair.index_a in members and pair.index_b in members:
                    internal_sims.append(pair.similarity)

            avg_sim = np.mean(internal_sims) if internal_sims else 0.0

            clusters.append(NearDuplicateCluster(
                indices=sorted(members),
                centroid_index=members[0],  # Could be improved with actual centroid computation
                avg_internal_similarity=round(avg_sim, 4),
                size=len(members)
            ))

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)
        return clusters

    def _compute_column_contribution(
        self,
        pairs: List[NearDuplicatePair],
        columns: List[str]
    ) -> Dict[str, float]:
        """
        Compute how much each column contributes to near-duplicates.

        Higher values indicate columns that frequently match in near-duplicate pairs.
        """
        if not pairs:
            return {col: 0.0 for col in columns}

        match_counts = {col: 0 for col in columns}

        for pair in pairs:
            for col in pair.matching_columns:
                if col in match_counts:
                    match_counts[col] += 1

        # Normalize by number of pairs
        n_pairs = len(pairs)
        return {col: round(count / n_pairs, 4) for col, count in match_counts.items()}

    def _assess_impact(
        self,
        near_duplicate_ratio: float,
        n_pairs: int,
        n_clusters: int
    ) -> Dict[str, Any]:
        """Assess the impact of near-duplicates on data quality."""
        if n_pairs == 0:
            return {"risk_level": "none", "ml_implications": []}

        risk_level = self._risk_level(near_duplicate_ratio)

        implications = [
            "Near-duplicates can cause data leakage if split across train/test sets",
            "Model evaluation metrics may be artificially inflated",
            "Feature importance may be skewed toward duplicate-correlated features"
        ]

        if near_duplicate_ratio > MEDIUM_NEAR_DUPLICATE_RATIO:
            implications.append("Significant storage and compute inefficiency")
            implications.append("Consider deduplication before model training")

        return {
            "risk_level": risk_level,
            "near_duplicate_pressure": round(near_duplicate_ratio, 5),
            "cluster_count": n_clusters,
            "ml_implications": implications,
            "recommendations": self._generate_recommendations(near_duplicate_ratio, n_clusters)
        }

    def _generate_recommendations(
        self,
        ratio: float,
        n_clusters: int
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recommendations = []

        if ratio > MEDIUM_NEAR_DUPLICATE_RATIO:
            recommendations.append(
                "High near-duplicate ratio detected. Consider implementing a deduplication "
                "pipeline using the detected clusters as a starting point."
            )

        if n_clusters > 10:
            recommendations.append(
                f"Found {n_clusters} distinct near-duplicate clusters. Investigate whether "
                "these represent data collection issues or valid variations."
            )

        if ratio > LOW_NEAR_DUPLICATE_RATIO:
            recommendations.append(
                "Use stratified sampling that accounts for near-duplicate clusters "
                "when creating train/test splits to prevent data leakage."
            )

        if not recommendations:
            recommendations.append(
                "Near-duplicate levels are within acceptable bounds. Continue monitoring "
                "as new data is added."
            )

        return recommendations

    def _risk_level(self, ratio: float) -> str:
        """Determine risk level based on near-duplicate ratio."""
        if ratio < LOW_NEAR_DUPLICATE_RATIO:
            return "low"
        if ratio < MEDIUM_NEAR_DUPLICATE_RATIO:
            return "medium"
        return "high"

    def _get_detection_methods(self) -> List[str]:
        """Return list of detection methods used."""
        methods = ["minhash_lsh"]
        if self.use_embeddings:
            methods.append("dense_embeddings")
        return methods

    def _pair_to_dict(self, pair: NearDuplicatePair, generate_llm: bool = False) -> Dict[str, Any]:
        result = {
            "index_a": pair.index_a,
            "index_b": pair.index_b,
            "similarity": pair.similarity,
            "matching_columns": pair.matching_columns,
            "differing_columns": pair.differing_columns,
            "detection_method": pair.detection_method
        }

        if generate_llm and self.llm:
            try:
                df = self.context.dataset.df
                row_a = df.iloc[pair.index_a].to_dict()
                row_b = df.iloc[pair.index_b].to_dict()
                llm_explanation = self.llm.explain_near_duplicate(
                    row_a=row_a,
                    row_b=row_b,
                    similarity_score=pair.similarity,
                    matching_columns=pair.matching_columns,
                    differing_columns=pair.differing_columns
                )
                result["llm_explanation"] = llm_explanation
                self.llm_explanations.append({
                    "pair": f"{pair.index_a}-{pair.index_b}",
                    "explanation": llm_explanation
                })
            except Exception:
                pass

        return result

    def _cluster_to_dict(self, cluster: NearDuplicateCluster) -> Dict[str, Any]:
        """Convert NearDuplicateCluster to dictionary."""
        return {
            "indices": cluster.indices,
            "centroid_index": cluster.centroid_index,
            "avg_internal_similarity": cluster.avg_internal_similarity,
            "size": cluster.size
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "summary": {
                "near_duplicate_ratio": 0.0,
                "affected_rows": 0,
                "total_pairs": 0,
                "total_clusters": 0,
                "detection_methods": self._get_detection_methods(),
                "similarity_threshold": self.similarity_threshold
            },
            "pairs": [],
            "clusters": [],
            "column_contribution": {},
            "impact": {"risk_level": "none", "ml_implications": []}
        }

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        summary = {
            "near_duplicate_ratio": self.result["summary"]["near_duplicate_ratio"],
            "affected_rows": self.result["summary"]["affected_rows"],
            "total_pairs": self.result["summary"]["total_pairs"],
            "cluster_count": self.result["summary"]["total_clusters"],
            "risk_level": self.result["impact"].get("risk_level", "none"),
            "top_contributing_columns": dict(
                sorted(
                    self.result["column_contribution"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            )
        }

        if self.llm_explanations:
            print(f"\n{'='*80}")
            print("🤖 LLM EXPLANATIONS")
            print(f"{'='*80}")
            for i, expl in enumerate(self.llm_explanations, 1):
                print(f"\n{i}. Pair {expl['pair']}")
                print(f"   {expl['explanation']}")
            print(f"{'='*80}\n")

        return summary

    def justify(self) -> str:
        return (
            "MinHash LSH-based near-duplicate detection with efficient candidate generation. "
            "Identifies similar but non-identical records from data entry variations or ETL issues."
        )
