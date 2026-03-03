import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import html

import numpy as np
import pandas as pd
from collections import defaultdict

from report_components.base_component import ReportComponent, AnalysisContext
from utils.consts import (
    NUM_EXAMPLES_LLM, LOW_NEAR_DUPLICATE_RATIO, MEDIUM_NEAR_DUPLICATE_RATIO,
    NEAR_DUPLICATE_SIMILARITY_THRESHOLD, NEAR_DUPLICATE_NUM_PERM,
    NEAR_DUPLICATE_NUM_BANDS, NEAR_DUPLICATE_NGRAM_SIZE, NEAR_DUPLICATE_MAX_PAIRS,
    NEAR_DUPLICATE_MAX_ROWS_FULL, NEAR_DUPLICATE_TIMEOUT_SECONDS
)


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
    def __init__(self, num_perm: int = NEAR_DUPLICATE_NUM_PERM, seed: int = 42):
        self.num_perm = num_perm
        self.seed = seed

        np.random.seed(seed)
        self._mersenne_prime = (1 << 61) - 1
        self._max_hash = (1 << 32) - 1

        self._a = np.random.randint(1, self._mersenne_prime, size=num_perm, dtype=np.uint64)
        self._b = np.random.randint(0, self._mersenne_prime, size=num_perm, dtype=np.uint64)

    def _hash_token(self, token: str) -> int:
        return hash(token) & self._max_hash

    def compute_signature(self, tokens: Set[str]) -> np.ndarray:
        if not tokens:
            return np.full(self.num_perm, self._max_hash, dtype=np.uint64)

        token_hashes = np.array([self._hash_token(t) for t in tokens], dtype=np.uint64)

        permuted = (self._a[np.newaxis, :] * token_hashes[:, np.newaxis] + self._b[np.newaxis, :]) % self._mersenne_prime
        permuted = permuted & self._max_hash
        signature = permuted.min(axis=0)

        return signature

    @staticmethod
    def estimate_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


class LSHIndex:
    def __init__(self, num_perm: int = NEAR_DUPLICATE_NUM_PERM, num_bands: int = NEAR_DUPLICATE_NUM_BANDS):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        self._buckets: List[Dict[int, List[int]]] = [
            defaultdict(list) for _ in range(num_bands)
        ]
        self._signatures: Dict[int, np.ndarray] = {}

    def insert(self, idx: int, signature: np.ndarray):
        self._signatures[idx] = signature

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            bucket_key = hash(band.tobytes())
            self._buckets[band_idx][bucket_key].append(idx)

    def query_candidates(self, idx: int) -> Set[int]:
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

        candidates.discard(idx)
        return candidates


class NearDuplicateDetectionComponent(ReportComponent):
    def __init__(
        self,
        context: AnalysisContext,
        similarity_threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
        num_perm: int = NEAR_DUPLICATE_NUM_PERM,
        num_bands: int = NEAR_DUPLICATE_NUM_BANDS,
        use_embeddings: bool = False,
        text_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        column_weights: Optional[Dict[str, float]] = None,
        ngram_size: int = NEAR_DUPLICATE_NGRAM_SIZE,
        max_pairs_to_report: int = NEAR_DUPLICATE_MAX_PAIRS,
        cluster_similar_records: bool = True,
        random_state: int = 42,
        use_llm_explanations: bool = True
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
        df = self.context.dataset.df

        if df is None or df.empty:
            self.result = self._empty_result("Dataset is empty or not provided")
            return

        if len(df) < 2:
            self.result = self._empty_result("Dataset too small (need at least 2 rows)")
            return

        use_columns = [c for c in df.columns if c not in self.exclude_columns]

        if not use_columns:
            self.result = self._empty_result("No columns available after exclusions")
            return

        try:
            work_df = df
            sample_indices = None
            is_sampled = False

            if len(df) > NEAR_DUPLICATE_MAX_ROWS_FULL:
                sample_size = NEAR_DUPLICATE_MAX_ROWS_FULL
                rng = np.random.RandomState(self.random_state)
                sample_indices = rng.choice(len(df), size=sample_size, replace=False)
                sample_indices.sort()
                work_df = df.iloc[sample_indices].reset_index(drop=True)
                is_sampled = True

            self._minhasher = MinHasher(num_perm=self.num_perm, seed=self.random_state)
            self._lsh_index = LSHIndex(num_perm=self.num_perm, num_bands=self.num_bands)

            signatures = self._compute_signatures(work_df, use_columns)

            for idx, sig in enumerate(signatures):
                self._lsh_index.insert(idx, sig)

            pairs = self._find_near_duplicate_pairs(work_df, signatures, use_columns)

            if is_sampled and sample_indices is not None:
                for p in pairs:
                    p.index_a = int(sample_indices[p.index_a])
                    p.index_b = int(sample_indices[p.index_b])

            clusters = []
            if self.cluster_similar_records and pairs:
                clusters = self._cluster_near_duplicates(pairs, len(df))

            column_contribution = self._compute_column_contribution(pairs, use_columns)

            affected_indices = set()
            for pair in pairs:
                affected_indices.add(pair.index_a)
                affected_indices.add(pair.index_b)

            affected_rows = len(affected_indices)
            near_duplicate_ratio = affected_rows / len(df) if len(df) > 0 else 0.0

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
                    "total_pairs_found": len(pairs),
                    "sampled": is_sampled,
                    "sample_size": len(work_df) if is_sampled else len(df)
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

            self.context.shared_artifacts["near_duplicate_pairs"] = pairs
            self.context.shared_artifacts["near_duplicate_ratio"] = near_duplicate_ratio
            self.context.shared_artifacts["near_duplicate_affected_indices"] = affected_indices

        except Exception as e:
            self.result = self._empty_result(f"Analysis failed: {str(e)}")

    def _compute_signatures(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> List[np.ndarray]:
        str_df = df[columns].fillna('__NULL__').astype(str).apply(lambda x: x.str.strip().str.lower())
        col_indices = {col: i for i, col in enumerate(columns)}

        signatures = []
        for idx in range(len(df)):
            tokens = set()
            for col in columns:
                ci = col_indices[col]
                val = str_df.iat[idx, ci]
                if val == '__null__':
                    tokens.add(f"{col}:__NULL__")
                    continue
                if not val:
                    tokens.add(f"{col}:__EMPTY__")
                    continue
                padded = f"__{val}__"
                ng = self.ngram_size
                for i in range(len(padded) - ng + 1):
                    tokens.add(f"{col}:{padded[i:i + ng]}")
                if len(val) > 10:
                    for word in val.split():
                        if len(word) > 2:
                            tokens.add(f"{col}:word:{word}")
            signatures.append(self._minhasher.compute_signature(tokens))

        return signatures

    def _find_near_duplicate_pairs(
        self,
        df: pd.DataFrame,
        signatures: List[np.ndarray],
        columns: List[str]
    ) -> List[NearDuplicatePair]:
        pairs = []
        seen_pairs = set()
        start_time = time.time()
        pair_cap = self.max_pairs_to_report * 3

        for idx in range(len(df)):
            if time.time() - start_time > NEAR_DUPLICATE_TIMEOUT_SECONDS:
                break
            if len(pairs) >= pair_cap:
                break

            candidates = self._lsh_index.query_candidates(idx)

            for candidate_idx in candidates:
                pair_key = (min(idx, candidate_idx), max(idx, candidate_idx))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                minhash_sim = MinHasher.estimate_similarity(
                    signatures[idx], signatures[candidate_idx]
                )

                if minhash_sim < self.similarity_threshold * 0.8:
                    continue

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
        parent = list(range(n_rows))
        rank = [0] * n_rows

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        for pair in pairs:
            union(pair.index_a, pair.index_b)

        affected = set()
        for pair in pairs:
            affected.add(pair.index_a)
            affected.add(pair.index_b)

        cluster_members: Dict[int, List[int]] = defaultdict(list)
        for idx in affected:
            cluster_members[find(idx)].append(idx)

        cluster_sims: Dict[int, List[float]] = defaultdict(list)
        for pair in pairs:
            root = find(pair.index_a)
            cluster_sims[root].append(pair.similarity)

        clusters = []
        for root, members in cluster_members.items():
            if len(members) < 2:
                continue

            members = sorted(set(members))
            sims = cluster_sims.get(root, [])
            avg_sim = float(np.mean(sims)) if sims else 0.0

            clusters.append(NearDuplicateCluster(
                indices=members,
                centroid_index=members[0],
                avg_internal_similarity=round(avg_sim, 4),
                size=len(members)
            ))

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

    def _empty_result(self, reason: str = None) -> Dict[str, Any]:
        result = {
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
        if reason:
            result["summary"]["skipped_reason"] = reason
        return result

    def summarize(self) -> dict:
        if self.result is None:
            raise RuntimeError("analyze() must be called before summarize()")

        if "skipped_reason" in self.result.get("summary", {}):
            return {
                "skipped": True,
                "reason": self.result["summary"]["skipped_reason"],
                "near_duplicate_ratio": 0.0
            }

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

        return summary

    def get_full_summary(self) -> str:
        if self.result is None:
            return "No analysis performed."

        if "skipped_reason" in self.result.get("summary", {}):
            return f"⚠️ Analysis skipped: {self.result['summary']['skipped_reason']}"

        lines = []
        df = self.context.dataset.df

        if self.llm_explanations and self.result.get("pairs"):
            cards_html = []
            for expl in self.llm_explanations:
                pair_str = expl['pair']
                idx_a, idx_b = map(int, pair_str.split('-'))

                pair_info = next((p for p in self.result["pairs"]
                                 if p.get("index_a") == idx_a and p.get("index_b") == idx_b), {})
                similarity = pair_info.get("similarity", 0)
                matching = pair_info.get("matching_columns", [])
                differing = pair_info.get("differing_columns", [])

                row_a = df.iloc[idx_a].to_dict() if idx_a < len(df) else {}
                row_b = df.iloc[idx_b].to_dict() if idx_b < len(df) else {}

                def format_row(row_data, diff_cols):
                    rows = []
                    for col, val in list(row_data.items())[:8]:
                        col_lower = col.lower()
                        is_id = 'id' in col_lower or 'name' in col_lower
                        if is_id:
                            continue
                        highlight = col in diff_cols
                        style = " style='background:#fef2f2;'" if highlight else ""
                        escaped_col = html.escape(str(col))
                        escaped_val = html.escape(str(val)[:30]) if val is not None else "N/A"
                        rows.append(f"<tr{style}><td class='ln-key'>{escaped_col}</td><td class='ln-val'>{escaped_val}</td></tr>")
                    return ''.join(rows)

                table_a = format_row(row_a, differing)
                table_b = format_row(row_b, differing)
                escaped_explanation = html.escape(str(expl['explanation']))

                card = f"""<div class='ln-card nd-pair-card'>
                    <div class='ln-title'>Pair: Row {idx_a} ↔ Row {idx_b}</div>
                    <div class='ln-meta'>Similarity: {similarity:.1%} · Matching: {len(matching)} cols · Differing: {len(differing)} cols</div>
                    <div class='nd-row-comparison'>
                        <div class='nd-row-col'><div class='nd-row-label'>Row {idx_a}</div><table class='ln-table'>{table_a}</table></div>
                        <div class='nd-row-divider'>↔</div>
                        <div class='nd-row-col'><div class='nd-row-label'>Row {idx_b}</div><table class='ln-table'>{table_b}</table></div>
                    </div>
                    <div class='ln-exp'>{escaped_explanation}</div>
                </div>"""
                cards_html.append(card)

            grid_html = "<div class='ln-grid'>" + ''.join(cards_html) + "</div>"
            lines.append("🤖 LLM EXAMPLE EXPLANATIONS")
            lines.append(grid_html)

        if self.llm:
            try:
                summary_data = self.summarize()
                if summary_data.get("skipped"):
                    return f"⚠️ Analysis skipped: {summary_data.get('reason', 'Unknown')}"
                component_summary = self.llm.generate_component_summary(
                    component_name="Near Duplicate Detection",
                    metrics={
                        "near_duplicate_ratio": summary_data["near_duplicate_ratio"],
                        "pairs": summary_data["total_pairs"]
                    },
                    findings=f"Found {summary_data['total_pairs']} near-duplicate pairs ({summary_data['near_duplicate_ratio']:.1%} affected rows)",
                    total_rows=len(self.context.dataset.df),
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
            "MinHash LSH-based near-duplicate detection with efficient candidate generation. "
            "Identifies similar but non-identical records from data entry variations or ETL issues."
        )
