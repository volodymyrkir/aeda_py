import pandas as pd
import polars as pl

import os
from pathlib import Path
from typing import Optional, Literal

from utils.consts import POLARS_SIZE_THRESHOLD_MB

Engine = Literal["pandas", "polars"]


class Dataset:
    def __init__(
        self,
        path: str,
        engine: Optional[Engine] = None,
        force_engine: bool = False
    ):
        self.path = Path(path)
        self.file_size_bytes = os.path.getsize(path)
        self.file_size_mb = self.file_size_bytes / (1024 * 1024)

        self._recommended_engine = self._determine_recommended_engine()

        if force_engine and engine:
            self._engine = engine
        elif engine:
            self._engine = engine
        else:
            self._engine = self._recommended_engine

        self._df_pandas: Optional[pd.DataFrame] = None
        self._df_polars = None
        self._schema: Optional[dict] = None

        self._load_data()

    def _determine_recommended_engine(self) -> Engine:
        return "polars" if self.file_size_mb > POLARS_SIZE_THRESHOLD_MB else "pandas"

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def recommended_engine(self) -> Engine:
        return self._recommended_engine

    @property
    def df(self) -> pd.DataFrame:
        if self._df_pandas is None:
            if self._engine == "polars" and self._df_polars is not None:
                self._df_pandas = self._df_polars.to_pandas()
            else:
                self._load_as_pandas()
        return self._df_pandas

    @property
    def schema(self) -> dict:
        if self._schema is None:
            self._schema = self._infer_schema()
        return self._schema

    def _load_data(self):
        if self._engine == "polars":
            self._load_as_polars()
        else:
            self._load_as_pandas()

    def _load_as_pandas(self):
        ext = self.path.suffix.lower()

        if ext == ".csv":
            self._df_pandas = pd.read_csv(self.path)
        elif ext == ".parquet":
            self._df_pandas = pd.read_parquet(self.path)
        elif ext == ".json":
            self._df_pandas = pd.read_json(self.path)
        elif ext == ".orc":
            self._df_pandas = pd.read_orc(self.path)
        elif ext == ".xlsx":
            # Requires: openpyxl
            self._df_pandas = pd.read_excel(self.path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_as_polars(self):
        ext = self.path.suffix.lower()

        if ext == ".csv":
            self._df_polars = pl.read_csv(self.path)
        elif ext == ".parquet":
            self._df_polars = pl.read_parquet(self.path)
        elif ext == ".json":
            self._df_polars = pl.read_json(self.path)
        elif ext == ".orc":
            self._df_pandas = pd.read_orc(self.path)
            self._df_polars = pl.from_pandas(self._df_pandas)
            return
        elif ext == ".xlsx":
            # Polars Excel support is not guaranteed across versions,
            # so we load with pandas and convert.
            self._df_pandas = pd.read_excel(self.path, engine="openpyxl")
            self._df_polars = pl.from_pandas(self._df_pandas)
            return
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self._df_pandas = self._df_polars.to_pandas()

    def switch_engine(self, engine: Engine) -> "Dataset":
        if engine == self._engine:
            return self
        return Dataset(str(self.path), engine=engine, force_engine=True)

    def _infer_schema(self) -> dict:
        schema = {
            "numeric": [],
            "categorical": [],
            "datetime": []
        }

        df = self.df
        for col in df.columns:
            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                schema["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema["datetime"].append(col)
            else:
                schema["categorical"].append(col)

        return schema

    def get_info(self) -> dict:
        return {
            "path": str(self.path),
            "file_size_mb": round(self.file_size_mb, 2),
            "engine": self._engine,
            "recommended_engine": self._recommended_engine,
            "rows": len(self.df),
            "columns": len(self.df.columns)
        }

    @classmethod
    def from_csv(cls, path: str, engine: Optional[Engine] = None) -> "Dataset":
        return cls(path, engine=engine)

    @classmethod
    def from_parquet(cls, path: str, engine: Optional[Engine] = None) -> "Dataset":
        return cls(path, engine=engine)

    @classmethod
    def from_json(cls, path: str, engine: Optional[Engine] = None) -> "Dataset":
        return cls(path, engine=engine)

    @classmethod
    def from_orc(cls, path: str, engine: Optional[Engine] = None) -> "Dataset":
        return cls(path, engine=engine)

    @classmethod
    def from_xlsx(cls, path: str, engine: Optional[Engine] = None) -> "Dataset":
        return cls(path, engine=engine)
