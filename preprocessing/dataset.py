import pandas as pd

class Dataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.schema = self._infer_schema()

    @classmethod
    def from_csv(cls, path: str) -> "Dataset":
        df = pd.read_csv(path)
        return cls(df)

    @classmethod
    def from_parquet(cls, path: str) -> "Dataset":
        df = pd.read_parquet(path)
        return cls(df)

    def _infer_schema(self) -> dict:
        schema = {
            "numeric": [],
            "categorical": [],
            "datetime": []
        }

        for col in self.df.columns:
            dtype = self.df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                schema["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema["datetime"].append(col)
            else:
                schema["categorical"].append(col)

        return schema
