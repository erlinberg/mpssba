import os
import json
import pandas as pd
from mpssba.config import Config


class BatchedData:
    def __init__(self, dfs):
        self.dfs = dfs
        
    @property
    def mean(self):
        return pd.concat(self.dfs).groupby(level=0).mean()
        
    @property
    def std(self):
        if len(self.dfs) <= 1:
            return None
        return pd.concat(self.dfs).groupby(level=0).std()


class RunDataLoader:
    def __init__(self, run_dir):
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        self.run_dir = run_dir
        self.config = self._load_config()
        self.raw_dir = os.path.join(self.run_dir, "raw")
        self.processed_dir = os.path.join(self.run_dir, "processed")

    def _load_config(self):
        config_path = os.path.join(self.run_dir, "config.json")
        if not os.path.exists(config_path):
            return Config.from_dict({})
        with open(config_path, "r") as f:
            return Config.from_dict(json.load(f))
            
    def _aggregate(self, load_fn, *args, **kwargs):
        if self.config.batch.runs <= 1:
            return BatchedData([load_fn(postfix="", *args, **kwargs)])
        
        dfs = []
        for run_idx in range(self.config.batch.runs):
            postfix = self.config.batch.postfix.format(run_idx=run_idx)
            try:
                dfs.append(load_fn(postfix=postfix, *args, **kwargs))
            except FileNotFoundError:
                # Fallback for old data where no postfix was used
                dfs.append(load_fn(postfix="", *args, **kwargs))
        return BatchedData(dfs)

    def get_learning_curves(self):
        return self._aggregate(self._get_learning_curves)

    def _get_learning_curves(self, postfix=""):
        path = os.path.join(self.raw_dir, f"learning_curves{postfix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Learning curves not found at {path}")
        return pd.read_csv(path, index_col="Epoch")

    def get_masses_data(self):
        return self._aggregate(self._get_masses_data)

    def _get_masses_data(self, postfix=""):
        path = os.path.join(self.processed_dir, f"masses_data{postfix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Masses data not found at {path}")
        return pd.read_csv(path)
        
    def get_wht_history(self, mask_name):
        return self._aggregate(self._get_wht_history, mask_name)

    def _get_wht_history(self, mask_name, postfix=""):
        path = os.path.join(self.raw_dir, f"wht_history_{mask_name}{postfix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"WHT history for {mask_name} not found at {path}")
        
        df = pd.read_csv(path, index_col="Epoch")
        df.columns = df.columns.astype(int)
        return df

    def get_exact_wht(self):
        return self._aggregate(self._get_exact_wht)

    def _get_exact_wht(self, postfix=""):
        path = os.path.join(self.raw_dir, f"exact_wht{postfix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Exact WHT not found at {path}")
        df = pd.read_csv(path)
        return df.sort_values("WHT_Index")

    def get_cov_error_history(self, mask_name):
        return self._aggregate(self._get_cov_error_history, mask_name)

    def _get_cov_error_history(self, mask_name, postfix=""):
        path = os.path.join(self.processed_dir, f"cov_error_history_{mask_name}{postfix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Covariance error history for {mask_name} not found at {path}")
        
        df = pd.read_csv(path, index_col="Epoch")
        df.columns = df.columns.astype(int)
        return df

    def get_wht_deltas_history(self, mask_name):
        return self._aggregate(self._get_wht_deltas_history, mask_name)

    def _get_wht_deltas_history(self, mask_name, postfix=""):
        path = os.path.join(self.processed_dir, f"wht_deltas_{mask_name}{postfix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"WHT deltas history for {mask_name} not found at {path}")
        
        df = pd.read_csv(path, index_col="Epoch")
        return df
