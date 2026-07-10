import os
import pandas as pd
import numpy as np
import pickle
from qfso.models.approximate.metrics import wht
from mpssba.masks import build_boolean_mask
from mpssba.config import Config
from mpssba.metrics import (
    calculate_hamming_mass_distribution, 
    calculate_correlation_mass_distribution,
    calculate_covariance_error,
    calculate_wht_deltas,
)

class PostProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.raw_dir = config.output.raw_dir
        self.processed_dir = config.output.processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        self.n_qubits = config.model.n_qubits
        self.n_states = 2 ** self.n_qubits

    def process(self, postfix: str = "") -> None:
        exact_df = pd.read_csv(os.path.join(self.raw_dir, f"exact_wht{postfix}.csv"))
        exact_wht = exact_df["Exact"].values
        exact_masses = calculate_hamming_mass_distribution(exact_wht, self.n_qubits) if self.config.training.masking.type == "hamming" else calculate_correlation_mass_distribution(exact_wht, self.n_qubits)
        exact_p = wht(exact_wht) / (self.n_states)

        masses_data = {"Exact": exact_masses}
        dists_data = {"State_Index": list(range(self.n_states)), "Exact": exact_p}

        _, mask_name = build_boolean_mask(self.config, self.config.model.n_qubits)
        pkl_file = os.path.join(self.raw_dir, f"mps_history_{mask_name}{postfix}.pkl")

        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"MPS history for mask {mask_name} not found at {pkl_file}. Please ensure the training has been completed and the raw data is available.")
            
        print(f"Compute WHT History for mask = {mask_name}...")
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        wht_snapshots = []
        for mps in data["snapshots"]:
            dense_state = mps.to_dense().flatten()
            probs = np.abs(dense_state) ** 2
            wht_vals = wht(probs)
            wht_snapshots.append(wht_vals)
            
        df_hist = pd.DataFrame(wht_snapshots)
        df_hist.index = data["epochs"]
        df_hist.index.name = "Epoch"
        
        hist_file = os.path.join(self.raw_dir, f"wht_history_{mask_name}{postfix}.csv")
        df_hist.to_csv(hist_file)
        
        final_wht = df_hist.iloc[-1].values
        
        masses_data[mask_name] = calculate_hamming_mass_distribution(final_wht, self.n_qubits) if self.config.training.masking.type == "hamming" else calculate_correlation_mass_distribution(final_wht, self.n_qubits)
        dists_data[mask_name] = wht(final_wht) / (self.n_states)

        pd.DataFrame(masses_data).to_csv(os.path.join(self.processed_dir, f"masses_data{postfix}.csv"), index=False)
        pd.DataFrame(dists_data).to_csv(os.path.join(self.processed_dir, f"distributions{postfix}.csv"), index=False)
        
        cov_error_history = [
            calculate_covariance_error(wht_t, exact_wht, self.n_qubits)
            for wht_t in wht_snapshots
        ]
        
        wht_deltas_history = [
            calculate_wht_deltas(wht_t, exact_wht, self.n_qubits, self.config.training.masking.type)
            for wht_t in wht_snapshots
        ]
            
        df_cov_err = pd.DataFrame(cov_error_history, index=data["epochs"])
        df_cov_err.index.name = "Epoch"
        cov_err_file = os.path.join(self.processed_dir, f"cov_error_history_{mask_name}{postfix}.csv")
        df_cov_err.to_csv(cov_err_file)

        df_deltas = pd.DataFrame(wht_deltas_history, index=data["epochs"])
        df_deltas.index.name = "Epoch"
        for col in df_deltas.columns:
            initial_error = df_deltas[col].iloc[0]
            if initial_error > 1e-12:
                df_deltas[col] = df_deltas[col] / initial_error
            else:
                df_deltas[col] = 0.0
                
        deltas_file = os.path.join(self.processed_dir, f"wht_deltas_{mask_name}{postfix}.csv")
        df_deltas.to_csv(deltas_file)

        print(f"Post-processing complete. Processed data saved to {self.processed_dir}")
