import os
import pickle
import random
from typing import Optional
from mpssba.config import Config

import pandas as pd

from mpssba.masks import build_boolean_mask
from mpssba.trainer import MPSTrainer
from mpssba.experiment_setup import ExperimentSetup
from mpssba.post_processor import PostProcessor


class BatchRunner:
    def __init__(self, config: Config):
        self.config = config
        self.raw_data_dir = config.output.raw_dir

    def execute_single_run(self, postfix: str = "", seed: Optional[int] = None) -> None:
        n_qubits = self.config.model.n_qubits
        n_states = 2**n_qubits

        setup = ExperimentSetup(self.config, seed=seed)

        pd.DataFrame(
            {"WHT_Index": list(range(n_states)), "Exact": setup.wht_exact}
        ).to_csv(os.path.join(self.raw_data_dir, f"exact_wht{postfix}.csv"), index=False)

        data_mask, mask_name = build_boolean_mask(self.config, n_qubits)
        print(f"\n--- Starting Training for Mask: {mask_name} ---")
        trainer = MPSTrainer(setup, data_mask=data_mask, mask_name=mask_name)
        results = trainer.train()

        pkl_path = os.path.join(self.raw_data_dir, f"mps_history_{mask_name}{postfix}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "epochs": results["epochs_recorded"],
                    "snapshots": results["mps_history"],
                },
                f,
            )

        pd.DataFrame({mask_name: pd.Series(results["losses"])}).to_csv(
            os.path.join(self.raw_data_dir, f"learning_curves{postfix}.csv"), index_label="Epoch"
        )
        print(f"\nRun complete. Raw data saved to {self.raw_data_dir}")

    def execute_batch_runs(self) -> None:
        base_seed = getattr(self.config.dist, "seed", random.randint(0, 65535))
        for run_idx in range(self.config.batch.runs):
            run_seed = base_seed + run_idx
            
            print(f"\n=== Starting Run {run_idx + 1}/{self.config.batch.runs} on seed {run_seed} ===")

            postfix = self.config.batch.postfix.format(run_idx=run_idx)
            self.execute_single_run(postfix=postfix, seed=run_seed)

            print(f"\n=== Starting Post-Processing for Run {run_idx + 1}/{self.config.batch.runs} ===")
            processor = PostProcessor(self.config)
            processor.process(postfix=postfix)
