import quimb.tensor as qtn
import numpy as np
from mpssba.physics import calculate_wht_mmd_loss, normalize_mps_state, precompute_pauli_z_indices
import jax
from typing import Any, Dict, Optional
from mpssba.experiment_setup import ExperimentSetup

_current_target = None

def fetch_target_array_for_jax() -> np.ndarray:
    """
    This global variable and function exist to bypass JAX's compilation tracing.
    For large qubit counts, compiling the loss function takes 5-6 minutes.
    By using jax.pure_callback to fetch this global variable during execution,
    we can tunnel new target arrays into the JIT-compiled graph without 
    triggering a massive recompile for every batch run since the shape
    and dtype of the target array remain constant across runs.
    """
    return _current_target

class MPSTrainer:
    _cached_optimizer: Optional[qtn.TNOptimizer] = None

    def __init__(self, setup: ExperimentSetup, data_mask: np.ndarray, mask_name: str = "UnknownMask"):
        self.setup = setup
        self.data_mask = data_mask
        self.mask_name = mask_name
        self.n_qubits = setup.config.model.n_qubits
        
        self.history = {
            "mps_snapshots": [],
            "epochs": []
        }

    def _callback(self, tnopt: qtn.TNOptimizer) -> None:
        """Captures intermediate state per epoch for later analysis."""
        if tnopt.nevals % self.setup.config.training.record_every == 0:
            mps_current = tnopt.get_tn_opt()
            self.history["mps_snapshots"].append(mps_current.copy())
            self.history["epochs"].append(tnopt.nevals)

    def train(self) -> Dict[str, Any]:
        global _current_target
        _current_target = self.setup.wht_exact[self.data_mask > 0.0]
        
        mps_init = self.setup.build_initial_mps()
        
        if MPSTrainer._cached_optimizer is None:
            train_precomputed_ops = precompute_pauli_z_indices(self.n_qubits, self.data_mask)
            target_shape = jax.ShapeDtypeStruct(_current_target.shape, _current_target.dtype)
            
            def _loss(mps):
                src_wht_masked = jax.pure_callback(fetch_target_array_for_jax, target_shape)
                return calculate_wht_mmd_loss(mps, src_wht_masked, train_precomputed_ops)
            
            MPSTrainer._cached_optimizer = qtn.TNOptimizer(
                mps_init,
                loss_fn=_loss,
                norm_fn=normalize_mps_state,
                optimizer="sgd",
                autodiff_backend="jax",
                callback=self._callback,
                jit_fn=True
            )
        else:
            MPSTrainer._cached_optimizer.reset(tn=mps_init)
            MPSTrainer._cached_optimizer.callback = self._callback
            
        print(f"Starting MPS optimization for {self.setup.config.training.epochs} epochs...")
        mps_opt = MPSTrainer._cached_optimizer.optimize(self.setup.config.training.epochs, self.setup.config.training.tolerance)
        
        return {
            "mps_opt": mps_opt,
            "losses": MPSTrainer._cached_optimizer.losses,
            "mps_history": self.history["mps_snapshots"],
            "epochs_recorded": self.history["epochs"]
        }
