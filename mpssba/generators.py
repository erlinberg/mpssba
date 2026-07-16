import quimb.tensor as qtn
import numpy as np
from jax import random
import pickle
import os
from qfso.distributions.generate import generate_distribution_with_target_entropy

def _generate_boltzmann_distribution(n_states: int, **kwargs) -> np.ndarray:
    seed = kwargs.get("seed", 42)
    target_entropy = kwargs.get("target_entropy", np.log(n_states) / 2.0)
    key = random.PRNGKey(seed)
    
    return np.asarray(
        generate_distribution_with_target_entropy(n_states, target_entropy, key)
    )

def _generate_uniform_distribution(n_states: int, **kwargs) -> np.ndarray:
    return np.ones(n_states) / n_states

DISTRIBUTION_REGISTRY = {
    "boltzmann": _generate_boltzmann_distribution,
    "uniform": _generate_uniform_distribution,
}

def get_target_distribution(method: str, n_states: int, **kwargs) -> np.ndarray:
    generator = DISTRIBUTION_REGISTRY.get(method)
    if not generator:
        raise ValueError(f"Unknown distribution method '{method}'. Available: {list(DISTRIBUTION_REGISTRY.keys())}")
    return generator(n_states, **kwargs)

def _create_random_factorized_mps(n_qubits: int, bond_dim: int, **kwargs) -> qtn.MatrixProductState:
    mps = qtn.MPS_rand_state(n_qubits, bond_dim=1, cyclic=False, normalize=True)
    mps.expand_bond_dimension_(bond_dim)

    return mps

def _create_random_entangled_mps(n_qubits: int, bond_dim: int, **kwargs) -> qtn.MatrixProductState:
    mps = qtn.MPS_rand_state(n_qubits, bond_dim=bond_dim, cyclic=False, normalize=True)

    return mps

def _create_uniform_superposition_mps(n_qubits: int, bond_dim: int, **kwargs) -> qtn.MatrixProductState:
    circ = qtn.CircuitMPS(n_qubits, max_bond=bond_dim, cutoff=1e-20)
    
    for i in range(n_qubits):
        circ.apply_gate("H", i)

    mps = circ.psi.copy()
    mps.expand_bond_dimension_(bond_dim)
    
    return mps

def _create_mps_from_pickle(file_path: str, **kwargs) -> qtn.MatrixProductState:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pickle file {file_path} not found.")
    
    with open(file_path, "rb") as f:
        # Load the last snapshot of the MPS from the pickle file
        mps = pickle.load(f)["snapshots"][-1]

    return mps

MPS_INIT_REGISTRY = {
    "rnd_full": _create_random_entangled_mps,
    "rnd_fact": _create_random_factorized_mps,
    "uniform": _create_uniform_superposition_mps,
    "from_pickle": _create_mps_from_pickle,
}

def get_initial_mps(method: str, **kwargs) -> qtn.MatrixProductState:
    initializer = MPS_INIT_REGISTRY.get(method)
    if not initializer:
        raise ValueError(f"Unknown MPS initialization '{method}'. Available: {list(MPS_INIT_REGISTRY.keys())}")
    return initializer(**kwargs)
