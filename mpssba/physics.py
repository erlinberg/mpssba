import quimb as qu
import jax.numpy as jnp
import autoray as ar
import jax
import numpy.typing as npt

def precompute_pauli_z_indices(n_qubits: int, data_mask: npt.ArrayLike) -> list[tuple[int, ...]]:
    precomputed = []
    for x in range(2**n_qubits):
        if data_mask[x] > 0.0:
            z_indices = tuple(i for i in range(n_qubits) if (x >> (n_qubits - 1 - i)) & 1)
            precomputed.append(z_indices)
    return precomputed

def compute_wht_from_mps(mps: qu.MatrixProductState, precomputed_ops: list[tuple[int, ...]]) -> jnp.ndarray:
    norm_sq = ar.do("real", mps @ mps.H)

    expectations_list = []
        
    for z_indices in precomputed_ops:
        mps_z = mps.copy()
        for site in z_indices:
            mps_z.gate_(qu.pauli('Z'), site, contract=True, inplace=True)

        expectations_list.append(ar.do("real", mps.H @ mps_z))


    return ar.do("stack", expectations_list, like=norm_sq)

def normalize_mps_state(mps: qu.MatrixProductState) -> qu.MatrixProductState:
    nfact = abs(mps.H @ mps) ** 0.5
    return mps.multiply(1 / nfact, spread_over="all")

def calculate_wht_mmd_loss(mps: qu.MatrixProductState, src_wht_masked: jnp.ndarray, precomputed_ops: list[tuple[int, ...]]) -> jnp.ndarray:
    mps_wht = compute_wht_from_mps(mps, precomputed_ops)
    diff = jnp.squeeze(src_wht_masked) - jnp.squeeze(mps_wht)
    loss = jnp.sum((diff ** 2))
    return loss