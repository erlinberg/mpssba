import numpy.typing as npt
from math import comb
from typing import List, Dict
from mpssba.masks import calculate_correlation_length


def calculate_hamming_mass_distribution(wht_array: npt.ArrayLike, n_qubits: int) -> List[float]:
    masses = [0.0] * (n_qubits + 1)
    for idx, coeff in enumerate(wht_array):
        masses[idx.bit_count()] += abs(coeff) ** 2 / comb(n_qubits, idx.bit_count())
    return masses

def calculate_correlation_mass_distribution(wht_array: npt.ArrayLike, n_qubits: int) -> List[float]:
    masses = [0.0] * (n_qubits + 1)
    for idx, coeff in enumerate(wht_array):
        masses[calculate_correlation_length(idx)] += abs(coeff) ** 2 / comb(n_qubits, idx.bit_count())
    return masses

def calculate_covariance_error(wht_t: npt.ArrayLike, exact_wht: npt.ArrayLike, n_qubits: int) -> Dict[int, float]:
    exact_cov = {}
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            exact_cov[(i, j)] = exact_wht[2**i + 2**j] - exact_wht[2**i] * exact_wht[2**j]

    distances = list(range(1, n_qubits))
    err_at_t = {}
    for d in distances:
        err_sum = 0.0
        exact_sum = 0.0
        count = 0
        for i in range(n_qubits - d):
            j = i + d
            cov_t = wht_t[2**i + 2**j] - wht_t[2**i] * wht_t[2**j]
            err_sum += abs(cov_t - exact_cov[(i, j)])
            exact_sum += abs(exact_cov[(i, j)])
            count += 1
        
        if exact_sum > 1e-12:
            err_at_t[d + 1] = err_sum / exact_sum
        else:
            err_at_t[d + 1] = err_sum / count if count > 0 else 0.0
            
    return err_at_t

def calculate_wht_deltas(wht_t: npt.ArrayLike, exact_wht: npt.ArrayLike, n_qubits: int, mask_type: str) -> Dict[str, float]:
    data_masks = {}
    if mask_type == "hamming":
        for h in range(1, n_qubits + 1):
            data_masks[h] = [x for x in range(2**n_qubits) if x.bit_count() == h]
        key_prefix = "h"
    elif mask_type == "correlation":
        for crl in range(1, n_qubits + 1):
            data_masks[crl] = [x for x in range(2**n_qubits) if calculate_correlation_length(x) == crl]
        key_prefix = "crl"
    elif mask_type == "corr_2body" or mask_type == "corr_2body_lower":
        for crl in range(1, n_qubits + 1):
            data_masks[crl] = [x for x in range(2**n_qubits) if calculate_correlation_length(x) == crl and x.bit_count() <= 2]
        key_prefix = "crl"
    else:
        raise ValueError(f"Unknown masking type: {mask_type}")

    deltas_at_t = {}
    for val, mask_indices in data_masks.items():
        if not mask_indices:
            continue
        err_sum = sum((wht_t[idx] - exact_wht[idx]) ** 2 for idx in mask_indices)
        deltas_at_t[f"{key_prefix}={val}"] = err_sum
    return deltas_at_t
