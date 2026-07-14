import numpy as np
from mpssba.config import Config

def calculate_correlation_length(state_idx: int) -> int:
    if state_idx == 0:
        return 0
    
    msb = state_idx.bit_length() - 1
    lsb = (state_idx & -state_idx).bit_length() - 1
    
    return msb - lsb + 1


def build_boolean_mask(config: Config, n_qubits: int) -> tuple[np.ndarray, str]:
    """
    mask_config: dict with 'type' and 'value', e.g., {'type': 'hw', 'value': 3}
    Returns: (data_mask_array, mask_name_string)
    """
    mask_type = config.training.masking.type
    val = config.training.masking.value

    n_states = 2 ** n_qubits
    
    if mask_type == "hamming":
        data_mask = np.array([1.0 if x.bit_count() <= val else 0.0 for x in range(n_states)])
        mask_name = f"HW_{val}"
        
    elif mask_type == "correlation":
        data_mask = np.array([1.0 if calculate_correlation_length(x) <= val else 0.0 for x in range(n_states)])
        mask_name = f"CORR_{val}"

    elif mask_type == "corr_2body":
        corr = val
        data_mask = np.array([
            1.0 if (calculate_correlation_length(x) <= corr and x.bit_count() <= 2) else 0.0 
            for x in range(n_states)
        ])
        mask_name = f"CORR_2BODY_{corr}"
    elif mask_type == "corr_2body_lower":
        corr = val
        # Include correlation length 1 since single qubit terms need to be included for 2-body correlations
        data_mask = np.array([
            1.0 if ((calculate_correlation_length(x) >= corr or calculate_correlation_length(x) == 1) and x.bit_count() <= 2) else 0.0 
            for x in range(n_states)
        ])
        mask_name = f"CORR_2BODY_LOWER_{corr}"
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
        
    return data_mask, mask_name