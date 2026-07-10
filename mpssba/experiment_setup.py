from qfso.models.approximate.metrics import wht
import quimb.tensor as qtn
from typing import Optional
from mpssba.config import Config

from mpssba.generators import get_initial_mps, get_target_distribution


class ExperimentSetup:
    def __init__(self, config: Config, seed: Optional[int] = None):
        self.config = config

        dist_args = {"seed": seed if seed is not None else config.dist.seed}
        dist_args.update(config.dist.kwargs)

        self.p_target = get_target_distribution(
            method=config.dist.method, n_states=2**self.config.model.n_qubits, **dist_args
        )

        self.wht_exact = wht(self.p_target)

    def build_initial_mps(self) -> qtn.MatrixProductState:
        """Returns a fresh MPS state based on the selected strategy."""
        init_args = {}
        init_args.update(self.config.init.kwargs)
        return get_initial_mps(
            self.config.init.method,
            self.config.model.n_qubits,
            self.config.model.bond_dim,
            **init_args,
        )
