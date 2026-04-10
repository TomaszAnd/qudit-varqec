# Abstract ququart pipeline (backward compat)
from src.legacy.ququart_pipeline import single_qudit_paulis, single_qudit_dephasing_paulis
from src.legacy.ququart_pipeline import build_error_sets, build_dephasing_error_sets
from src.legacy.ququart_pipeline import G_theta_unitary, ent_layer, single_qudit_layer
from src.legacy.ququart_pipeline import create_encoder

# Native trapped-ion pipeline
from src.gates import XY_gate, Z_gate, MS_gate, CSUM_gate, CSUB_gate, light_shift_gate
from src.errors import (
    qudit_hardware_error_basis, build_native_error_set,
    hardware_error_basis, build_qutrit_error_set,
    make_hardware_noise_fn, build_native_error_set_factored,
    ErrorModel,
)
from src.encoder import create_native_encoder

# Loss functions
from src.loss import (
    kl_loss_fast, kl_loss_detection_minibatch, save_varqec_result, load_varqec_result,
)
