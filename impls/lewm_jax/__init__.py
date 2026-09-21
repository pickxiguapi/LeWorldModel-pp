"""LeWM JAX using OGBench's native IMPALA-small encoder."""

from lewm_jax.checkpoints import checkpoint_image_shape, load_frozen_lewm
from lewm_jax.loss import lewm_loss, sigreg_loss
from lewm_jax.model import LeWM

ARCHITECTURE = 'lewm_impala_small'

__all__ = [
    'ARCHITECTURE',
    'LeWM',
    'checkpoint_image_shape',
    'lewm_loss',
    'load_frozen_lewm',
    'sigreg_loss',
]
