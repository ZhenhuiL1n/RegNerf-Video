import functools
import gc
import time
import sys

from absl import app
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints

sys.path.append('./')

from internal import configs, datasets, math, models, utils, vis  # pylint: disable=g-multiple-import
from dataset import Multicam_video
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from skimage.metrics import structural_similarity

configs.define_common_flags()
jax.config.parse_flags_with_absl()
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')


@flax.struct.dataclass
class TrainStats:
    """Collection of stats for logging."""
    loss: float
    losses: float
    losses_georeg: float
    disp_mses: float
    normal_maes: float
    weight_l2: float
    psnr: float
    psnrs: float
    grad_norm: float
    grad_abs_max: float
    grad_norm_clipped: float


def tree_sum(tree):
    """Sums all values in the tree."""
    return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)

def tree_norm(tree):
    """Computes the L2 norm of all values in the tree."""
    return jnp.sqrt(tree_sum(jax.tree_map(lambda x: jnp.sum(x**2), tree)))


# one optimization step

# test what we got from the dataset loader


def main(unused_argv):
      
    rng = random.PRNGKey(20200823)
    # Shift the numpy random seed by host_id() to shuffle data loaded by different
    # hosts.
    np.random.seed(20201473 + jax.host_id())


    config = configs.load_config()
    dataset = datasets.load_dataset('train', config.data_dir, config)
    test_dataset = datasets.load_dataset('test', config.data_dir, config)
    print("................... finish loading dataset ...................")


if __name__ == '__main__':
    app.run(main)