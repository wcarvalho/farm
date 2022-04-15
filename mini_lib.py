
from absl import app
from absl import flags
import acme
import bsuite
import haiku as hk
import jax
import jax.numpy as jnp

from acme import specs
from acme.agents.jax import r2d2
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme import wrappers

import farm

class SimpleFarmQNetwork(hk.RNNCore):
  """Simple Vanilla RNN Q Network.
  """
  
  def __init__(self, num_actions, module_size=128, nmodules=4):
    super().__init__(name='simple_r2d2_network')
    self._embed = hk.Sequential(
        [hk.Flatten(),
         hk.nets.MLP([50, 50])])
    self._core = farm.FARM(
      module_size=module_size,
      nmodules=nmodules)
    self._head = hk.nets.MLP([num_actions])

  def __call__(
      self,
      inputs: jnp.ndarray,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
  ):
    image = inputs.observation
    B = image.shape[0]
    image = image.reshape(B, -1) / 255.0

    embeddings = self._embed(image)  # [B, D+A+1]
    core_outputs, new_state = self._core(embeddings, state)
    q_values = self._head(core_outputs)
    return q_values, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: jnp.ndarray,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
  ):
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    image = inputs.observation
    T,B = image.shape[:2]
    image = image.reshape(T, B, -1) / 255.0

    embeddings = hk.BatchApply(self._embed)(image)  # [T, B, D+A+1]
    core_outputs, new_states = hk.static_unroll(self._core, embeddings, state)
    q_values = hk.BatchApply(self._head)(core_outputs)  # [T, B, A]
    return q_values, new_states

def make_networks(batch_size, env_spec, NetworkCls, NetKwargs):
  """Builds networks."""
  # ======================================================
  # Functions for use
  # ======================================================
  def forward_fn(x, s):
    model = NetworkCls(**NetKwargs)
    return model(x, s)

  def initial_state_fn(batch_size = None):
    model = NetworkCls(**NetKwargs)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state):
    model = NetworkCls(**NetKwargs)
    return model.unroll(inputs, state)

  # Make networks purely functional.
  forward_hk = hk.transform(forward_fn)
  initial_state_hk = hk.transform(initial_state_fn)
  unroll_hk = hk.transform(unroll_fn)

  # ======================================================
  # Define networks init functions.
  # ======================================================
  def initial_state_init_fn(rng, batch_size):
    return initial_state_hk.init(rng, batch_size)
  dummy_obs_batch = utils.tile_nested(
      utils.zeros_like(env_spec.observations), batch_size)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs_batch)

  def unroll_init_fn(rng, initial_state):
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  # Make FeedForwardNetworks.
  forward = networks_lib.FeedForwardNetwork(
      init=forward_hk.init, apply=forward_hk.apply)
  unroll = networks_lib.FeedForwardNetwork(
      init=unroll_init_fn, apply=unroll_hk.apply)
  initial_state = networks_lib.FeedForwardNetwork(
      init=initial_state_init_fn, apply=initial_state_hk.apply)

  # this conforms to both R2D2 & DQN APIs
  return r2d2.R2D2Networks(
      forward=forward,
      unroll=unroll,
      initial_state=initial_state)

def make_bsuite_environment():
  env = bsuite.load_and_record_to_csv(
      bsuite_id='deep_sea/0',
      results_dir='/tmp/bsuite',
      overwrite=True,
  )
  wrapper_list = [
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)
