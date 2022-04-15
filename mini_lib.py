import collections
from absl import app
from absl import flags
import acme
import bsuite
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

from acme import specs
from acme.agents.jax import r2d2
from acme.jax import networks as networks_lib
from acme.jax.networks import base
from acme.jax import utils
from acme import wrappers

import farm

from babyai.levels.iclr19_levels import Level_GoToRedBallGrey
from gym_minigrid.wrappers import RGBImgPartialObsWrapper


Images = jnp.ndarray
BabyAiObs = collections.namedtuple('BabyAiObs', ('image', ))

# ======================================================
# Network
# ======================================================
def convert_floats(inputs):
  return jax.tree_map(lambda x: x.astype(jnp.float32), inputs)

def make_farm_input(obs_embed, action, reward, num_actions):
  action = jax.nn.one_hot(
        action, num_classes=num_actions)  # [T?, B, A]

  # Map rewards -> [-1, 1].
  reward = jnp.tanh(reward)

  # Add dummy trailing dimensions to rewards if necessary.
  while reward.ndim < action.ndim:
    reward = jnp.expand_dims(reward, axis=-1)

  return farm.FarmInputs(
    image=obs_embed,
    vector=jnp.concatenate([action, reward], axis=-1)
    )

def flatten_farm_output(memory_out):
  return memory_out.reshape(*memory_out.shape[:-2], -1)

class AtariVisionTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self, flatten=True, extra_conv_dim = 16):
    super().__init__(name='atari_torso')
    layers = [
        hk.Conv2D(32, [8, 8], 4),
        jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2),
        jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1),
        jax.nn.relu,
    ]
    if extra_conv_dim:
      layers.append(hk.Conv2D(extra_conv_dim, [1, 1], 1))
    self._network = hk.Sequential(layers)
    self.flatten = flatten


  def __call__(self, inputs: Images) -> jnp.ndarray:
    inputs_rank = jnp.ndim(inputs)
    batched_inputs = inputs_rank == 4
    if inputs_rank < 3 or inputs_rank > 4:
      raise ValueError('Expected input BHWC or HWC. Got rank %d' % inputs_rank)


    outputs = self._network(inputs)
    if not self.flatten:
      return outputs

    if batched_inputs:
      flat = jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
    else:
      flat = jnp.reshape(outputs, [-1])  # [D]

    return flat


class SimpleFarmQNetwork(hk.RNNCore):
  """Simple Vanilla RNN Q Network.
  """
  
  def __init__(self, num_actions, module_size=128, nmodules=4):
    super().__init__(name='simple_r2d2_network')
    self._embed = AtariVisionTorso(flatten=False)
    self._core = farm.FARM(
      module_size=module_size,
      nmodules=nmodules)
    self._head = hk.nets.MLP([num_actions])
    self.num_actions = num_actions

  def __call__(
      self,
      inputs: jnp.ndarray,  # [B, ...]
      state: hk.LSTMState,  # [B, ...]
  ):
    inputs = convert_floats(inputs)
    image = inputs.observation.image / 255.0

    embeddings = self._embed(image)  # [B, D+A+1]

    farm_input = make_farm_input(embeddings, inputs.action, inputs.reward,
      self.num_actions)
    core_outputs, new_state = self._core(farm_input, state)
    core_outputs = flatten_farm_output(core_outputs)
    q_values = self._head(core_outputs)

    return q_values, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> hk.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: jnp.ndarray,  # [T, B, ...]
      state: hk.LSTMState,  # [T, ...]
  ):
    inputs = convert_floats(inputs)
    """Efficient unroll that applies torso, core, and duelling mlp in one pass."""
    image = inputs.observation.image / 255.0

    embeddings = hk.BatchApply(self._embed)(image)  # [T, B, D+A+1]

    farm_input = make_farm_input(embeddings, inputs.action, inputs.reward,
      self.num_actions)
    core_outputs, new_states = hk.static_unroll(self._core, farm_input, state)
    core_outputs = flatten_farm_output(core_outputs)
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

# ======================================================
# Environment
# ======================================================

class BabyAI(dm_env.Environment):
  """
  """

  def __init__(self, LvlCls=Level_GoToRedBallGrey):
    """Initializes BabyAI environment."""
    env = LvlCls()
    self.gym_env = RGBImgPartialObsWrapper(env, tile_size=8)
    self.env = wrappers.GymWrapper(self.gym_env)


  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    env_obs = self.gym_env.reset()
    image = env_obs['image']
    obs = BabyAiObs(image=image)
    timestep = dm_env.restart(obs)

    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    env_obs, reward, done, info = self.gym_env.step(action)
    image = env_obs['image']
    obs = BabyAiObs(image=image)
    reward = float(reward)

    if done:
      timestep = dm_env.termination(
        reward=reward, observation=obs)
    else:
      timestep = dm_env.transition(
        reward=reward, observation=obs)

    return timestep

  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return self.env.action_spec()

  def observation_spec(self):
    default = self.env.observation_spec() # dict_keys(['image'])
    default = BabyAiObs(**default)
    return default

def make_babyai_environment():
  env = BabyAI()
  wrapper_list = [
    wrappers.ObservationActionRewardWrapper,
    wrappers.SinglePrecisionWrapper,
  ]

  return wrappers.wrap_all(env, wrapper_list)


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
