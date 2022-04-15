"""Example running R2D2, on Atari."""

from absl import app
from absl import flags
import acme
import bsuite

import haiku as hk
import jax
import jax.numpy as jnp

from acme import specs
from acme.agents.jax import r2d2
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme import wrappers
import launchpad as lp
from launchpad.nodes.python.local_multi_processing import PythonProcess

from mini_lib import SimpleFarmQNetwork, make_networks, make_babyai_environment

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_actors', 4, 'Number of parallel actors.')
flags.DEFINE_bool('evaluator', True, 'evaluator')



def main(_):
  # Access flag value.
  environment_factory = (
      lambda seed: make_babyai_environment())

  config = r2d2.R2D2Config(
      batch_size=32,
      burn_in_length=0,
      trace_length=20,
      sequence_period=40,
      prefetch_size=0,
      samples_per_insert_tolerance_rate=0.1,
      samples_per_insert=0.0,
      num_parallel_calls=1,
      min_replay_size=1_000,
      max_replay_size=10_000,
    )

  def net_factory(env_spec: specs.EnvironmentSpec):
    num_actions = env_spec.actions.num_values
    return make_networks(
      batch_size=config.batch_size,
      env_spec=env_spec,
      NetworkCls=SimpleFarmQNetwork,
      NetKwargs=dict(num_actions=num_actions),
      )

  env = environment_factory(False)
  env_spec = acme.make_environment_spec(env)

  if FLAGS.evaluator:
    evaluator_factories = None
  else:
    evaluator_factories = []

  program = r2d2.DistributedR2D2FromConfig(
      seed=0,
      environment_factory=environment_factory,
      network_factory=net_factory,
      config=config,
      evaluator_factories=evaluator_factories,
      num_actors=FLAGS.num_actors,
      environment_spec=env_spec,
      workdir="./farm_results/",
  ).build()

  lp.launch(program,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal='current_terminal',
    local_resources = {
      'actor':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES='')),
      'evaluator':
          PythonProcess(env=dict(CUDA_VISIBLE_DEVICES=''))}
  )


if __name__ == '__main__':
  app.run(main)