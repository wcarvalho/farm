# Feature-Attending Recurrent Modules (FARM)
Official codebase for:

[Feature-Attending Recurrent Modules for Generalization in Reinforcement Learning](https://arxiv.org/abs/2112.08369?context=cs.AI)<br>
Wilka Carvalho, Andrew Lampinen, Kyriacos Nikiforou, Felix Hill, Murray Shanahan<br>
https://arxiv.org/abs/2112.08369

<img src="images/architecture-intro.png" alt="FARM" style="zoom:40%;" />

**Abstract**: Many important tasks are defined in terms of object. To generalize across these tasks, a reinforcement learning (RL) agent needs to exploit the structure that the objects induce. Prior work has either hard-coded object-centric features, used complex object-centric generative models, or updated state using local spatial features. However, these approaches have had limited success in enabling general RL agents. Motivated by this, we introduce “Feature- Attending Recurrent Modules” (FARM), an architecture for learning state representations that relies on simple, broadly applicable inductive biases for capturing spatial and temporal regularities. FARM learns a state representation that is distributed across multiple modules that each attend to spatiotemporal features with an expressive feature attention mechanism. We show that this improves an RL agent’s ability to generalize across object-centric tasks. We study task suites in both 2D and 3D environments and find that FARM better generalizes compared to competing architectures that leverage attention or multiple modules.

Includes R2D2-based RL Agent based on the [ACME](https://github.com/deepmind/acme) codebase.

# Install
```
bash setup.sh
```
**Important note: Place conda env inside library path**
```
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/farm/lib/
```
Why? An ACME library will give a complaint similar to the following:
```
ImportError: libpython3.9.so.1.0: cannot open shared object file: No such file or directory
```

See Makefile for an example.

# Run FARM w/ R2D2 on BabyAI

## Synchronous
```
conda activate farm
make train_sync
```


## Asynchronous
```
conda activate farm
make train_async
```

Notes:
- this runs multiple actors
- actor/evaluator/learner all run in parallel thanks to ACME



## Cite

If you make use of this code in your own work, please cite our paper:

```
@article{carvalho2021feature,
  title={Feature-Attending Recurrent Modules for Generalizing Object-Centric Behavior},
  author={Carvalho, Wilka and Lampinen, Andrew and Nikiforou, Kyriacos and Hill, Felix and Shanahan, Murray},
  journal={arXiv preprint arXiv:2112.08369},
  year={2021}
}
```



## Todo
- [ ] test attention between modules during update
- [ ] info in readme about how to use with own environment
