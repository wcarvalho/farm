# FARM
Feature-Attending Recurrent Modules with an RL Agent based on the [ACME](https://github.com/deepmind/acme) codebase


# Install
```
bash setup.sh
```


# Run FARM w/ R2D2 on BSUITE
```
python train_distributed.py
```

Notes:
- this runs multiple actors
- actor/evaluator/learner all run in parallel thanks to ACME