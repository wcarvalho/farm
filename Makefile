cuda?=0

export PYTHONPATH:=$(PYTHONPATH):.
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):$(HOME)/miniconda3/envs/farm/lib/

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true


train_synch:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue train_sync.py

train_asynch:
	CUDA_VISIBLE_DEVICES=$(cuda) \
	python -m ipdb -c continue train_async.py
