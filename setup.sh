arch=gpu
conda env create --force -f $arch.yaml
eval "$(conda shell.bash hook)"
conda activate farm

conda install -c anaconda cudnn==8.2.1 --force

git clone https://github.com/deepmind/acme.git _acme
cd _acme
git checkout 6e1d71104371998e8cd0143cb8090c24263c50c4 3.0.0
pip install --editable .[jax,tf,testing,envs]
cd ..

pip install --upgrade jax[cuda]==0.2.26 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# EXPECTED ERRORS for jax>=0.2.26
# 1. rlax 0.1.1 requires <=0.2.21
# 2. distrax 0.1.0 requires jax<=0.2.21,

