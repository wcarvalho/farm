arch=gpu
conda env create --force -f $arch.yaml
eval "$(conda shell.bash hook)"
conda activate farm

conda install -c anaconda cudnn==8.2.1 --force

# pip install git+https://github.com/deepmind/acme.git@0.4.0
git clone https://github.com/deepmind/acme.git _acme
cd _acme
git checkout e7e99762369c2ab2871d1c4bc6b6ab776eddf48c # 4.0.0
pip install --editable .[jax,tf,testing,envs]
cd ..


git clone https://github.com/mila-iqia/babyai.git _babyai
cd _babyai
pip install --editable .
cd ..


pip install --upgrade jax[cuda]==0.2.27 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# EXPECTED ERRORS for jax>=0.2.27
# 1. rlax 0.1.1 requires <=0.2.21
# 2. distrax 0.1.0 requires jax<=0.2.21,

