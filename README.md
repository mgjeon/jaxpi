# JAX-PI

```
git clone https://github.com/mgjeon/jaxpi
cd jaxpi
git remote add upstream https://github.com/PredictiveIntelligenceLab/jaxpi
git checkout learn
git fetch upstream
git merge upstream/main
git push origin learn
```

```
mamba create -n jaxpi
mamba activate jaxpi
mamba update -y mamba
mamba install -y python=3.10 ipykernel
pip install --upgrade pip
mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd ~/miniforge3/envs/pidon/lib
ln -sfn libnvrtc.so.11.8.89 libnvrtc.so
cd -
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
mamba install -y numpy scipy matplotlib
pip install flax optax absl-py wandb ml_collections

pip install -e .
```