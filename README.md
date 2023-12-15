# JAX-PI


## Installation

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
cd ~/miniforge3/envs/jaxpi/lib
ln -sfn libnvrtc.so.11.8.89 libnvrtc.so
cd -
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
mamba install -y numpy scipy matplotlib
pip install --upgrade flax optax absl-py wandb ml_collections tabulate

pip install -e .
```

## Quickstart

Training
```
python main.py
```

Evaluation
```
python main.py --config.mode=eval
```

Config file
```
python main.py --config ./configs/default.py
```

Batch size
```
python main.py --config.training.batch_size_per_device=4096
```

Help
```
python main.py --help
```

Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1 python main.py
```


## Code structure
The code corresponding to each problem is entered in a folder in the examples directory (e.g., examples/laplace/). Here is an overview of the contents of each such file:

- configs
    - Folder containing all config files
- config file (e.g., configs/default.py)
    - Contains training-related configurations such as network architecture, batch size, iterations, and problem-specific variables
- models.py
    - Specifies the loss functions and core methods. Here is where most changes are done when adapting the code to a new PDE
- train.py
    - Specifies the training process
- eval.py
    - Evaluates the model and creates plots
- main.py
    - Runs either train.py or eval.py depending on mode


### configs
The config files contain a larger number of training and problem-specific configurations, variables, and settings. Here is an overview of some of the most frequently modified: 

- config.mode
    - `train` for training, `eval` for evaluation
- config.arch.num_layers
    - Depth of NN (Recommended setting: from 3 to 6) 
- config.arch.hidden_dim
    - Width of NN (Recommended setting: from 128 to 512)
- config.arch.activation
    - Activation function (Recommended setting: `tanh`, `sin`, `gelu`)
- config.arch.periodicity
    - Imposing periodic boundary conditions: `period` = $\frac{2\pi}{P}$ for a period $P$
- config.arch.fourier_emb
    - Random Fourier feature embeddings (Recommended setting: `embed_scale`=$\sigma \in [1, 10]$)
- config.arch.reparam
    - Random weight factorization (Recommended setting: $\mu$=0.5 or 1 and $\sigma$=0.1)
- config.training.max_steps
    - Number of iterations before stopping
- config.training.batch_size_per_device
    - Batch size when training (a smaller value than the default 4096 recommended)
- config.weighting.scheme
    - Loss Balancing. `grad_norm` for Grad Norm Weighting, `ntk` for NTK Weighting
- config.weighting.momentum
    - Recommended setting: $\alpha=0.9$
- config.weighting.update_every_steps
    - Recommended setting: $f=1000$
- config.weighting.use_causal
    - Use modified PDE residual loss to avoid violating causality. Should be None or False for non-time-dependent PDEs


### models.py
The file models.py contains the model describing the PDE and the related losses. The core functions are as follows: 

- u_net
    - Performs a forward pass of the neural network and outputs a model prediction $u(t,x)$. Here is where hard boundary conditions should be introduced, if any.
- r_net
    - Calculates the PDE residual for a given (t,x).
- losses
    - Computes the squared initial, boundary, residual, and observation loss (if applicable) across a sampled batch.
- res_and_w
    - Calculates weights for each sequential segment of the temporal domain if config.weighting.causal_tol = True (i.e., if using modified residual loss to avoid violating causality). Not applicable for non-time-dependent PDEs.


## References
- https://github.com/PredictiveIntelligenceLab/jaxpi
- https://github.com/felixagren97/jaxpi
