import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Get dataset
    T = 1.0  # final time
    L = 2 * jnp.pi  # length of the domain
    c = 80  # advection speed
    n_t = 200  # number of time steps
    n_x = 128  # number of spatial points

    # Get  dataset
    u_ref, t_star, x_star = get_dataset(T, L, c, n_t, n_x)
    u0 = u_ref[0, :]

    # Restore model
    model = models.Advection(config, u0, t_star, x_star, c)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")

    # Plot results
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, jnp.abs(u_ref - u_pred), cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Absolute error/ L2err: {l2_error:.3e}")
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "adv.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    from jaxpi.samplers import UniformSampler
    import jax
    from jax.tree_util import tree_map
    from jax import jacrev
    from jaxpi.utils import flatten_pytree
    import seaborn as sns

    t0 = t_star[0]
    t1 = t_star[-1]
    x0 = x_star[0]
    x1 = x_star[-1]
    dom = jnp.array([[t0, t1], [x0, x1]])
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))
    batch = next(res_sampler)
    batch = jax.device_get(tree_map(lambda x: x[0], batch))

    grads = jacrev(model.losses)(params, batch)

    grad_dict = {}
    for key, value in grads.items():
        flattened_grad = flatten_pytree(value)
        grad_dict[key] = flattened_grad

    fig = plt.figure(figsize=(6, 5))
    sns.kdeplot(grad_dict['ics'], fill=True, alpha=0.5, color='blue', label='ics')
    sns.kdeplot(grad_dict['res'], fill=True, alpha=0.5, color='red', label='res')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.xlim([-0.1, 0.1])
    plt.legend()
    fig_path = os.path.join(save_dir, "adv_grad.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6, 5))
    sns.kdeplot(grad_dict['ics'], fill=True, alpha=0.5, color='blue', label='ics')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.xlim([-0.1, 0.1])
    plt.legend()
    fig_path = os.path.join(save_dir, "adv_grad_ics.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()

    fig = plt.figure(figsize=(6, 5))
    sns.kdeplot(grad_dict['res'], fill=True, alpha=0.5, color='red', label='res')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.xlim([-0.1, 0.1])
    plt.legend()
    fig_path = os.path.join(save_dir, "adv_grad_res.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()