import jax.numpy as jnp
from jax import vmap


def get_dataset(T=1.0, L=2 * jnp.pi, c=80, n_t=200, n_x=128):
    """
    T : final time
    L : length of the domain
    c : advection speed
    n_t : number of time steps
    n_x : number of spatial points
    """
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    u_exact_fn = lambda t, x: jnp.sin(jnp.mod(x - c * t, L))

    u_exact = vmap(vmap(u_exact_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, t_star, x_star
