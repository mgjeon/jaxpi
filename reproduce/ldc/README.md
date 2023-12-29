# Problem

- `utils.get_dataset`

$$
\begin{aligned}
    &\mathbf{u} \cdot \nabla \mathbf{u}+\nabla p-\frac{1}{R e} \nabla^2 \mathbf{u}&=0, \quad  (x,y) \in (0,1)^2, \\
    &\nabla \cdot \mathbf{u}&=0, \quad  (x,y) \in (0,1)^2, \\\\
\end{aligned}
$$

- solution
$$
\begin{aligned}
    &\mathbf{u} = (u, v) \\
    &u = u(x, y) \\
    &v = v(x, y)
\end{aligned}
$$

- u : x-component of velocity
- v : y-component of velocity
- p : pressure

- BC

    - top : $\mathbf{u}=(1, 0)$
    - others : non-slip boundary condition $\mathbf{u}=\mathbf{0}$

```
python main.py --config ./configs/plain.py
python main.py --config ./configs/plain.py --config.model=eval
```


# Additional techniques

- Trainable temporal periodicity
    - $u(0, x) = u(P_t, x)$
    - $P_t = 1 \rightarrow w_t = 2\pi/P_t = 2\pi$

- Fourier feature embeddings
    - `{"embed_scale": 1.0, "embed_dim": 256}`

- Weight factorization
    - `{"type": "weight_fact", "mean": 1.0, "stddev": 0.1}`

- Loss Balancing
    - Grad Norm Weighting

- Respect causality
    - Time-dependent PDE

```
python main.py --config ./configs/default.py
python main.py --config ./configs/default.py --config.model=eval
```