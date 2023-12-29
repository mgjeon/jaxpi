# Problem

- `utils.get_dataset`

$$
u = u(t, x)
$$

$$
\begin{aligned}
    & \frac{\partial u}{\partial t} + c\frac{\partial u}{\partial x} = 0,\quad t\in[0,1],\ x\in (0, 2\pi),\\
    & u(0,x) = g(x), \quad x\in(0, 2\pi) \\\\
    & c=80, \quad g(x) = \sin(x)
\end{aligned}
$$

- periodic BC

    - $u(t, 0) = u(t, 2\pi)$
    - $P_x = 2\pi \rightarrow w_x = 2\pi/P_x = 1$

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