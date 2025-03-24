# stochastic LQR with additive homogenenous noise has the same solution as determinitic dynamics does.
# instead of doing adjoint to optimize dynamics, we can use XFADS trajectory and iLQR.
# finite horizon, discrete time
# z[t+1] = A[t]z[t] + B[t]u[t]
import functools as ft

from jax import lax, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array


def lqr_backward(z, Fz, Fu, Czz, Cuu):
    # Backward in Algo. 1
    # we can set the cross cost to be 0 (equiv. independent joint distribution of z and u)
    # the state cost is posterior of z
    # the control cost is prior of u
    
    # v[T] initial cost
    # V[T] initial cost
    # Check cz, cu, Czz, Cuz, Cuu in S15, the indices indicate the shape

    def step(carry: tuple[Array, Array], xs: tuple):
        # z[t+1] = Az[t] + Bu[t]
        S_tp1, s_tp1 = carry
        z_t, A, B, Czz, Cuu = xs
        
        M = jnp.linalg.multi_dot((B.T, S_tp1, B))
        L = cho_factor(M + Cuu)
        M = cho_solve(L, B.T)
        K_t = jnp.linalg.multi_dot((M, S_tp1, A))
        D = A - B @ K_t
        S_t = jnp.linalg.multi_dot((A.T, S_tp1, D)) + Czz
        s_t = D.T @ s_tp1 + Czz @ z_t
        k_t = M @ s_tp1
        u_t = - K_t @ z_t + k_t
        
        return (S_t, s_t), u_t


    # def step(carry: tuple[Array, Array], xs: tuple):
    #     # v, V: value function, V(z)
    #     # q, Q: action value function, Q(z, u)
    #     v, V = carry
    #     Fz, Fu, cz, cu, Czz, Cuz, Cuu = xs

    #     Qzz = Czz + Fz.T @ V @ Fz  # (Z, Z)
    #     Quz = Cuz + Fu.T @ V @ Fz  # (U, Z)
    #     Quu = Cuu + Fu.T @ V @ Fu  # (U, U)

    #     qz = cz + Fz.T @ v  # (Z,)
    #     qu = cu + Fu.T @ v  # (U,)
        
    #     Luu = cho_factor(Quu)  # (U, U)

    #     K = -cho_solve(Luu, Quz)  # (U, Z)
    #     k = -cho_solve(Luu, qu)  # (U,)
    #     V = Qzz + Quz.T @ K  # (Z, Z) + (Z, U) (U, Z) -> (Z, Z)
    #     v = qz + K.T @ qu  # (Z,) + (Z, U) (U,) -> (Z,)
        
    #     return (v, V), (k, K)
    
    S_T = Czz[-1]
    s_t = Czz[-1] @ z[-1]
    _, u = lax.scan(step, init=(S_T, s_t), xs=(z[:-1], Fz[:-1], Fu[:-1], Czz[:-1], Cuu[:-1]), reverse=True)

    return u


def lqr_forward(Fz, Fu, u, z0):
    def step(z_t, xs):
        Fz, Fu, u_t = xs
        z_tp1 = Fz @ z_t + Fu @ u_t
        return z_tp1, z_tp1
    
    _, zhat = lax.scan(step, init=z0, xs=(Fz[:-1], Fu[:-1], u))
    zhat = jnp.vstack((z0, zhat))
    return zhat


def cost(z, u):
    # iLQR-VAE eq.11
    # posterior of z
    # prior of u
    pass


def forward(f, z, u, k, K, gamma):    
    def step(carry, xs):
        zp, up, a = carry
        z, u, k, K = xs

        zp = f(zp, up)
        up = K @ (zp - z) + a * k
        a = a * gamma
        
        return (zp, up, a), (zp, up)

    a = 0.
    while True:
        (_, _, a), (zp, up) = lax.scan(step, init=(z[0], k[0], a), xs=(z[1:], u[1:], k[1:], K[1:]))
        if cost(zp, up) < cost(z, u):
            break
    
    return zp, up


def check_convergence(z, u, zn, zu):
    pass


def ilqr_solve(f, z0, u, gamma):
    converged = False
    z = f.rollout(z0, u)  # let f handle it
    while not converged:
        # linearized f
        Fz, Fu, cz, cu, Czz, Cuz, Cuu = f.linearize(z, u)  # let f handle its own linearization
        k, K = lqr_backward(Fz, Fu, cz, cu, Czz, Cuz, Cuu)
        zn, un = forward(f, z, u, k, K, gamma, cost)

        converged = check_convergence(z, u, zn, un)
    
    return u  # supposedly the posterior mean
