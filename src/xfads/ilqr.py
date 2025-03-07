# stochastic LQR with additive homogenenous noise has the same solution as determinitic dynamics does.
# instead of doing adjoint to optimize dynamics, we can use xfads trajectory and iLQR.
# finite horizon, discrete time
# z[t+1] = A[t]z[t] + B[t]u[t]

from jax import lax, numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve


def lqr_solve(Fz, Fu, cz, cu, Czz, Cuz, Cuu):
    # Q: state cost
    # R: control cost
    # N: cross cost
    # u[t] = -K[t]z[t]
    # K[t] = (R[t] + B[t].T @ P[t+1] @ B[t])^{-1} (B[t].T @ P[t+1] A[t] + N[t].T)
    # P[t-1] = A[t].T P[t] A[t] - (B[t].T @ P[t+1] A[t] + N[t].T).T (R[t] + B[t].T @ P[t+1] @ B[t])^{-1} (B[t].T @ P[t+1] A[t] + N[t].T) + Q[t]
    # P[T+1] = Q
    
    # we can set the cross cost to be 0 (equiv. independent joint distribution of z and u)
    # the state cost is posterior of z
    # the control cost is prior of u
    
    # v[T] initial cost
    # V[T] initial cost

    def step(carry, xs):
        v, V = carry
        Fz, Fu, cz, cu, Czz, Cuz, Cuu = xs

        Qzz = Czz + Fz.T @ V @ Fz
        Quz = Cuz + Fu.T @ V @ Fz
        Quu = Cuu + Fu.T @ V @ Fu

        qz = cz + Fz @ v
        qu = cu + Fu @ v
        
        cho = cho_factor(Quu)

        K = -cho_solve(cho, Quz)
        k = -cho_solve(cho, qu)
        V = Qzz + Quz.T @ K
        v = qz + K @ qu
        
        return (v, V), (k, K)
    
    _, (k, K) = lax.scan(step, init=(cz[-1], Czz[-1]), xs=(Fz[:-1], Fu[:-1], cz[:-1], cu[:-1], Czz[:-1], Cuz[:-1], Cuu[:-1]), reverse=True)

    return k, K


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
        k, K = lqr_solve(Fz, Fu, cz, cu, Czz, Cuz, Cuu)
        zn, un = forward(f, z, u, k, K, gamma, cost)

        converged = check_convergence(z, u, zn, un)
    
    return u  # supposedly the posterior mean
