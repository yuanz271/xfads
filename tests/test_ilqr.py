from pathlib import Path

import jax
from jax import lax, numpy as jnp, random as jrnd, vmap
import chex
import matplotlib.pyplot as plt

from xfads.ilqr import lqr_backward, lqr_forward, ilqr


def test_lqr_solve():
    # prepare matrices
    T = 10
    zd = 2
    ud = 1
    
    A = jrnd.uniform(jrnd.key(0), shape=(zd, zd))
    _, A = jnp.linalg.eig(A)
    A = jnp.real(A)
    
    B = jrnd.normal(jrnd.key(1), shape=(zd, ud))

    u = jrnd.uniform(jrnd.key(2), shape=(T, ud), minval=-1, maxval=1) * 2
    
    def step(z, u):
        z = A@z + B@u
        return z, z
    
    _, z = lax.scan(step, init=jnp.ones(zd), xs=u)

    # >>>
    Fz = jnp.broadcast_to(A, (T, zd, zd))
    Fu = jnp.broadcast_to(B, (T, zd, ud))
    chex.assert_shape(Fz, (T, zd, zd))
    chex.assert_shape(Fu, (T, zd, ud))
    
    Czz = jnp.broadcast_to(jnp.eye(zd), (T, zd, zd))
    Cuu = jnp.broadcast_to(jnp.eye(ud) * 0.05, (T, ud, ud))
    # <<<
    
    u_hat = lqr_backward(z, Fz, Fu, Czz, Cuu)
    z0 = z[0]
    z_hat = lqr_forward(Fz, Fu, u_hat, z0)
    
    fig, ax = plt.subplots()
    ax.plot(*z.T, color='b')
    ax.scatter(*z[0], marker='x', color='b')
    ax.plot(*z_hat.T, color='r')
    ax.scatter(*z_hat[0], marker='x', color='r')
    fig.tight_layout()
    fig.savefig(Path.cwd() / "test_lqr.pdf")
    plt.close(fig)


def test_jacobian():
    T = 5
    zd = 2
    ud = 1
    
    A = jrnd.uniform(jrnd.key(0), shape=(zd, zd))
    _, A = jnp.linalg.eig(A)
    A = jnp.real(A)
    
    B = jrnd.normal(jrnd.key(1), shape=(zd, ud))

    def f(x, u):
        return A@x + B@u
    
    def step(x, u):
        x = f(x, u)
        return x, x

    u = jrnd.uniform(jrnd.key(2), shape=(T, ud), minval=-1, maxval=1) * 2
    _, z = lax.scan(step, init=jnp.ones(zd), xs=u)

    
    F = jax.jacobian(f, argnums=(0, 1))
    print(A, B)
    print(F(z[0], u[0]))


def test_ilqr():
    # prepare matrices
    T = 10
    zd = 2
    ud = 1
    
    A = jrnd.uniform(jrnd.key(0), shape=(zd, zd))
    _, A = jnp.linalg.eig(A)
    A = jnp.real(A)
    
    B = jrnd.normal(jrnd.key(1), shape=(zd, ud))

    u = jrnd.uniform(jrnd.key(2), shape=(T, ud), minval=-1, maxval=1) * 2
    
    def f(x, u):
        return A@x + B@u
    
    def step(x, u):
        x = f(x, u)
        return x, x
    
    _, z = lax.scan(step, init=jnp.ones(zd), xs=u)

    # >>>
    Q = jnp.broadcast_to(jnp.eye(zd) * 10, (T, zd, zd))
    R = jnp.broadcast_to(jnp.eye(ud) * 0.01, (T, ud, ud))
    # <<<
    
    u, x = ilqr(f, z, jnp.zeros_like(u), Q, R)
    
    fig, ax = plt.subplots()
    ax.plot(*z.T, color='b')
    ax.scatter(*z[0], marker='x', color='b')
    ax.plot(*x.T, color='r')
    ax.scatter(*x[0], marker='x', color='r')
    fig.tight_layout()
    fig.savefig(Path.cwd() / "test_ilqr.pdf")
    plt.close(fig)


if __name__ == "__main__":
    # test_lqr_solve()
    # test_jacobian()
    test_ilqr()
