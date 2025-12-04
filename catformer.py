import jax
from jax import nn
from jax import numpy as jnp
from jax import random as jr
from simple_pytree import Pytree, static_field


class CatFormer(Pytree):
    num_features: int = static_field()
    causal_mask: bool = static_field()
    scratchpad: bool = static_field()

    def attn(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        if self.causal_mask:
            attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn

    def embed(self, x):
        # wte = jnp.eye(self.num_features)[x]
        wte = x
        wpe = jnp.eye(x.shape[-2])
        wpe = jnp.broadcast_to(wpe, (*x.shape[:-1], x.shape[-2]))
        return jnp.concatenate([wte, wpe], -1)

    def __init__(
        self,
        seq_len,
        num_features,
        heads,
        causal_mask,
        scratchpad,
    ):
        self.num_features = num_features
        self.causal_mask = causal_mask
        self.scratchpad = scratchpad
        d = seq_len + num_features
        self.A = []
        for n_head in heads:
            self.A.append(jnp.zeros([n_head, d, d]))
            d *= 1 + n_head
        # Initialize A1
        # adj = [1 if i % 2 == 0 else 0 for i in range(seq_len-1)]
        # adj_mat_block = jnp.diag(jnp.array(adj), k=-1) 
        # self.A[0] = self.A[0].at[0,self.num_features:(self.num_features + seq_len), self.num_features:(self.num_features + seq_len)].set(adj_mat_block)

        # Initialize A2
        # a2 = jnp.diag(jnp.array([1 if i % 2 == 0 else -1 for i in range(num_features)]))
        # self.A[1] = self.A[1].at[0,:num_features,(num_features+seq_len):(2*num_features + seq_len)].set(a2)
        
        # Initialize W
        self.W = jnp.zeros((d, num_features))
        # self.W = self.W.at[:, 2*(num_features + seq_len):3*num_features + 2*seq_len].set(jnp.eye(num_features))

    def __call__(self, x):
        x = self.embed(x)
        for Ai in self.A:
            attn = jax.vmap(self.attn, (None, 0), -2)(x, Ai)
            attn = attn.reshape(*attn.shape[:-2], -1)
            x = jnp.concatenate([x, attn], -1)
        if self.scratchpad:
            if self.causal_mask:
                x = x[..., self.num_features*3:, :]
            else:
                x = x[..., :self.num_features, :]
        else:
            x = x[..., :self.num_features, :]
        # x = x[..., :self.num_features, :]
        # return nn.softmax(x @ self.W)
        return x @ self.W
    