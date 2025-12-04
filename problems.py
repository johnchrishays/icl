from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from jax.numpy import linalg as jla


class InContextTree:
    def __init__(self, num_features, scratchpad):
        self.num_features = num_features
        self.scratchpad = scratchpad

    def sample(self, key):
        seq_key, test_key = jr.split(key, 2)
        
        perm_key, seq_key = jr.split(seq_key)
        perm_indices = jr.permutation(perm_key, self.num_features)
        permutation = jnp.eye(self.num_features)[perm_indices]

        # add permutation to beginning (left of columns)
        y = jnp.float32(jr.bernoulli(test_key, 0.5, (self.num_features, self.num_features)))
        # y_perm_indices = jr.permutation(test_key, self.num_features)
        # y = jnp.eye(self.num_features)[y_perm_indices]
        if self.scratchpad:
            s = jnp.zeros((self.num_features, self.num_features))
            # s = jnp.concatenate([permutation, y[perm_indices]], axis=0)
            x = jnp.concatenate([s, permutation, y[perm_indices], s], axis=0)
        else:
            x = jnp.concatenate([permutation, y[perm_indices]], axis=0)
        return x, y

    def bayes(self, seq):
        return seq[:self.num_features].T @ seq[self.num_features:self.num_features*2]
        # return seq[:self.num_features].T @ seq[self.num_features:]


