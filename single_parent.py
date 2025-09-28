from pathlib import Path

import optax
import tyro
from util import *
from PIL import Image

import wandb
from tqdm.auto import tqdm
from catformer import CatFormer
from plots import *
from problems import *


def main(
    num_features: int = 10, # this is the total number of features, not the number of features in each subspace
    seed = 1,
    lr: float = 1,
    wd: float = 0,
    steps: int = 2**16,
    n_save: int = 128,
    batch_size: int = 1024,
    max_size: int = 2**20,
):
    config = locals()
    rng = RNG(seed)
    seq_len = num_features*2

    problem = InContextTree(
        num_features=num_features,
    )

    model = CatFormer(
        seq_len=seq_len,
        num_features=num_features,
        heads=[1, 1],
    )

    @jit
    def criterion(f, y):
        # _criterion = lambda f, y: -jnp.log(f) @ y
        _criterion = lambda f, y: jnp.sum((f - y)**2)
        for _ in range(y.ndim - 1):
            _criterion = vmap(_criterion)
        return _criterion(f, y).mean()

    @jit
    def loss_fn(model, batch):
        x, y = batch
        return criterion(model(x), y)

    def accuracy_fn(model, batch):
        x, y = batch
        return jnp.sum(model(x) == y) / x.shape[0]

    wandb.init(project="ICL", config=config, name="single_parent_20")


    print("Computing Bayes")

    testx, testy = vmap(problem.sample)(rng.next(2**16))
    answers = vmap(problem.bayes)(testx)
    bayes = criterion(answers, testy)

    print("Training")
    save_every = steps // n_save
    epoch_len = max_size // batch_size
    sample_fn = jit(lambda k: vmap(problem.sample)(jr.split(k, epoch_len * batch_size)))

    def batch_iterator(key):
        while True:
            key, subkey = jr.split(key)
            batches = sample_fn(subkey)
            for i in range(epoch_len):
                yield tree_map(
                    lambda x: x[batch_size * i : batch_size * (i + 1)], batches
                )

    @jit
    def step_fn(model, batch, lr, wd):
        g = jax.grad(loss_fn)(model, batch)
        g = tree_map(lambda x: jnp.clip(x, -1.0, 1.0), g)
        g = tree_add_scalar_mul(g, wd, model)
        model = tree_add_scalar_mul(model, -lr, g)
        return model

    iterator = batch_iterator(rng.next())
    schedule = optax.cosine_decay_schedule(lr, steps)
    test_losses = []
    pbar = tqdm(total=steps)
    for i in range(steps):
        if i % save_every == 0:
            test_loss = loss_fn(model, (testx, testy))
            # print(model(testx[0]))
            test_losses.append(test_loss)
            wandb.log(dict(loss=test_loss, bayes=bayes, step=i, lr=schedule(i)))
            pbar.n = i
            pbar.refresh()
        model = step_fn(model, next(iterator), lr=schedule(i), wd=wd)
    pbar.n = steps
    pbar.refresh()
    pbar.close()
    test_losses = jnp.array(test_losses)

    fig = plot_losses(test_losses, bayes, save_every)
    filename = wandb.run.dir + "/losses.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({"losses/test": wandb.Image(Image.open(filename))})
    plt.close(fig)

    lower = 15
    upper = 100
    
    fig = plot_A1(model.A[0][0], num_features, seq_len, lower, upper)
    filename = wandb.run.dir + f"/A1.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({f"A1/{lower}_{upper}": wandb.Image(Image.open(filename))})
    plt.close(fig)

    fig = plot_A2(model.A[1][0], num_features, seq_len, lower, upper)
    filename = wandb.run.dir + f"/A2.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({f"A2/{lower}_{upper}": wandb.Image(Image.open(filename))})
    plt.close(fig)

    fig = plot_W(model.W.T, num_features, seq_len, lower, upper)
    filename = wandb.run.dir + f"/W.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({f"W/{lower}_{upper}": wandb.Image(Image.open(filename))})
    plt.close(fig)

    test_acc = accuracy_fn(model, (testx, testy))
    wandb.log({"accuracy/test": test_acc})

    # test_loss = loss_fn(model, (testx, testy))
    # wandb.log({"loss/test": test_loss})
        
    wandb.finish()


tyro.cli(main)
# tyro.cli(main)
