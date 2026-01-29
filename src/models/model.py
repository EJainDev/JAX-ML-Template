import flax.nnx as nnx


class Model(nnx.Module):
    fc: nnx.Linear

    def __init__(self, num_features, num_targets, rngs):
        self.fc = nnx.Linear(num_features, num_targets, rngs=rngs)

    def __call__(self, x):
        return self.fc(x)
