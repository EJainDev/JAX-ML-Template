import os
import jax
import grain.python as pygrain
from flax import nnx
from typing import Any
import joblib
import orbax.checkpoint as ocp
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
import grain
from tqdm.auto import tqdm

from ..evaluation.criterion import compute_mse

from ..models.dataset import Datasource

from ..models.model import Model

from ..config import *


def evaluate(
    model: Model,
    test_dataset: grain.IterDataset,
    num_test_samples: int,
) -> float:
    """
    Evaluates the given model on a test dataset and returns the loss.

    :param model: The trained model to evaluate.
    :type model: Model
    :param test_dataset: A grain iterable dataset that contains the test samples and on iteration returns a dictionary with keys 'features' and 'targets'
    :type test_dataset: grain.IterDataset
    :param num_test_samples: The number of test samples in the test dataset
    :type num_test_samples: int
    :return: The average loss on the test dataset
    :rtype: float
    """

    model.eval()  # Switch to evaluation mode

    # Get the graphdef (constant), parameters (changing values), and static (everything else)
    graphdef, params, static = nnx.split(model, nnx.Param, nnx.Everything)

    # Get the constant graph defs of the params for later reconstruction as well as the leaves
    param_leaves, param_graph_def = jax.tree.flatten(params)

    @jax.jit
    def test_step(param_leaves: list[Any], x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Does one evaluation step on the given inputs and returns loss.

        :param param_leaves: The leaves of the pytree representing the model parameters
        :type param_leaves: list[Any]
        :param x: The input features for the batch to be fed directly into the model
        :type x: jax.Array
        :param y: The target predictions for the batch to be fed directly into the loss function
        :type y: jax.Array
        :return: The loss value for the batch
        :rtype: jax.Array
        """
        # Restore the params pytree from the leaves
        params = jax.tree.unflatten(param_graph_def, param_leaves)

        # Restore the model with the current params
        model = nnx.merge(graphdef, params, static)

        # Get model predictions
        preds = model(x)

        # Get loss value
        loss = compute_mse(preds, y)

        return loss

    # Calculate number of batches
    test_steps = num_test_samples // BATCH_SIZE

    # Accumulate loss
    test_loss = 0.0

    # Use tqdm for progress bars
    for data in tqdm(
        test_dataset,
        total=test_steps,
        desc="Evaluating",
        leave=True,
    ):
        x: jax.Array = jax.device_put(data["features"])  # Move data to accelerator
        y: jax.Array = jax.device_put(data["targets"])  # Move data to accelerator

        # Evaluate
        loss = test_step(param_leaves, x, y)

        # Accumulate loss
        test_loss += loss.item()

    test_loss /= test_steps  # Get average test loss

    print(f"Test Loss: {test_loss:.6f}")

    return test_loss


if __name__ == "__main__":
    # Do your data loading here
    data = pl.read_parquet(PROCESSED_DATA_FILE)
    scaler = joblib.load(SCALER_PATH)

    X: np.ndarray = scaler.transform(data.to_numpy()[:-1])
    y: np.ndarray = data.select(["x", "y", "z"]).to_numpy()[1:]

    train_X, val_test_X, train_y, val_test_y = train_test_split(
        X,
        y,
        train_size=TRAIN_VAL_TEST_SPLIT[0],
        random_state=METADATA["seed"],
        shuffle=False,
    )

    val_X, test_X, val_y, test_y = train_test_split(
        val_test_X,
        val_test_y,
        train_size=0.5,
        random_state=METADATA["seed"],
        shuffle=False,
    )

    # Grain dataset for test set
    test_dataset = (
        pygrain.MapDataset.source(Datasource(test_X, test_y))  # type: ignore
        .to_iter_dataset()
        .batch(BATCH_SIZE, drop_remainder=True)
        .mp_prefetch(pygrain.MultiprocessingOptions(5, 16))
    )

    # Get the number of test samples
    num_test_samples = test_X.shape[0]

    # Create a flax compatible key
    rngs = nnx.Rngs(METADATA["seed"])

    # Create the model
    model = Model(test_X.shape[1], test_y.shape[1], rngs)

    # Load checkpoint
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        best_fn=lambda tree: tree["val_loss"] + tree["train_loss"] / 5,
        best_mode="min",
    )
    mngr = ocp.CheckpointManager(
        CHECKPOINT_DIR,
        options=options,
        item_names=("model_state", "optimizer_state", "metadata"),
    )

    if os.listdir(CHECKPOINT_DIR) != []:
        restored = mngr.restore(
            step=None,
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(),
                optimizer_state=ocp.args.PyTreeRestore(),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        graphdef, params, static = nnx.split(model, nnx.Param, nnx.Everything)
        param_graph_def = jax.tree.flatten(params)[1]
        params = jax.tree.unflatten(
            param_graph_def, jax.tree.leaves(restored["model_state"])
        )
        model = nnx.merge(graphdef, params, static)
        print("Model checkpoint loaded successfully.")
    else:
        print("Warning: No checkpoint found. Evaluating untrained model.")

    # Evaluate on test set
    try:
        test_loss = evaluate(model, test_dataset, num_test_samples)

        # Save loss to file
        with open(TEST_RESULTS_DIR / "test_results.txt", "w") as f:
            f.write(f"{test_loss:.6f}")
        print(f"Test loss saved to {TEST_RESULTS_DIR / "test_results.txt"}")
        print(f"Test Loss: {test_loss:.6f}")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
