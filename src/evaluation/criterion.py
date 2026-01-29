import jax
import jax.numpy as jnp


def compute_accuracy(
    predictions: jax.Array, targets: jax.Array, tolerance=1e-3
) -> jax.Array:
    """
    Compute the accuracy of predictions against the targets.

    Args:
        predictions (jax.Array): The predicted labels.
        targets (jax.Array): The true labels.

    Returns:
        jax.Array: The accuracy as a jax.Array of shape (1,) with a value between 0 and 1.
    """
    correct_predictions = jnp.sum(jnp.abs(predictions - targets) < tolerance)
    total_predictions = predictions.shape[0]
    return correct_predictions / total_predictions


def compute_mse(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """
    Compute the Mean Squared Error (MSE) between predictions and targets.

    Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true values.

    Returns:
        jax.Array: The MSE as a jax.Array of shape (1,).
    """
    mse = jnp.mean((predictions - targets) ** 2)
    return mse


def compute_mae(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """
    Compute the Mean Absolute Error (MAE) between predictions and targets.

    Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true values.

    Returns:
        jax.Array: The MAE as a jax.Array of shape (1,).
    """
    mae = jnp.mean(jnp.abs(predictions - targets))
    return mae


def compute_rmse(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """
    Compute the Root Mean Squared Error (RMSE) between predictions and targets.

    Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true values.

    Returns:
        jax.Array: The RMSE as a jax.Array of shape (1,).
    """
    rmse = jnp.sqrt(jnp.mean((predictions - targets) ** 2))
    return rmse


def compute_r2_score(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """
    Compute the R-squared (R2) score between predictions and targets.

    Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true values.

    Returns:
        jax.Array: The R2 score as a jax.Array of shape (1,).
    """
    ss_res = jnp.sum((targets - predictions) ** 2)
    ss_tot = jnp.sum((targets - jnp.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score


def compute_log_loss(
    predictions: jax.Array, targets: jax.Array, epsilon=1e-5
) -> jax.Array:
    """
    Compute the Log Loss (Cross-Entropy Loss) between predictions and targets.

    Args:
        predictions (jax.Array): The predicted probabilities.
        targets (jax.Array): The true labels (0 or 1).

    Returns:
        jax.Array: The Log Loss as a jax.Array of shape (1,).
    """
    predictions = jnp.clip(predictions, epsilon, 1 - epsilon)
    log_loss = -jnp.mean(
        targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions)
    )
    return log_loss
