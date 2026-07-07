#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_fractional_optimizer_activation_experiments.py

Self-contained TensorFlow experiment runner for testing:
    - standard optimizers,
    - Herrera-style fractional optimizers,
    - explicit memory-based fractional optimizers,
    - adaptive variable-order fractional Adam,

across:
    - standard activations,
    - fractal activations from the uploaded activation module,

on classification datasets.

Main design goals
-----------------
1. No argparse. All settings are defined at the top of the file.
2. Similar result layout to the uploaded reporting script:
       <RESULTS_ROOT>/<dataset>/<config_name>.json
   where each JSON contains:
       dataset, optimizer, activation, vderivs, per-run metrics, etc.
3. Easy activation selection:
   fractal activations are all listed in one place so they can be commented out.
4. Includes training and testing wall-clock timing.
5. Uses a custom training loop so that:
   - all optimizers are handled consistently,
   - the adaptive loss-driven variable-order optimizer can call
     optimizer.set_current_loss(loss) when available.
6. Intended for classification benchmarks and optimizer/activation interaction studies.

Why this structure
------------------
The uploaded summary script expects one JSON per configuration under
<root>/<dataset>/*.json, and then extracts:
    dataset, optimizer, activation, vderiv, runs, accuracy
from a "vderivs" list. This script writes results in that format so that the
uploaded summary script can be reused directly afterwards.

Uploaded sources used as design reference
-----------------------------------------
- The report script expects JSON files in <SOURCE_DIR>/<dataset>/*.json and
  computes summaries over fields including dataset, optimizer, activation,
  vderiv, and per-run accuracy results.
- The uploaded fractal activation module exposes several fractal activation
  functions such as modulated Blancmange, modified Weierstrass, and
  Weierstrass-Mandelbrot variants.
- The uploaded optimizer modules provide:
    * Herrera-style fractional optimizers,
    * memory-based fractional optimizers,
    * adaptive variable-order memory-based Adam.

Experimental questions this script supports
-------------------------------------------
1. Do fractional optimizers improve classification performance relative to
   standard baselines?
2. Do fractal activations interact differently with:
   - classical optimizers,
   - Herrera-style fractional optimizers,
   - explicit memory-based fractional optimizers,
   - adaptive variable-order fractional optimizers?
3. What is the computational overhead in training and evaluation time?
4. Does variable-order adaptation change behavior relative to fixed-order
   memory-based methods?

Recommended workflow
--------------------
1. Edit the parameter block at the top.
2. Run the script.
3. Reuse the uploaded summary script on the results folder to create
   text reports of average accuracy, standard deviation, max, and min.

Notes
-----
- This script uses scikit-learn tabular datasets by default so it is
  immediately runnable without additional local data files.
- If you want to plug in your exact previous datasets, edit `load_dataset()`.
- The script uses one common MLP backbone to isolate optimizer / activation effects.
- For fairness, all compared optimizers share the same network architecture,
  seed protocol, split sizes, and training loop.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------------------------------------------------
# IMPORT UPLOADED OPTIMIZER AND ACTIVATION MODULES
# ---------------------------------------------------------------------
# Adjust these import names if your local filenames differ.
# They are aligned with the uploaded files referenced in this conversation.
from class_b_tf_fractional_optimizers_OHA import (
    FSGD,
    FAdam,
    FAdagrad,
    FAdadelta,
    FRMSprop,
    FAdamW,
)
from class_c_tf_fractional_optimizers_SR import (
    MemoryFSGD,
    MemoryFRMSprop,
    MemoryFAdam,
    MemoryFAdadelta,
)
from class_e_tf_gen_var_fadam import AdaptiveModesVariableOrderMemoryFAdam

from class_tf_gen_var_frmsprop import AdaptiveRandomVariableOrderMemoryFRMSprop

import fractal_activation_functions as faf


# =============================================================================
# 0) GLOBAL EXPERIMENT PARAMETERS
# =============================================================================

# -----------------------------
# Reproducibility
# -----------------------------
GLOBAL_SEED = 42
RUN_SEEDS = [11, 22, 33]   # one seed per repeated run

# -----------------------------
# Output folders
# -----------------------------
RESULTS_ROOT = Path("results_fractional_activation_optimizer_study")
HISTORY_ROOT = RESULTS_ROOT / "histories"
MODEL_ROOT = RESULTS_ROOT / "saved_models"  # optional; set SAVE_MODELS=False if not needed

SAVE_MODELS = False
SAVE_PER_RUN_HISTORY_JSON = True
SAVE_PER_RUN_PREDICTIONS = False

# -----------------------------
# Datasets to test
# -----------------------------
# Built-in tabular classification datasets for a self-contained experiment.
# Replace or extend this list in `load_dataset()` if you want your earlier datasets.
DATASETS_TO_RUN = [
    "breast_cancer",
    "wine",
    "digits",
]

# -----------------------------
# Train/validation/test split
# -----------------------------
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20

# -----------------------------
# Training hyperparameters
# -----------------------------
EPOCHS = 40
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 8
SHUFFLE_EACH_EPOCH = True

# -----------------------------
# Network architecture
# -----------------------------
HIDDEN_UNITS = [128, 64]
DROPOUT_RATE = 0.10
USE_BATCH_NORM = False

# -----------------------------
# Activation selection
# -----------------------------
# Standard activations
STANDARD_ACTIVATIONS = {
    #"relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    #"elu": tf.nn.elu,
    #"swish": tf.nn.swish,
}

# Fractal activations from uploaded module.
# Comment out any entries you do not want to test.
FRACTAL_ACTIVATIONS = {
    #"modulated_blancmange_curve": faf.modulated_blancmange_curve,
    #"decaying_cosine_function_tf": faf.decaying_cosine_function_tf,
    "modified_weierstrass_function_tanh": faf.modified_weierstrass_function_tanh,
    #"modified_weierstrass_function_relu": faf.modified_weierstrass_function_relu,
    #"weierstrass_mandelbrot_function_xsinsquared": faf.weierstrass_mandelbrot_function_xsinsquared,
    #"weierstrass_mandelbrot_function_xpsin": faf.weierstrass_mandelbrot_function_xpsin,
    #"weierstrass_mandelbrot_function_relupsin": faf.weierstrass_mandelbrot_function_relupsin,
    #"weierstrass_mandelbrot_function_tanhpsin": faf.weierstrass_mandelbrot_function_tanhpsin,
    #"weierstrass_function_tf": faf.weierstrass_function_tf,
}

# Choose which activation families to include
INCLUDE_STANDARD_ACTIVATIONS = True
INCLUDE_FRACTAL_ACTIVATIONS = True

# -----------------------------
# Optimizer comparison design
# -----------------------------
# Baseline optimizers
BASELINE_OPTIMIZERS = [
    #"SGD",
    #"Adam",
    #"RMSprop",
    #"Adagrad",
    #"Adadelta",
]

# Herrera-style fractional optimizers
HERRERA_OPTIMIZERS = [
    #"FSGD",
    "FAdam",
    #"FRMSprop",
    #"FAdagrad",
    #"FAdadelta",
    #"FAdamW",
]

# Fixed-order memory-based optimizers
MEMORY_OPTIMIZERS = [
    #"MemoryFSGD",
    #"MemoryFRMSprop",
    #"MemoryFAdam",
    #"MemoryFAdadelta",
    "AdaptiveRandomVariableOrderMemoryFRMSprop",
]

# Variable-order optimizer family
VARIABLE_ORDER_OPTIMIZERS = [
    "AdaptiveModesVariableOrderMemoryFAdam",
]

INCLUDE_BASELINE_OPTIMIZERS = True
INCLUDE_HERRERA_OPTIMIZERS = True
INCLUDE_MEMORY_OPTIMIZERS = True
INCLUDE_VARIABLE_ORDER_OPTIMIZERS = True

# -----------------------------
# Fractional parameter grids
# -----------------------------
# For standard non-fractional optimizers, vderiv is fixed to 1.0.
HERRERA_VDERIVS = [0.6, 0.8, 1.0, 1.2]
MEMORY_VDERIVS = [0.6, 0.8, 1.0, 1.2]
MEMORY_HISTORY_SIZES = [6]  # you can expand to [4, 6, 8, 10]
MEMORY_NORMALIZE_COEFFS = False

# Variable-order optimizer settings
VAR_ORDER_INIT = 0.85
VAR_ORDER_MIN = 0.6
VAR_ORDER_MAX = 1.2
VAR_ORDER_HISTORY_SIZE = 6
VAR_ORDER_NORMALIZE_COEFFS = False

VAR_ORDER_MODES = [
    "ema_smoothed_gradient_variability",
    "schedule",
    "hybrid_transition",
]

VAR_ORDER_SCHEDULE_TYPES = [
    "linear",
    "cosine",
    "exponential",
]

# Additional improved adaptive optimizer parameters
VAR_ORDER_STABLE_NU_CAP = 1.0
VAR_ORDER_VARIABILITY_CLIP = 10.0
VAR_ORDER_ORDER_EMA_GAMMA = 0.90
VAR_ORDER_WARMUP_STEPS = 50
VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT = 1.0

VAR_ORDER_LOSS_EMA_BETA = 0.95
VAR_ORDER_LR_DECAY_PATIENCE = 50
VAR_ORDER_LR_DECAY_FACTOR = 0.7
VAR_ORDER_MIN_LR_FACTOR = 0.2
VAR_ORDER_LOSS_PLATEAU_TOLERANCE = 1e-4

# Adaptive randomized variable-order memory FRMSprop settings
# In this optimizer:
#   effective_nu_min = max(0.05, vderiv_init - ARVFRMSPROP_NU_MIN_OFFSET)
#   effective_nu_max = vderiv_init + ARVFRMSPROP_NU_MAX_OFFSET
ARVFRMSPROP_NU_MIN_OFFSET = 0.25
ARVFRMSPROP_NU_MAX_OFFSET = 1.5
ARVFRMSPROP_RANDOM_PERTURB_SCALE = 0.03
ARVFRMSPROP_RANDOM_DECAY = 0.999
ARVFRMSPROP_STABILITY_STEP_SIZE = 0.01
ARVFRMSPROP_STABILITY_EMA_GAMMA = 0.95
ARVFRMSPROP_STABLE_NU_CAP = 1.0
ARVFRMSPROP_WARMUP_STEPS = 50
ARVFRMSPROP_USE_SIGNED_RANDOM_SEARCH = True

# -----------------------------
# Learning rates and optimizer-specific defaults
# -----------------------------
LR_SGD = 1e-2
LR_ADAM = 1e-3
LR_RMSPROP = 1e-3
LR_ADAGRAD = 1e-2
LR_ADADELTA = 1.0
LR_ADAMW = 1e-3

SGD_MOMENTUM = 0.9
SGD_NESTEROV = False

RMSPROP_RHO = 0.9
RMSPROP_MOMENTUM = 0.0
RMSPROP_CENTERED = False

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-7
ADAM_AMSGRAD = False

ADAGRAD_INITIAL_ACCUMULATOR = 0.1
ADAGRAD_EPSILON = 1e-7

ADADELTA_RHO = 0.95
ADADELTA_EPSILON = 1e-7

ADAMW_WEIGHT_DECAY = 4e-3

# Herrera-style optimizer extra parameter
HERRERA_FRAC_EPSILON = 1e-6

# Variable-order adaptation parameters
VAR_ORDER_ADAPT_RATE = 0.05
VAR_ORDER_DELTA = 1e-8
VAR_ORDER_EMA_GAMMA = 0.95
VAR_ORDER_TOTAL_SCHEDULE_STEPS = EPOCHS * 200  # approximate default
VAR_ORDER_SCHEDULE_EXP_GAMMA = 4.0

# -----------------------------
# Runtime controls
# -----------------------------
VERBOSE_TRAINING = 0
PRINT_PROGRESS = True
TF_FLOATX = "float32"

# =============================================================================
# 1) REPRODUCIBILITY
# =============================================================================

tf.keras.backend.set_floatx(TF_FLOATX)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =============================================================================
# 2) DATASETS
# =============================================================================

def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a classification dataset.

    This function is intentionally simple and self-contained. Replace this with
    your exact earlier dataset-loading logic if needed.
    """
    if dataset_name == "breast_cancer":
        ds = load_breast_cancer()
        return ds.data.astype(np.float32), ds.target.astype(np.int32)

    if dataset_name == "wine":
        ds = load_wine()
        return ds.data.astype(np.float32), ds.target.astype(np.int32)

    if dataset_name == "digits":
        ds = load_digits()
        return ds.data.astype(np.float32), ds.target.astype(np.int32)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_dataset(
    dataset_name: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Prepare train/validation/test splits and standardization.
    """
    X, y = load_dataset(dataset_name)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        random_state=seed,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    n_features = X_train.shape[1]
    n_classes = int(np.max(y) + 1)

    return {
        "dataset_name": dataset_name,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }


# =============================================================================
# 3) MODEL
# =============================================================================

class ActivationLayer(tf.keras.layers.Layer):
    """
    Wrap an activation callable in a Keras Layer.

    A unique layer name can be passed from the outside so that several
    activation layers using the same activation function can coexist
    in one model.
    """
    def __init__(
        self,
        activation_fn: Callable[[tf.Tensor], tf.Tensor],
        activation_name: str,
        layer_name: Optional[str] = None,
        **kwargs,
    ):
        if layer_name is not None:
            kwargs["name"] = layer_name
        super().__init__(**kwargs)
        self.activation_fn = activation_fn
        self.activation_name = activation_name

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.activation_fn(inputs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"activation_name": self.activation_name})
        return config


def build_mlp(
    input_dim: int,
    n_classes: int,
    activation_fn: Callable[[tf.Tensor], tf.Tensor],
    activation_name: str,
) -> tf.keras.Model:
    """
    Common backbone for all optimizer/activation comparisons.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for i, units in enumerate(HIDDEN_UNITS):
        x = tf.keras.layers.Dense(units, name=f"dense_{i+1}")(x)
        if USE_BATCH_NORM:
            x = tf.keras.layers.BatchNormalization(name=f"bn_{i+1}")(x)

        # unique name per hidden layer
        x = ActivationLayer(
            activation_fn=activation_fn,
            activation_name=activation_name,
            layer_name=f"act_{activation_name}_{i+1}",
        )(x)

        if DROPOUT_RATE > 0.0:
            x = tf.keras.layers.Dropout(DROPOUT_RATE, name=f"dropout_{i+1}")(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"mlp_{activation_name}",
    )
    return model


# =============================================================================
# 4) OPTIMIZER FACTORY
# =============================================================================

def build_optimizer(
    optimizer_name: str,
    vderiv: float = 1.0,
    history_size: int = 6,
    adaptation_mode: str = "ema_smoothed_gradient_variability",
    schedule_type: str = "cosine",
) -> tf.keras.optimizers.Optimizer:
    """
    Create a configured optimizer instance.
    """
    # -----------------------------
    # Standard baselines
    # -----------------------------
    if optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
        )

    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    if optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
        )

    if optimizer_name == "Adagrad":
        return tf.keras.optimizers.Adagrad(
            lr=LR_ADAGRAD,
            initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR,
            epsilon=ADAGRAD_EPSILON,
        )

    if optimizer_name == "Adadelta":
        return tf.keras.optimizers.Adadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
        )

    if optimizer_name == "AdamW":
        return tf.keras.optimizers.AdamW(
            lr=LR_ADAMW,
            weight_decay=ADAMW_WEIGHT_DECAY,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
        )

    # -----------------------------
    # Herrera-style fractional
    # -----------------------------
    if optimizer_name == "FSGD":
        return FSGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdam":
        return FAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FRMSprop":
        return FRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdagrad":
        return FAdagrad(
            lr=LR_ADAGRAD,
            initial_accumulator_value=ADAGRAD_INITIAL_ACCUMULATOR,
            epsilon=ADAGRAD_EPSILON,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdadelta":
        return FAdadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    if optimizer_name == "FAdamW":
        return FAdamW(
            lr=LR_ADAMW,
            weight_decay=ADAMW_WEIGHT_DECAY,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=vderiv,
            frac_epsilon=HERRERA_FRAC_EPSILON,
        )

    # -----------------------------
    # Memory-based fractional
    # -----------------------------
    if optimizer_name == "MemoryFSGD":
        return MemoryFSGD(
            lr=LR_SGD,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFRMSprop":
        return MemoryFRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFAdam":
        return MemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "MemoryFAdadelta":
        return MemoryFAdadelta(
            lr=LR_ADADELTA,
            rho=ADADELTA_RHO,
            epsilon=ADADELTA_EPSILON,
            vderiv=vderiv,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
        )

    if optimizer_name == "AdaptiveRandomVariableOrderMemoryFRMSprop":
        return AdaptiveRandomVariableOrderMemoryFRMSprop(
            lr=LR_RMSPROP,
            rho=RMSPROP_RHO,
            momentum=RMSPROP_MOMENTUM,
            epsilon=ADAM_EPSILON,
            centered=RMSPROP_CENTERED,
            vderiv_init=vderiv,
            nu_min=ARVFRMSPROP_NU_MIN_OFFSET,
            nu_max=ARVFRMSPROP_NU_MAX_OFFSET,
            history_size=history_size,
            normalize_coefficients=MEMORY_NORMALIZE_COEFFS,
            order_delta=VAR_ORDER_DELTA,
            random_perturb_scale=ARVFRMSPROP_RANDOM_PERTURB_SCALE,
            random_decay=ARVFRMSPROP_RANDOM_DECAY,
            stability_step_size=ARVFRMSPROP_STABILITY_STEP_SIZE,
            stability_ema_gamma=ARVFRMSPROP_STABILITY_EMA_GAMMA,
            stable_nu_cap=ARVFRMSPROP_STABLE_NU_CAP,
            warmup_steps=ARVFRMSPROP_WARMUP_STEPS,
            use_signed_random_search=ARVFRMSPROP_USE_SIGNED_RANDOM_SEARCH,
        )

    # -----------------------------
    # Variable-order fractional Adam
    # -----------------------------
    if optimizer_name == "AdaptiveModesVariableOrderMemoryFAdam":
        return AdaptiveModesVariableOrderMemoryFAdam(
            lr=LR_ADAM,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
            amsgrad=ADAM_AMSGRAD,
            vderiv_init=VAR_ORDER_INIT,
            nu_min=VAR_ORDER_MIN,
            nu_max=VAR_ORDER_MAX,
            history_size=VAR_ORDER_HISTORY_SIZE,
            normalize_coefficients=VAR_ORDER_NORMALIZE_COEFFS,
            adaptation_mode=adaptation_mode,
            order_adapt_rate=VAR_ORDER_ADAPT_RATE,
            order_delta=VAR_ORDER_DELTA,
            ema_smoothing_gamma=VAR_ORDER_EMA_GAMMA,
            schedule_type=schedule_type,
            total_schedule_steps=VAR_ORDER_TOTAL_SCHEDULE_STEPS,
            schedule_exponential_gamma=VAR_ORDER_SCHEDULE_EXP_GAMMA,
            stable_nu_cap=VAR_ORDER_STABLE_NU_CAP,
            variability_clip=VAR_ORDER_VARIABILITY_CLIP,
            order_ema_gamma=VAR_ORDER_ORDER_EMA_GAMMA,
            warmup_steps=VAR_ORDER_WARMUP_STEPS,
            hybrid_final_fractional_second_moment=VAR_ORDER_HYBRID_FINAL_FRAC_SECOND_MOMENT,
            loss_ema_beta=VAR_ORDER_LOSS_EMA_BETA,
            lr_decay_patience=VAR_ORDER_LR_DECAY_PATIENCE,
            lr_decay_factor=VAR_ORDER_LR_DECAY_FACTOR,
            min_lr_factor=VAR_ORDER_MIN_LR_FACTOR,
            loss_plateau_tolerance=VAR_ORDER_LOSS_PLATEAU_TOLERANCE,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


# =============================================================================
# 5) TRAINING UTILITIES
# =============================================================================

LOSS_FN = tf.keras.losses.SparseCategoricalCrossentropy()


def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


@tf.function
def forward_loss(model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor, training: bool) -> tf.Tensor:
    y_pred = model(x, training=training)
    return LOSS_FN(y, y_pred)


def evaluate_model(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Dict[str, float]:
    start = time.perf_counter()
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    y_true_all = []
    y_pred_all = []

    for xb, yb in ds:
        logits = model(xb, training=False)
        preds = tf.argmax(logits, axis=1)
        y_true_all.append(yb.numpy())
        y_pred_all.append(preds.numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    elapsed = time.perf_counter() - start

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "test_time_seconds": float(elapsed),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_run(
    dataset_bundle: Dict[str, Any],
    optimizer: tf.keras.optimizers.Optimizer,
    activation_fn: Callable[[tf.Tensor], tf.Tensor],
    activation_name: str,
    run_seed: int,
) -> Dict[str, Any]:
    """
    Custom training loop so all optimizers are handled consistently.
    Supports the loss-driven variable-order optimizer through set_current_loss().
    """
    set_global_seed(run_seed)

    X_train = dataset_bundle["X_train"]
    y_train = dataset_bundle["y_train"]
    X_val = dataset_bundle["X_val"]
    y_val = dataset_bundle["y_val"]
    X_test = dataset_bundle["X_test"]
    y_test = dataset_bundle["y_test"]
    n_features = dataset_bundle["n_features"]
    n_classes = dataset_bundle["n_classes"]

    model = build_mlp(
        input_dim=n_features,
        n_classes=n_classes,
        activation_fn=activation_fn,
        activation_name=activation_name,
    )

    train_ds = make_tf_dataset(X_train, y_train, BATCH_SIZE, SHUFFLE_EACH_EPOCH, run_seed)
    val_ds = make_tf_dataset(X_val, y_val, BATCH_SIZE, False, run_seed)

    best_val_loss = np.inf
    best_weights = None
    epochs_no_improve = 0

    epoch_history: List[Dict[str, float]] = []

    train_start = time.perf_counter()

    for epoch in range(EPOCHS):
        # -----------------------------
        # Training epoch
        # -----------------------------
        train_losses = []

        for xb, yb in train_ds:
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss = LOSS_FN(yb, logits)

            grads = tape.gradient(loss, model.trainable_variables)

            # Required by the adaptive variable-order optimizer
            if hasattr(optimizer, "set_current_loss"):
                try:
                    optimizer.set_current_loss(loss)
                except Exception:
                    pass

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(float(loss.numpy()))

        # -----------------------------
        # Validation epoch
        # -----------------------------
        val_losses = []
        val_true = []
        val_pred = []

        for xb, yb in val_ds:
            logits = model(xb, training=False)
            loss = LOSS_FN(yb, logits)
            preds = tf.argmax(logits, axis=1)

            val_losses.append(float(loss.numpy()))
            val_true.append(yb.numpy())
            val_pred.append(preds.numpy())

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        val_acc = float(accuracy_score(val_true, val_pred))

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "val_accuracy": val_acc,
        }

        # Track current variable orders for the adaptive variable-order optimizer
        if hasattr(optimizer, "current_orders"):
            try:
                orders = optimizer.current_orders()
                epoch_record["mean_current_order"] = float(
                    np.mean([float(o.numpy()) for o in orders])
                )
            except Exception:
                pass

        if hasattr(optimizer, "current_lr_factor"):
            try:
                epoch_record["current_lr_factor"] = float(optimizer.current_lr_factor())
            except Exception:
                pass

        epoch_history.append(epoch_record)

        # Early stopping on validation loss
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_weights = model.get_weights()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if PRINT_PROGRESS:
            print(
                f"  epoch={epoch+1:03d} "
                f"train_loss={mean_train_loss:.5f} "
                f"val_loss={mean_val_loss:.5f} "
                f"val_acc={val_acc:.5f}"
            )

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break

    train_elapsed = time.perf_counter() - train_start

    if best_weights is not None:
        model.set_weights(best_weights)

    test_metrics = evaluate_model(model, X_test, y_test, BATCH_SIZE)

    result = {
        "epochs_completed": len(epoch_history),
        "best_val_loss": float(best_val_loss),
        "training_time_seconds": float(train_elapsed),
        "test_time_seconds": float(test_metrics["test_time_seconds"]),
        "accuracy": float(test_metrics["accuracy"]),
        "f1_macro": float(test_metrics["f1_macro"]),
        "precision_macro": float(test_metrics["precision_macro"]),
        "recall_macro": float(test_metrics["recall_macro"]),
        "epoch_history": epoch_history,
    }

    if SAVE_PER_RUN_PREDICTIONS:
        result["y_true"] = test_metrics["y_true"].tolist()
        result["y_pred"] = test_metrics["y_pred"].tolist()

    if SAVE_MODELS:
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        save_path = MODEL_ROOT / f"model_seed_{run_seed}_{int(time.time())}.keras"
        model.save(save_path)
        result["saved_model_path"] = str(save_path)

    return result


# =============================================================================
# 6) CONFIGURATION ENUMERATION
# =============================================================================

def get_activation_map() -> Dict[str, Callable[[tf.Tensor], tf.Tensor]]:
    acts = {}
    if INCLUDE_STANDARD_ACTIVATIONS:
        acts.update(STANDARD_ACTIVATIONS)
    if INCLUDE_FRACTAL_ACTIVATIONS:
        acts.update(FRACTAL_ACTIVATIONS)
    return acts


def build_experiment_plan() -> List[Dict[str, Any]]:
    """
    Enumerate all experiment configurations.
    """
    configs: List[Dict[str, Any]] = []
    activation_map = get_activation_map()

    for activation_name in activation_map.keys():
        # Standard baselines
        if INCLUDE_BASELINE_OPTIMIZERS:
            for opt_name in BASELINE_OPTIMIZERS:
                configs.append(
                    {
                        "optimizer_name": opt_name,
                        "activation_name": activation_name,
                        "vderiv": 1.0,
                        "history_size": None,
                        "adaptation_mode": None,
                        "schedule_type": None,
                    }
                )

        # Herrera-style
        if INCLUDE_HERRERA_OPTIMIZERS:
            for opt_name in HERRERA_OPTIMIZERS:
                for vderiv in HERRERA_VDERIVS:
                    configs.append(
                        {
                            "optimizer_name": opt_name,
                            "activation_name": activation_name,
                            "vderiv": float(vderiv),
                            "history_size": None,
                            "adaptation_mode": None,
                            "schedule_type": None,
                        }
                    )

        # Memory-based
        if INCLUDE_MEMORY_OPTIMIZERS:
            for opt_name in MEMORY_OPTIMIZERS:
                for vderiv in MEMORY_VDERIVS:
                    for history_size in MEMORY_HISTORY_SIZES:
                        configs.append(
                            {
                                "optimizer_name": opt_name,
                                "activation_name": activation_name,
                                "vderiv": float(vderiv),
                                "history_size": int(history_size),
                                "adaptation_mode": None,
                                "schedule_type": None,
                            }
                        )

        # Variable-order
        if INCLUDE_VARIABLE_ORDER_OPTIMIZERS:
            for opt_name in VARIABLE_ORDER_OPTIMIZERS:
                for mode in VAR_ORDER_MODES:
                    if mode in {"schedule", "hybrid_transition"}:
                        for sched in VAR_ORDER_SCHEDULE_TYPES:
                            configs.append(
                                {
                                    "optimizer_name": opt_name,
                                    "activation_name": activation_name,
                                    "vderiv": float(VAR_ORDER_INIT),
                                    "history_size": int(VAR_ORDER_HISTORY_SIZE),
                                    "adaptation_mode": mode,
                                    "schedule_type": sched,
                                }
                            )
                    else:
                        configs.append(
                            {
                                "optimizer_name": opt_name,
                                "activation_name": activation_name,
                                "vderiv": float(VAR_ORDER_INIT),
                                "history_size": int(VAR_ORDER_HISTORY_SIZE),
                                "adaptation_mode": mode,
                                "schedule_type": None,
                            }
                        )

    return configs


# =============================================================================
# 7) RESULT WRITING
# =============================================================================

def config_to_filename(cfg: Dict[str, Any]) -> str:
    parts = [
        cfg["optimizer_name"],
        cfg["activation_name"],
    ]
    if cfg["history_size"] is not None:
        parts.append(f"h{cfg['history_size']}")
    if cfg["adaptation_mode"] is not None:
        parts.append(cfg["adaptation_mode"])
    if cfg["schedule_type"] is not None:
        parts.append(cfg["schedule_type"])
    parts.append(f"v{str(cfg['vderiv']).replace('.', 'p')}")
    return "__".join(parts) + ".json"


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    accs = [r["accuracy"] for r in runs]
    f1s = [r["f1_macro"] for r in runs]
    train_times = [r["training_time_seconds"] for r in runs]
    test_times = [r["test_time_seconds"] for r in runs]
    return {
        "avg_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=0)),
        "max_accuracy": float(np.max(accs)),
        "min_accuracy": float(np.min(accs)),
        "avg_f1_macro": float(np.mean(f1s)),
        "avg_training_time_seconds": float(np.mean(train_times)),
        "avg_test_time_seconds": float(np.mean(test_times)),
    }


def write_result_json(
    dataset_name: str,
    cfg: Dict[str, Any],
    run_results: List[Dict[str, Any]],
) -> None:
    """
    Write one JSON file per configuration in a structure compatible with the
    uploaded report script.
    """
    out_dir = RESULTS_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary = summarize_runs(run_results)

    # Keep the report-compatible schema:
    # data["vderivs"] -> list of objects with vderiv and results
    payload = {
        "dataset": dataset_name,
        "optimizer": cfg["optimizer_name"],
        "activation": cfg["activation_name"],
        "history_size": cfg["history_size"],
        "adaptation_mode": cfg["adaptation_mode"],
        "schedule_type": cfg["schedule_type"],
        "n_runs": len(run_results),
        "vderivs": [
            {
                "vderiv": float(cfg["vderiv"]),
                "avg accuracy": run_summary["avg_accuracy"],
                "std accuracy": run_summary["std_accuracy"],
                "avg f1 macro": run_summary["avg_f1_macro"],
                "avg training time seconds": run_summary["avg_training_time_seconds"],
                "avg test time seconds": run_summary["avg_test_time_seconds"],
                "results": [
                    {
                        "run_seed": RUN_SEEDS[i],
                        "accuracy": float(r["accuracy"]),
                        "f1_macro": float(r["f1_macro"]),
                        "precision_macro": float(r["precision_macro"]),
                        "recall_macro": float(r["recall_macro"]),
                        "training_time_seconds": float(r["training_time_seconds"]),
                        "test_time_seconds": float(r["test_time_seconds"]),
                        "epochs_completed": int(r["epochs_completed"]),
                        "best_val_loss": float(r["best_val_loss"]),
                    }
                    for i, r in enumerate(run_results)
                ],
            }
        ],
    }

    if SAVE_PER_RUN_HISTORY_JSON:
        payload["run_histories"] = [
            {
                "run_seed": RUN_SEEDS[i],
                "epoch_history": r["epoch_history"],
            }
            for i, r in enumerate(run_results)
        ]

    out_file = out_dir / config_to_filename(cfg)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[SAVED] {out_file}")


# =============================================================================
# 8) MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiments() -> None:
    set_global_seed(GLOBAL_SEED)

    activation_map = get_activation_map()
    experiment_plan = build_experiment_plan()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Fractional optimizer × fractal activation experiments")
    print("=" * 88)
    print(f"Datasets: {DATASETS_TO_RUN}")
    print(f"Activations: {list(activation_map.keys())}")
    print(f"Total configurations per dataset: {len(experiment_plan)}")
    print(f"Runs per configuration: {len(RUN_SEEDS)}")
    print("=" * 88)

    for dataset_name in DATASETS_TO_RUN:
        print(f"\n[DATASET] {dataset_name}")
        print("-" * 88)

        for cfg_idx, cfg in enumerate(experiment_plan, start=1):
            print(
                f"\n[CONFIG {cfg_idx}/{len(experiment_plan)}] "
                f"optimizer={cfg['optimizer_name']} "
                f"activation={cfg['activation_name']} "
                f"vderiv={cfg['vderiv']} "
                f"history={cfg['history_size']} "
                f"mode={cfg['adaptation_mode']} "
                f"schedule={cfg['schedule_type']}"
            )

            run_results = []

            for run_seed in RUN_SEEDS:
                print(f" [RUN seed={run_seed}]")

                dataset_bundle = prepare_dataset(dataset_name, seed=run_seed)
                activation_fn = activation_map[cfg["activation_name"]]

                optimizer = build_optimizer(
                    optimizer_name=cfg["optimizer_name"],
                    vderiv=cfg["vderiv"],
                    history_size=cfg["history_size"] if cfg["history_size"] is not None else 6,
                    adaptation_mode=cfg["adaptation_mode"] if cfg["adaptation_mode"] else "ema_smoothed_gradient_variability",
                    schedule_type=cfg["schedule_type"] if cfg["schedule_type"] else "cosine",
                )

                result = train_one_run(
                    dataset_bundle=dataset_bundle,
                    optimizer=optimizer,
                    activation_fn=activation_fn,
                    activation_name=cfg["activation_name"],
                    run_seed=run_seed,
                )

                run_results.append(result)

                print(
                    f"   accuracy={result['accuracy']:.5f} "
                    f"f1={result['f1_macro']:.5f} "
                    f"train_time={result['training_time_seconds']:.3f}s "
                    f"test_time={result['test_time_seconds']:.3f}s "
                    f"epochs={result['epochs_completed']}"
                )

            write_result_json(dataset_name, cfg, run_results)

    print("\nDone.")


# =============================================================================
# 9) OPTIONAL: EXPERIMENTAL DESIGN NOTES
# =============================================================================

def print_experimental_design_notes() -> None:
    """
    Print a compact description of the intended comparisons.
    """
    print("\nExperimental design:")
    print("1. Baseline comparison:")
    print("   Compare standard optimizers with Herrera-style, memory-based, and variable-order optimizers.")
    print("2. Activation comparison:")
    print("   Compare standard activations against fractal activations under the same optimizer conditions.")
    print("3. Interaction study:")
    print("   Identify whether certain optimizer families work better with certain fractal activations.")
    print("4. Cost study:")
    print("   Compare average training and test time overheads.")
    print("5. Adaptive-order ablation:")
    print("   Compare fixed-order memory-based Adam to variable-order modes:")
    print("      - ema_smoothed_gradient_variability")
    print("      - schedule")
    print("      - hybrid_transition")


# =============================================================================
# 10) SCRIPT ENTRY
# =============================================================================

if __name__ == "__main__":
    print_experimental_design_notes()
    run_experiments()
