
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from numpy.random import seed
import numpy as np
import json
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import fractal_activation_functions as fractal
import random
from tensorflow.keras import backend as K
K.clear_session()

# TensorFlow and CUDA/cuDNN info
print("TensorFlow version:", tf.__version__)
print("Is TensorFlow using cuDNN:", tf.test.is_built_with_cuda())

# List all physical devices
print("\nList of Physical Devices:")
physical_devices = tf.config.list_physical_devices()
for device in physical_devices:
    print(f"- {device.device_type}: {device.name}")

# Load dataset
def load_dataset(data_name):
    data_sk = fetch_openml(name=data_name, version=1, as_frame=True, parser='auto')
    X = data_sk.data
    y = data_sk.target

    # Encode non-numeric features
    if isinstance(X, pd.DataFrame):
        X = encode_non_numeric_features(X)
    if isinstance(y, pd.Series) and y.dtype == 'object' or 'category':
        y = LabelEncoder().fit_transform(y)  # encode non-numeric labels
    else:
        y = y.to_numpy(dtype='int32')

    return X, y

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "f1_score": f1_score(y_true, y_pred, average='macro'),
    }
    return metrics

# Encode non-numeric features
def encode_non_numeric_features(df):
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[column].unique()
        value_to_number = {value: idx / (len(unique_values) - 1) for idx, value in enumerate(unique_values)}
        df.loc[:, column] = df[column].map(value_to_number)
    return df

# Plot training loss and accuracy curves
def plot_loss_curve(history, data_name, optimizer_name, activation_name, neurons, batch, test_acc):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8), dpi=200)
    ax1.plot(history.history['val_accuracy'], label="validation accuracy")
    ax1.plot(history.history['accuracy'], label="training accuracy")
    ax2.plot(history.history['val_loss'], label="validation loss")
    ax2.plot(history.history['loss'], label="training loss")
    ax1.set_ylabel('validation accuracy')
    ax2.set_ylabel('validation loss')
    ax2.set_xlabel('epochs')
    ax1.set_title('accuracy and loss for ' + data_name + " with " + optimizer_name + ", " + activation_name + ", " + str(neurons) + ' neurons, ' + str(batch) + ' batch size')
    ax2.set_title('test accuracy = ' + str(test_acc))
    ax1.legend()
    ax2.legend()
    fig.tight_layout(pad=1)
    plt.show()

# Split and scale dataset
def split_scale(X, y, n):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, X_test, y_train, y_test = dc(train_test_split(X, y, test_size=0.3, random_state=n, shuffle=True))
    X_train = dc(scaler.fit_transform(X_train))
    X_test = dc(scaler.transform(X_test))
    return X_train, X_test, y_train, y_test

# Load JSON data
def load_json(target_path):
    with open(target_path, "r") as json_file:
        json_data = json.load(json_file)
    return json_data

# Save JSON data
def save_json(target_path, data):
    with open(target_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Saved log to ", target_path)

# Binary classification model
def binary_model(n_features, n_classes, neurons, activation):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=n_features),
        tf.keras.layers.Dense(n_classes, activation='sigmoid', kernel_initializer=initializer)
    ])
    return model

# Multi-class classification model
def groups_model(n_features, n_classes, neurons, activation):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=n_features),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(neurons*2, activation=activation, kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer=initializer)
    ])
    return model

# Main function for training and testing
def train_and_test_(results_dir, data_name, neurons, batch, ep, act, op, run_nrs=1, fr_vderivs=[1.0], seed_val=None):
    #
    if seed_val is None:
        seed_val = int(time.time())  # Generate a random seed based on system time
    random.seed(seed_val)
    np.random.seed(seed_val)
    tf.random.set_seed(seed_val)

    print(f"Running with seed: {seed_val}")

    ns = np.arange(seed_val, seed_val + run_nrs, dtype='int')
    activation_name = act[0]
    optimizer_name = op[0]

    # Dataset preprocessing
    X, y = dc(load_dataset(data_name))
    n_classes = len(set(y))
    n_features = X.shape[1]

    data1 = {
        "dataset": data_name,
        "neurons": neurons,
        "batch size": batch,
        "epochs": ep,
        "activation": activation_name,
        "test runs": run_nrs,
        "optimizer": optimizer_name,
        "vderivs": [],
    }

    # Prepare file structure and check for previous results
    file_name = optimizer_name + "_" + activation_name + "_" + data_name + ".json"
    folder_str = results_dir + "/" + "results_" + str(run_nrs) + "_runs/" + data_name
    target_path = folder_str + "/" + file_name
    folder_path = Path(folder_str)
    folder_path.mkdir(parents=True, exist_ok=True)

    if os.path.exists(target_path):
        existing_data = load_json(target_path)
        keys = ["dataset", "neurons", "batch size", "epochs", "activation", "test runs", "optimizer"]
        for key in keys:
            if not existing_data[key] == data1[key]:
                raise Exception(f"Existing data does not match the current run's data: {key}: {existing_data[key]} vs {data1[key]}")
    else:
        save_json(target_path, data1)

    vderivs = fr_vderivs if optimizer_name[0] == 'F' else [1.0]
    for v in vderivs:
        test_accs, test_losses, training_times, testing_times = [], [], [], []
        test_precisions, test_recalls, test_f1s, training_accs = [], [], [], []  # To track precision, recall, F1
        best_training_acc = 0  # To track best training accuracy

        data_v = {"vderiv": float(v), "avg accuracy": 0, "avg loss": 0, "avg time": 0, "results": []}

        for n in ns:
            tf.keras.backend.clear_session()
            # Define the model
            if n_classes == 2:
                model = binary_model(n_features, n_classes, neurons, act_fn[1])

            elif n_classes > 2:
                model = groups_model(n_features, n_classes, neurons, act_fn[1])

            print(f'All seeds: {ns}')
            print(f'Current Seed: {n}')
            seed(n)
            np.random.seed(n)
            tf.random.set_seed(n)

            # Split and scale data
            X_train, X_test, y_train, y_test = dc(split_scale(X, y, n))

            # Compile the model
            optimizer_class = op[1](vderiv=v, learning_rate=op[2]) if optimizer_name[0] == 'F' else op[1](learning_rate=op[2])
            model.compile(optimizer=optimizer_class, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            start_time_train = time.time()
            history = model.fit(X_train, y_train, epochs=ep, batch_size=batch, validation_data=(X_test, y_test))
            end_time_train = time.time()
            training_time = end_time_train - start_time_train
            training_times.append(training_time)

            # Update best training accuracy
            best_training_acc = max(history.history['accuracy'])
            training_accs.append(best_training_acc)

            # Evaluate the model
            test_loss, test_acc = model.evaluate(X_test, y_test)

            test_accs.append(test_acc)
            test_losses.append(test_loss)

            # Get predictions and calculate precision, recall, and F1
            start_time_test = time.time()
            y_pred = model.predict(X_test)
            end_time_test = time.time()
            testing_time = end_time_test - start_time_test
            y_pred_classes = np.argmax(y_pred, axis=1)
            metrics = calculate_metrics(y_test, y_pred_classes)
            testing_times.append(testing_time)

            test_precisions.append(metrics['precision'])
            test_recalls.append(metrics['recall'])
            test_f1s.append(metrics['f1_score'])

            # Log the results
            log = f"RUN #{n}\n dataset = {data_name}\n neurons = {neurons}\n batch size = {batch}\n activation = {activation_name}\n vderiv = {v}\n optimizer = {optimizer_name}\n test accuracy = {test_acc}\n test loss = {test_loss}\n training time = {training_time}\n testing time = {testing_time}\n precision = {metrics['precision']}\n recall = {metrics['recall']}\n f1 score = {metrics['f1_score']}"
            print(log)

            data_run = {
                "run #": int(n),
                "accuracy": float(test_acc),
                "loss": float(test_loss),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "f1_score": float(metrics['f1_score']),
                "best_training_acc": float(best_training_acc),
                "training time": float(training_time),
                "testing time": float(testing_time),
            }
            data_v['results'].append(data_run)
            tf.keras.backend.clear_session()
            os.system('pkill -f "tensorboard"')
            del history, y_pred, X_train, X_test, y_train, y_test, model  # as soon as metrics have been extracted

        # Find averages
        avg_acc = np.average(test_accs)
        avg_loss = np.average(test_losses)
        avg_training_time = np.average(training_times)
        avg_testing_time = np.average(testing_times)
        avg_precision = np.average(test_precisions)
        avg_recall = np.average(test_recalls)
        avg_f1 = np.average(test_f1s)
        avg_best_training_acc = np.average(training_accs)

        data_v['avg accuracy'] = avg_acc
        data_v['avg loss'] = avg_loss
        data_v['avg training time'] = avg_training_time
        data_v['avg testing time'] = avg_testing_time
        data_v['avg precision'] = avg_precision
        data_v['avg recall'] = avg_recall
        data_v['avg f1'] = avg_f1
        data_v['avg best training accuracy'] = avg_best_training_acc  # Track the best training accuracy

        print(f"Average accuracy (over {run_nrs} runs) for {optimizer_name} = {avg_acc}")

        # Save the results as JSON
        json_data = load_json(target_path)
        json_data['vderivs'].append(data_v)
        save_json(target_path, json_data)

    return "Finished"

# Your variables, datasets, and loops are the same as before, so I have omitted them.

# Variables
optimizers = [
                ('sgd', tf.keras.optimizers.SGD, 0.01),
              ('rmsprop', tf.keras.optimizers.RMSprop, 0.001),
              ('adam', tf.keras.optimizers.Adam, 0.001),
              ('adagrad', tf.keras.optimizers.Adagrad, 1.0),
              ('adadelta', tf.keras.optimizers.Adadelta, 1.0),
              ]

activation_functions = [
    ('weierstrass', fractal.weierstrass_function_tf),
    ('weierstrass_mandelbrot_xpsin', fractal.weierstrass_mandelbrot_function_xpsin),
    ('weierstrass_mandelbrot_xsinsquared', fractal.weierstrass_mandelbrot_function_xsinsquared),
    ('weierstrass_mandelbrot_relupsin', fractal.weierstrass_mandelbrot_function_relupsin),
    ('weierstrass_mandelbrot_tanhpsin', fractal.weierstrass_mandelbrot_function_tanhpsin),
    ('blancmange', fractal.modulated_blancmange_curve),
    ('decaying_cosine', fractal.decaying_cosine_function_tf),
    ('modified_weierstrass_tanh', fractal.modified_weierstrass_function_tanh),
    ('modified_weierstrass_ReLU', fractal.modified_weierstrass_function_relu),
    ('relu', 'relu'),
    ('sigmoid', 'sigmoid'),
    ('tanh', 'tanh'),]

# ------------------------------------------------------------------
#  Dataset  |  n-instances | n-features | n-classes | NN-params
#           |              |            |  (target) |  (neurons, batch, epochs)
# ------------------------------------------------------------------
data_with_params = [

    # ---- original block ---------------------------------------------------
    ("diabetes", 64, 32, 30),  # # diabetes diagnosis records, binary, 9 features, 768 instances
    ("tic-tac-toe",                       64,  32, 25),   #  958 ×  9 → 2 cls
    ("vertebra-column",                   32,  16, 30),   #  310 ×  6 → 2 cls
    ("vehicle",                          128,  32, 25),   #  846 × 18 → 4 cls
    ("climate-model-simulation-crashes",  32,  32, 30),   #  540 × 21 → 2 cls
    ("iris",                              32,  16, 30),   #  150 ×  4 → 3 cls
    ("wine",                              64,  32, 30),   #  178 × 13 → 3 cls
    ("glass",                             64,  32, 30),   #  214 ×  9 → 6 cls
    ("ionosphere",                        128,  32, 30),   #  351 × 34 → 2 cls
    ("seeds",                             64,  32, 30),   #  210 ×  7 → 3 cls
]

runs = 40 #
seed_val = 238974 #201 still needs to be run
vderivs = np.round(np.arange(0.1, 2.0, 0.1), decimals=1)
target_dir = f"results_may12_{runs}runs_seedval{seed_val}_2025_nf2"

#  Loop through datasets, optimizers and vderiv values, repeating each run n times
for data in data_with_params:
    for act_fn in activation_functions:
        for optimizer in optimizers:
            train_and_test_(target_dir, data[0], data[1], data[2], data[3], act_fn, optimizer, runs, vderivs, seed_val=seed_val)
























