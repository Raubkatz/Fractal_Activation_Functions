import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tnp # start using tnp instead of numpy or math library
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.experimental import numpy as tnp


# Custom activation function: Revised Modulated Blancmange Curve
def modulated_blancmange_curve(x, n_terms=30, a=0.75):
    y = tf.zeros_like(x)
    for n in range(n_terms):
        factor = 2 ** n
        modulation = tf.tanh(a * factor * x)  # Modulating factor using tanh
        ax = a * tf.sqrt(tf.abs(x))
        y += modulation * tf.abs(x * factor % 2 - 1 * ax) / factor
    return y / 2


# Blended function with tanh and cosine components with exponential decay using TensorFlow
def decaying_cosine_function_tf(x, a=0.5, b=3, c=0.5, d=2, n_terms=75,  zeta=0.2666):
    # Mirrored function using TensorFlow
    def mirrored_function_tf(x):
        return tf.where(x < 0, tf.ones_like(x), -tf.ones_like(x))

    #if not (0 < a < 1):
    #    raise ValueError("Parameter 'a' must be in the interval (0, 1).")

    #if b <= 0:
    #    raise ValueError("Parameter 'b' must be positive.")

    #if c <= 0:
    #    raise ValueError("Parameter 'c' must be positive.")

    #if d <= 0:
    #    raise ValueError("Parameter 'd' must be positive.")

    #if n_terms <= 0:
    #    raise ValueError("Parameter 'n_terms' must be a positive integer.")

    # Initialize the output tensor
    w = tf.zeros_like(x)

    for n in range(n_terms):
        # Exponentially decaying cosine part
        decay = tf.exp(-tf.abs(x) * 0.5)
        # Sum the components: tanh and decaying cosine with mirrored behavior
        w += zeta * (0.05 * tf.tanh(tnp.pi * x) + (c ** n) * tf.cos((d ** n) * tnp.pi * x) * decay * mirrored_function_tf(x))

    return w


# Custom activation function: Modulated Weierstrass function using tnp
def modified_weierstrass_function_tanh(x, a=0.5, b=1.5, n_terms=100):
    #if not (0 < a < 1):
    #    raise ValueError("Parameter 'a' must be in the interval (0, 1).")

    #if b % 2 == 0 or b <= 0:
    #    raise ValueError("Parameter 'b' must be a positive odd integer.")

    #if n_terms <= 0:
    #    raise ValueError("Parameter 'n_terms' must be a positive integer.")

    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Calculate the modified Weierstrass function in float64
    w = tf.zeros_like(x, dtype=tf.float64)
    for n in range(n_terms):
        w += ((-1) ** n) * (a ** n) * tf.cos((b ** n) * tnp.pi * x)

    # Blend with the hyperbolic tangent function, using tnp.exp and tnp.abs
    combined_function = w * tnp.exp(-0.75 * tnp.abs(x)) + tnp.tanh(x)

    # Finally, convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(combined_function, tf.float32)


# Custom activation function: Modulated Weierstrass function with ReLU using tnp
def modified_weierstrass_function_relu(x, a=0.5, b=3, n_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Calculate the modified Weierstrass function in float64
    w = tf.zeros_like(x, dtype=tf.float64)
    for n in range(n_terms):
        w += ((-1) ** n) * (a ** n) * tf.cos((b ** n) * tnp.pi * x)

    # Blend with the ReLU function, using tnp.exp and tnp.abs
    combined_function = w * tnp.exp(-0.75 * tnp.abs(x)) + tf.nn.relu(x)  # Replaced tanh with ReLU

    # Finally, convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(combined_function, tf.float32)


def weierstrass_mandelbrot_function_xsinsquared(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function
    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * x * (tnp.sin(2 * tnp.pi * lambda_val**k * x) ** 2)
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)


def weierstrass_mandelbrot_function_xpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function
    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * (x + (tnp.sin(2 * tnp.pi * lambda_val**k * x)))
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)

def weierstrass_mandelbrot_function_relupsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function with ReLU(x) + sin(...)
    for k in range(1, num_terms):
        relu_term = tf.nn.relu(x)
        sin_term = tnp.sin(2 * tnp.pi * lambda_val**k * x)
        term = (2**(-k * gamma)) * (relu_term + sin_term)
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)

def weierstrass_mandelbrot_function_tanhpsin(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    M_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass-Mandelbrot function with ReLU(x) + sin(...)
    for k in range(1, num_terms):
        tanh_term = tnp.tanh(x)
        sin_term = tnp.sin(2 * tnp.pi * lambda_val**k * x)
        term = (2**(-k * gamma)) * (tanh_term + sin_term)
        M_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(M_x, tf.float32)


# Custom activation function: Classical Weierstrass function using TensorFlow
def weierstrass_function_tf(x, gamma=0.5, lambda_val=2, num_terms=100):
    # Ensure x is float64 for all calculations
    x = tf.cast(x, tf.float64)

    # Initialize the function value to zeros
    W_x = tf.zeros_like(x, dtype=tf.float64)

    # Compute the Weierstrass function
    for k in range(1, num_terms):
        term = (2**(-k * gamma)) * tnp.sin(2 * tnp.pi * lambda_val**k * x)
        W_x += term

    # Convert the result back to float32 for compatibility with TensorFlow layers
    return tf.cast(W_x, tf.float32)



