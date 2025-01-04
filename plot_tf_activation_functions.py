from fractal_activation_functions import *
import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tnp
import matplotlib.pyplot as plt

from legacy_plot_fractal_Activations_tf import y_modified_weierstrass

# Load your functions

x = tnp.linspace(-5, 5, 1000)

# Call the activation functions
y_blancmange = modulated_blancmange_curve(x)
y_decaying_cosine = decaying_cosine_function_tf(x)
y_fav_fractal = new_fav_fractal_function(x)
y_weierstrass = weierstrass_function_tf(x)
y_weierstrass_relu = modified_weierstrass_function_relu(x)
y_weierstrass_tanh = modified_weierstrass_function_tanh(x)
y_weierstrass_mandelbrot_xsinsquared = weierstrass_mandelbrot_function_xsinsquared(x)
y_weierstrass_mandelbrot_xpsin = weierstrass_mandelbrot_function_xpsin(x)
y_weierstrass_mandelbrot_relupsin = weierstrass_mandelbrot_function_relupsin(x)
y_weierstrass_mandelbrot_tanhpsin = weierstrass_mandelbrot_function_tanhpsin(x)

# Create subplots to display all the functions
plt.figure(figsize=(12, 12))

# Plot modulated Blancmange Curve
plt.subplot(5, 2, 1)
plt.plot(x, y_blancmange.numpy(), label="Modulated Blancmange Curve")
plt.title("Modulated Blancmange Curve")
plt.grid(True)
plt.legend()

# Plot decaying cosine function
plt.subplot(5, 2, 2)
plt.plot(x, y_decaying_cosine.numpy(), label="Decaying Cosine Function (TF)", color='orange')
plt.title("Decaying Cosine Function")
plt.grid(True)
plt.legend()

# Plot favorite fractal function
plt.subplot(5, 2, 3)
plt.plot(x, y_fav_fractal.numpy(), label="New Fav Fractal Function", color='green')
plt.title("New Fav Fractal Function")
plt.grid(True)
plt.legend()

# Plot modified Weierstrass function
plt.subplot(5, 2, 4)
plt.plot(x, y_weierstrass.numpy(), label="Weierstrass Function", color='purple')
plt.title("Weierstrass Function")
plt.grid(True)
plt.legend()

# Plot modified Weierstrass function with ReLU
plt.subplot(5, 2, 5)
plt.plot(x, y_weierstrass_relu.numpy(), label="Modified Weierstrass Function ReLU", color='purple')
plt.title("Modified Weierstrass Function ReLU")
plt.grid(True)
plt.legend()

# Plot Weierstrass-Mandelbrot function
plt.subplot(5, 2, 6)
plt.plot(x, y_weierstrass_mandelbrot_xsinsquared.numpy(), label="Weierstrass-Mandelbrot Function x*sin²", color='red')
plt.title("Weierstrass-Mandelbrot x*sin² Function")
plt.grid(True)
plt.legend()

# Plot Weierstrass-Mandelbrot function
plt.subplot(5, 2, 7)
plt.plot(x, y_weierstrass_mandelbrot_xpsin.numpy(), label="Weierstrass-Mandelbrot Function x+sin", color='teal')
plt.title("Weierstrass-Mandelbrot x+sin Function")
plt.grid(True)
plt.legend()


# Plot Weierstrass-Mandelbrot function
plt.subplot(5, 2, 8)
plt.plot(x, y_weierstrass_mandelbrot_relupsin.numpy(), label="Weierstrass-Mandelbrot Function relu+sin", color='teal')
plt.title("Weierstrass-Mandelbrot relu+sin Function")
plt.grid(True)
plt.legend()

# Plot Weierstrass-Mandelbrot function
plt.subplot(5, 2, 9)
plt.plot(x, y_weierstrass_mandelbrot_tanhpsin.numpy(), label="Weierstrass-Mandelbrot Function tanh+sin", color='teal')
plt.title("Weierstrass-Mandelbrot tanh+sin Function")
plt.grid(True)
plt.legend()

# Plot Weierstrass-Mandelbrot function
plt.subplot(5, 2, 10)
plt.plot(x, y_weierstrass_tanh.numpy(), label="Modified Weierstrass tanh Function", color='teal')
plt.title("Modified Weierstrass tanh Function")
plt.grid(True)
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# Repeat for a different range of x values
x = tnp.linspace(-10, 10, 1000)

# Call the activation functions again for the new range
y_blancmange = modulated_blancmange_curve(x)
y_decaying_cosine = decaying_cosine_function_tf(x)
y_fav_fractal = new_fav_fractal_function(x)
y_weierstrass = modified_weierstrass_function(x)
y_weierstrass_relu = modified_weierstrass_function_with_relu(x)
y_weierstrass_mandelbrot = weierstrass_mandelbrot_function(x)

# Create subplots to display all the functions for the new range
plt.figure(figsize=(12, 12))

# Plot modulated Blancmange Curve
plt.subplot(3, 2, 1)
plt.plot(x, y_blancmange.numpy(), label="Modulated Blancmange Curve")
plt.title("Modulated Blancmange Curve")
plt.grid(True)
plt.legend()

# Plot decaying cosine function
plt.subplot(3, 2, 2)
plt.plot(x, y_decaying_cosine.numpy(), label="Decaying Cosine Function (TF)", color='orange')
plt.title("Decaying Cosine Function")
plt.grid(True)
plt.legend()

# Plot favorite fractal function
plt.subplot(3, 2, 3)
plt.plot(x, y_fav_fractal.numpy(), label="New Fav Fractal Function", color='green')
plt.title("New Fav Fractal Function")
plt.grid(True)
plt.legend()

# Plot modified Weierstrass function
plt.subplot(3, 2, 4)
plt.plot(x, y_weierstrass.numpy(), label="Modified Weierstrass Function", color='purple')
plt.title("Modified Weierstrass Function")
plt.grid(True)
plt.legend()

# Plot modified Weierstrass function with ReLU
plt.subplot(3, 2, 5)
plt.plot(x, y_weierstrass_relu.numpy(), label="Modified Weierstrass Function ReLU", color='purple')
plt.title("Modified Weierstrass Function ReLU")
plt.grid(True)
plt.legend()

# Plot Weierstrass-Mandelbrot function
plt.subplot(3, 2, 6)
plt.plot(x, y_weierstrass_mandelbrot.numpy(), label="Weierstrass-Mandelbrot Function", color='red')
plt.title("Weierstrass-Mandelbrot Function")
plt.grid(True)
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
