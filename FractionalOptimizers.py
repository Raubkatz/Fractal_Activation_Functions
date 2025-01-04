#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
Last update Sat Nov 19 23:34:54 2022
@author: Oscar Herrera
Reference:
- [Herrera-Alcántara O., 2022]
Citation:
Herrera-Alcántara, O. Fractional Derivative Gradient-Based Optimizers for Neural Networks and Human Activity Recognition. Appl. Sci. 2022, 12, 9264. 
https://doi.org/10.3390/app12189264 
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow import keras
import math

# IMPORTS
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#from keras.datasets import mnist
#from tensorflow.keras import utils
#from keras.utils import to_categorical
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, LeakyReLU
#from keras.layers.noise import AlphaDropout
#from keras.utils.generic_utils import get_custom_objects
#from keras import backend as K
from keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adam

from numpy.random import seed

##
'''
Fractional gradient required definitions
'''
#vderiv = 1.0  #the fractional order tested for real values - interval [0, 2)
#uno_menos_v = 1 - vderiv
#dos_menos_v = 2 - vderiv
#gamma_2_nu = math.gamma(2 - vderiv)

##START Modification of original SGD to get FSGD the fractional derivative SGD version.
#from keras.optimizer_experimental import optimizer
#from keras.utils import generic_utils






import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Optimizer
import math

class RaubFSGD(optimizers.experimental.Optimizer):
    r"""Fractional Gradient Descent (FSGD) optimizer with momentum and Nesterov acceleration.

    This optimizer implements the Grünwald-Letnikov discretization for fractional derivatives,
    introducing non-locality into the gradient updates by considering a history of past gradients.

    Attributes:
        learning_rate: A float, a schedule that is a `tf.keras.optimizers.schedules.LearningRateSchedule`,
          or a callable that takes no arguments and returns the actual value to use. The learning rate.
        momentum: Float hyperparameter >= 0 that accelerates gradient descent in the relevant direction
          and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent.
        nesterov: Boolean. Whether to apply Nesterov momentum. Defaults to `False`.
        vderiv: The fractional derivative order (alpha), tested for real values in the interval [0, 2).
        history_size: Integer. The number of past gradients to consider in the fractional derivative approximation.
        **kwargs: Additional keyword arguments.

    Usage:

    ```python
    opt = FSGD(learning_rate=0.1, vderiv=0.5, history_size=5)
    ```
    """

    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False,
                 amsgrad=False,
                 clipnorm=None,
                 clipvalue=None,
                 global_clipnorm=None,
                 use_ema=False,
                 ema_momentum=0.99,
                 ema_overwrite_frequency=100,
                 jit_compile=False,
                 name='FSGD',
                 vderiv=1.5,
                 history_size=10,
                 **kwargs):
        super(FSGD, self).__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.vderiv = vderiv
        self.history_size = history_size
        if not 0 <= momentum <= 1:
            raise ValueError('`momentum` must be between 0 and 1.')

    def build(self, var_list):
        """Initialize optimizer variables.

        FSGD optimizer has variables for momentum and gradient histories.

        Args:
          var_list: List of model variables to build optimizer variables on.
        """
        super().build(var_list)
        if hasattr(self, '_built') and self._built:
            return
        self.momentums = []
        if self.momentum != 0:
            for var in var_list:
                self.momentums.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name='m'))
        # Initialize gradient histories
        self._grad_histories = {}
        for var in var_list:
            self._grad_histories[self._var_key(var)] = [
                tf.zeros_like(var) for _ in range(self.history_size)
            ]
        self._built = True

    def compute_coefficients(self, alpha, n_terms):
        """Compute the coefficients for the Grünwald-Letnikov discretization."""
        coeffs = [1.0]
        for k in range(1, n_terms):
            coeff = (1 - (alpha + 1) / k) * coeffs[k - 1]
            coeffs.append(coeff)
        return coeffs

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        var_key = self._var_key(variable)
        if var_key not in self._index_dict:
            raise KeyError(f'Optimizer cannot recognize variable {variable.name}, '
                           f'this usually means you are calling an optimizer '
                           f'previously used on a different model. Please try '
                           f'creating a new optimizer instance.')

        lr = tf.cast(self.learning_rate, variable.dtype)
        m = None
        if self.momentum != 0:
            momentum = tf.cast(self.momentum, variable.dtype)
            m = self.momentums[self._index_dict[var_key]]

        # Start of fractional derivative implementation
        grad_history = self._grad_histories[var_key]

        # Update gradient history: remove oldest, append current gradient
        grad_history.pop(0)
        grad_history.append(gradient)

        # Compute coefficients for the fractional derivative
        coeffs = self.compute_coefficients(self.vderiv, self.history_size)

        # Compute fractional gradient using the Grünwald-Letnikov discretization
        frac_gradient = tf.zeros_like(gradient)
        for k in range(self.history_size):
            coeff = coeffs[k]
            frac_gradient += coeff * grad_history[-(k + 1)]

        # Proceed with momentum and Nesterov acceleration as per standard optimizer
        if isinstance(frac_gradient, tf.IndexedSlices):
            # Handle sparse gradients
            add_value = tf.IndexedSlices(-frac_gradient.values * lr, frac_gradient.indices)
            if m is not None:
                m.assign(m * momentum)
                m.scatter_add(add_value)
                if self.nesterov:
                    variable.scatter_add(add_value)
                    variable.assign_add(m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.scatter_add(add_value)
        else:
            # Handle dense gradients
            if m is not None:
                m.assign(m * momentum - frac_gradient * lr)
                if self.nesterov:
                    variable.assign_add(m * momentum - frac_gradient * lr)
                else:
                    variable.assign_add(m)
            else:
                variable.assign_add(-frac_gradient * lr)
        # End of fractional derivative implementation

    def get_config(self):
        config = super(FSGD, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter(self._learning_rate),
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'vderiv': self.vderiv,
            'history_size': self.history_size,
        })
        return config






# comment to avoid error "Custom>FSGD has already been registered to <class '__main__.FSGD'>"   @generic_utils.register_keras_serializable()
class FSGD(keras.optimizers.experimental.Optimizer):
  r"""Gradient descent (with momentum) optimizer.

  Update rule for parameter `w` with gradient `g` when `momentum` is 0:

  ```python
  w = w - learning_rate * g
  ```

  Update rule when `momentum` is larger than 0:

  ```python
  velocity = momentum * velocity - learning_rate * g
  w = w + velocity
  ```

  When `nesterov=True`, this rule becomes:

  ```python
  velocity = momentum * velocity - learning_rate * g
  w = w + momentum * velocity - learning_rate * g
  ```

  Attributes:
    vderiv: the fractional order tested for real values - interval [0, 2)
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use. The
      learning rate. Defaults to 0.001.
    momentum: float hyperparameter >= 0 that accelerates gradient descent
      in the relevant
      direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient
      descent.
    nesterov: boolean. Whether to apply Nesterov momentum.
      Defaults to `False`.
    clipnorm: see the `clipnorm` argument of `optimizer_experimental.Optimizer`.
    clipvalue: see the `clipvalue` argument of
      `optimizer_experimental.Optimizer`.
    global_clipnorm: see the `global_clipnorm` argument of
      `optimizer_experimental.Optimizer`.
    use_ema: see the `use_ema` argument of `optimizer_experimental.Optimizer`.
    ema_momentum: see the `ema_momentum` argument of
      `optimizer_experimental.Optimizer`.
    ema_overwrite_frequency: see the `ema_overwrite_frequency` argument of
      `optimizer_experimental.Optimizer`.
    jit_compile: see the `jit_compile` argument of
      `optimizer_experimental.Optimizer`.
    name: Optional name prefix for the operations created when applying
      gradients. Defaults to `"SGD"`.
    **kwargs: see the `**kwargs` argument of `optimizer_experimental.Optimizer`.

  Usage:

  >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1)
  >>> var = tf.Variable(1.0)
  >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> # Step is `- learning_rate * grad`
  >>> var.numpy()
  0.9

  >>> opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
  >>> var = tf.Variable(1.0)
  >>> val0 = var.value()
  >>> loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
  >>> # First step is `- learning_rate * grad`
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> val1 = var.value()
  >>> (val0 - val1).numpy()
  0.1
  >>> # On later steps, step-size increases because of momentum
  >>> step_count = opt.minimize(loss, [var]).numpy()
  >>> val2 = var.value()
  >>> (val1 - val2).numpy()
  0.18

  Reference:
      - For `nesterov=True`, See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
  """

  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               amsgrad=False,
               clipnorm=None,
               clipvalue=None,
               global_clipnorm=None,
               use_ema=False,
               ema_momentum=0.99,
               ema_overwrite_frequency=100,
               jit_compile=False,
               name='FSGD',
               vderiv=1.0,
               **kwargs):
    super(FSGD, self).__init__(
        name=name,
        clipnorm=clipnorm,
        clipvalue=clipvalue,
        global_clipnorm=global_clipnorm,
        use_ema=use_ema,
        ema_momentum=ema_momentum,
        ema_overwrite_frequency=ema_overwrite_frequency,
        jit_compile=jit_compile,
        **kwargs)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.momentum = momentum
    self.nesterov = nesterov
    self.vderiv = vderiv
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError('`momentum` must be between [0, 1].')

  def build(self, var_list):
    """Initialize optimizer variables.

    SGD optimizer has one variable `momentums`, only set if `self.momentum`
    is not 0.

    Args:
      var_list: list of model variables to build SGD variables on.
    """
    super().build(var_list)
    if hasattr(self, '_built') and self._built:
      return
    self.momentums = []
    if self.momentum != 0:
      for var in var_list:
        self.momentums.append(
            self.add_variable_from_reference(
                model_variable=var, variable_name='m'))
    self._built = True

  def update_step(self, gradient, variable):
    """Update step given gradient and the associated model variable."""
    if self._var_key(variable) not in self._index_dict:
      raise KeyError(f'Optimizer cannot recognize variable {variable.name}, '
                     f'this usually means you are calling an optimizer '
                     f'previously used on a different model. Please try '
                     f'creating a new optimizer instance.')

##start oscar contribution FSGD
    potencia = tf.pow(tf.abs(variable) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    gradient = tf.multiply( gradient , potencia)
##end oscar contribution FSGD

    lr = tf.cast(self.learning_rate, variable.dtype)
    m = None
    var_key = self._var_key(variable)
    if self.momentum != 0:
      momentum = tf.cast(self.momentum, variable.dtype)
      m = self.momentums[self._index_dict[var_key]]

    # TODO(b/204321487): Add nesterov acceleration.
    if isinstance(gradient, tf.IndexedSlices):
      # Sparse gradients.
      add_value = tf.IndexedSlices(-gradient.values * lr, gradient.indices)
      if m is not None:
        m.assign(m * momentum)
        m.scatter_add(add_value)
        if self.nesterov:
          variable.scatter_add(add_value)
          variable.assign_add(m * momentum)
        else:
          variable.assign_add(m)
      else:
        variable.scatter_add(add_value)
    else:
      # Dense gradients
      if m is not None:
        m.assign(-gradient * lr + m * momentum)
        if self.nesterov:
          variable.assign_add(-gradient * lr + m * momentum)
        else:
          variable.assign_add(m)
      else:                
        #https://guru99.es/tensor-tensorflow/          
        #USE uno_menos_v = 1 - vderiv
        #USE dos_menos_v = 2 - vderiv
        #USE gamma_2_nu = math.gamma(dos_menos_v)
        variable.assign_add(-gradient * lr)        #ORIGINAL

  def get_config(self):
    config = super(FSGD, self).get_config()

    config.update({
        'learning_rate': self._serialize_hyperparameter(self._learning_rate),
        'momentum': self.momentum,
        'nesterov': self.nesterov,
    })
    return config
##ENDModification FSGD


##START Modification SGDP to get FSGDP the fractional version of SGDP
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
#from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

class FSGDP(keras.optimizers.Optimizer):#optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True
    def __init__(self,
                 learning_rate=0.1,
                 momentum=0.0,
                 dampening=0.0,
                 weight_decay=0.0,
                 nesterov=False,
                 epsilon=1e-8,
                 delta=0.1,
                 wd_ratio=0.1,
                 name="FSGDP",
                 vderiv=1.0,
                 **kwargs):

        super(FSGDP, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

        self._set_hyper("momentum", momentum)
        self._set_hyper("dampening", dampening)
        self._set_hyper("epsilon", epsilon)
        self._set_hyper("delta", delta)
        self._set_hyper("wd_ratio", wd_ratio)

        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.vderiv = vderiv
        self.name = name

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        for var in var_list:
            self.add_slot(var, "buf")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(FSGDP, self)._prepare_local(var_device, var_dtype, apply_state)
        lr = apply_state[(var_device, var_dtype)]['lr_t']

        momentum = array_ops.identity(self._get_hyper("momentum", var_dtype))
        dampening = array_ops.identity(self._get_hyper('dampening', var_dtype))
        delta = array_ops.identity(self._get_hyper('delta', var_dtype))
        wd_ratio = array_ops.identity(self._get_hyper('wd_ratio', var_dtype))

        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                weight_decay=ops.convert_to_tensor_v2(self.weight_decay, var_dtype),
                momentum=momentum,
                dampening=dampening,
                delta=delta,
                wd_ratio=wd_ratio))


    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
##oscar FSGDP
        potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
        grad = tf.multiply( grad , potencia)
##oscar FSGDP

        buf = self.get_slot(var, 'buf')
        b_scaled_g_values = grad * (1 - coefficients['dampening'])
        buf_t = state_ops.assign(buf, buf * coefficients['momentum'] + b_scaled_g_values, use_locking=self._use_locking)

        if self.nesterov:
            d_p = grad + coefficients['momentum'] * buf_t
        else:
            d_p = buf_t

        # Projection
        wd_ratio = 1
        if len(var.shape) > 1:
            d_p, wd_ratio = self._projection(var, grad, d_p, coefficients['delta'], coefficients['wd_ratio'], coefficients['epsilon'])

        # Weight decay
        if self.weight_decay > 0:
            var = state_ops.assign(var, var * (1 - coefficients['lr'] * coefficients['weight_decay'] * wd_ratio), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, coefficients['lr'] * d_p, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, buf_t])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
##oscar FSGDP
        potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
        grad = tf.multiply( grad , potencia)
##oscar FSGDP

        buf = self.get_slot(var, 'buf')
        b_scaled_g_values = grad * (1 - coefficients['dampening'])
        buf_t = state_ops.assign(buf, buf * coefficients['momentum'], use_locking=self._use_locking)

        with ops.control_dependencies([buf_t]):
            buf_t = self._resource_scatter_add(buf, indices, b_scaled_g_values)

        if self.nesterov:
            d_p = self._resource_scatter_add(buf_t * coefficients['momentum'], indices, grad)
        else:
            d_p = buf_t

        # Projection
        wd_ratio = 1
        if len(array_ops.shape(var)) > 1:
            d_p, wd_ratio = self._projection(var, grad, d_p, coefficients['delta'], coefficients['wd_ratio'],
                                             coefficients['epsilon'])

        # Weight decay
        if self.weight_decay > 0:
            var = state_ops.assign(var, var * (1 - coefficients['lr'] * coefficients['weight_decay'] * wd_ratio), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, coefficients['lr'] * d_p, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, buf_t])

    def _channel_view(self, x):
        return array_ops.reshape(x, shape=[array_ops.shape(x)[0], -1])

    def _layer_view(self, x):
        return array_ops.reshape(x, shape=[1, -1])

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = math_ops.euclidean_norm(x, axis=-1) + eps
        y_norm = math_ops.euclidean_norm(y, axis=-1) + eps
        dot = math_ops.reduce_sum(x * y, axis=-1)

        return math_ops.abs(dot) / x_norm / y_norm

    def _projection(self, var, grad, perturb, delta, wd_ratio, eps):
        # channel_view
        cosine_sim = self._cosine_similarity(grad, var, eps, self._channel_view)
        cosine_max = math_ops.reduce_max(cosine_sim)
        compare_val = delta / math_ops.sqrt(math_ops.cast(self._channel_view(var).shape[-1], dtype=delta.dtype))

        perturb, wd = control_flow_ops.cond(pred=cosine_max < compare_val,
                                            true_fn=lambda : self.channel_true_fn(var, perturb, wd_ratio, eps),
                                            false_fn=lambda : self.channel_false_fn(var, grad, perturb, delta, wd_ratio, eps))

        return perturb, wd

    def channel_true_fn(self, var, perturb, wd_ratio, eps):
        expand_size = [-1] + [1] * (len(var.shape) - 1)
        var_n = var / (array_ops.reshape(math_ops.euclidean_norm(self._channel_view(var), axis=-1), shape=expand_size) + eps)
        perturb = state_ops.assign_sub(perturb, var_n * array_ops.reshape(math_ops.reduce_sum(self._channel_view(var_n * perturb), axis=-1), shape=expand_size))
        wd = wd_ratio

        return perturb, wd

    def channel_false_fn(self, var, grad, perturb, delta, wd_ratio, eps):
        cosine_sim = self._cosine_similarity(grad, var, eps, self._layer_view)
        cosine_max = math_ops.reduce_max(cosine_sim)
        compare_val = delta / math_ops.sqrt(math_ops.cast(self._layer_view(var).shape[-1], dtype=delta.dtype))

        perturb, wd = control_flow_ops.cond(cosine_max < compare_val,
                                              true_fn=lambda : self.layer_true_fn(var, perturb, wd_ratio, eps),
                                              false_fn=lambda : self.identity_fn(perturb))

        return perturb, wd

    def layer_true_fn(self, var, perturb, wd_ratio, eps):
        expand_size = [-1] + [1] * (len(var.shape) - 1)
        var_n = var / (array_ops.reshape(math_ops.euclidean_norm(self._layer_view(var), axis=-1), shape=expand_size) + eps)
        perturb = state_ops.assign_sub(perturb, var_n * array_ops.reshape(math_ops.reduce_sum(self._layer_view(var_n * perturb), axis=-1), shape=expand_size))
        wd = wd_ratio

        return perturb, wd

    def identity_fn(self, perturb):
        wd = 1.0

        return perturb, wd

    def get_config(self):
        config = super(FSGDP, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'momentum': self._serialize_hyperparameter('momentum'),
            'dampening': self._serialize_hyperparameter('dampening'),
            'delta': self._serialize_hyperparameter('delta'),
            'wd_ratio': self._serialize_hyperparameter('wd_ratio'),
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            "nesterov": self.nesterov,
        })
        return config
##END modification FSGDP

#START ADAM adaptation to get FADAM the  fractional version of ADAM
'''FADAM
Adapted from original: anaconda3/envs/tensorflow/lib/python3.7/site-packages/keras/optimizer_v2/adam.py

'''
"""FAdam optimizer implementation. Oscar Fractional Derivative Optimizer """

#import tensorflow.compat.v2 as tf
#from tensorflow.python.util.tf_export import keras_export


#@keras_export('keras.optimizers.FAdam')
#class FAdam(optimizer_v2.OptimizerV2): dont forget delete V2 and to use keras.optimizers.Optimizer
class FAdam(keras.optimizers.Optimizer):        
  r"""Optimizer that implements the Adam algorithm.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.

  According to
  [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
  the method is "*computationally
  efficient, has little memory requirement, invariant to diagonal rescaling of
  gradients, and is well suited for problems that are large in terms of
  data/parameters*".

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use, The
      learning rate. Defaults to 0.001.
    beta_1: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use. The
      exponential decay rate for the 1st moment estimates. Defaults to 0.9.
    beta_2: A float value or a constant float tensor, or a callable
      that takes no arguments and returns the actual value to use, The
      exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
      the paper "On the Convergence of Adam and beyond". Defaults to `False`.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Adam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Usage:

  >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> # The first step is `-learning_rate*sign(grad)`
  >>> var1.numpy()
  9.9

  Reference:
    - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
    - [Reddi et al., 2018](
        https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

  Notes:

  The default value of 1e-7 for epsilon might not be a good default in
  general. For example, when training an Inception network on ImageNet a
  current good choice is 1.0 or 0.1. Note that since Adam uses the
  formulation just before Section 2.1 of the Kingma and Ba paper rather than
  the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
  hat" in the paper.

  The sparse implementation of this algorithm (used when the gradient is an
  IndexedSlices object, typically because of `tf.gather` or an embedding
  lookup in the forward pass) does apply momentum to variable slices even if
  they were not used in the forward pass (meaning they have a gradient equal
  to zero). Momentum decay (beta1) is also applied to the entire momentum
  accumulator. This means that the sparse behavior is equivalent to the dense
  behavior (in contrast to some momentum implementations which ignore momentum
  unless a variable slice was actually used).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name='FAdam',
               vderiv=1.0,
               **kwargs):
    super(FAdam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.amsgrad = amsgrad
    self.vderiv = vderiv
    self.name = name

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(FAdam, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = tf.pow(beta_1_t, local_step)
    beta_2_power = tf.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
          (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            epsilon=tf.convert_to_tensor(
                self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t))

  def set_weights(self, weights):
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(Adam, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

##oscar FAdam
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
##oscar FAdam

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    if not self.amsgrad:
      return tf.raw_ops.ResourceApplyAdam(
          var=var.handle,
          m=m.handle,
          v=v.handle,
          beta1_power=coefficients['beta_1_power'],
          beta2_power=coefficients['beta_2_power'],
          lr=coefficients['lr_t'],
          beta1=coefficients['beta_1_t'],
          beta2=coefficients['beta_2_t'],
          epsilon=coefficients['epsilon'],
          grad=grad,
          use_locking=self._use_locking)
    else:
      vhat = self.get_slot(var, 'vhat')
      return tf.raw_ops.ResourceApplyAdamWithAmsgrad(
          var=var.handle,
          m=m.handle,
          v=v.handle,
          vhat=vhat.handle,
          beta1_power=coefficients['beta_1_power'],
          beta2_power=coefficients['beta_2_power'],
          lr=coefficients['lr_t'],
          beta1=coefficients['beta_1_t'],
          beta2=coefficients['beta_2_t'],
          epsilon=coefficients['epsilon'],
          grad=grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
##oscar FAdam
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
##oscar FAdam
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = tf.compat.v1.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)
    with tf.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = tf.compat.v1.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)
    with tf.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    if not self.amsgrad:
      v_sqrt = tf.sqrt(v_t)
      var_update = tf.compat.v1.assign_sub(
          var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return tf.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat_t = tf.maximum(v_hat, v_t)
      with tf.control_dependencies([v_hat_t]):
        v_hat_t = tf.compat.v1.assign(
            v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = tf.sqrt(v_hat_t)
      var_update = tf.compat.v1.assign_sub(
          var,
          coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return tf.group(*[var_update, m_t, v_t, v_hat_t])

  def get_config(self):
    config = super(Adam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config


##END ADAM Adaptation


##Start Modification of Adagrad to get FAdagrad the fractional version of Adagrad
#Modification from original anaconda3/envs/tensorflow/lib/python3.7/site-packages/keras/optimizer_v2/adagrad.py
class FAdagrad(keras.optimizers.Optimizer):
  r"""Optimizer that implements the Adagrad algorithm.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  Args:
    learning_rate: Initial value for the learning rate:
      either a floating point value,
      or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
      Defaults to 0.001.
      Note that `Adagrad` tends to benefit from higher initial learning rate
      values compared to other optimizers.
      To match the exact form in the original paper, use 1.0.
    initial_accumulator_value: Floating point value.
      Starting value for the accumulators (per-parameter momentum values).
      Must be non-negative.
    epsilon: Small floating point value used to maintain numerical stability.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Adagrad"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm and represents
      the maximum L2 norm of each weight variable;
      `"clipvalue"` (float) clips gradient by value and represents the
      maximum absolute value of each weight variable.

  Reference:
    - [Duchi et al., 2011](
      http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               name='FAdagrad',
               vderiv=1.0,
               **kwargs):
    if initial_accumulator_value < 0.0:
      raise ValueError('initial_accumulator_value must be non-negative: %s' %
                       initial_accumulator_value)
    if epsilon is None:
      epsilon = backend_config.epsilon()
    super(FAdagrad, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._initial_accumulator_value = initial_accumulator_value
    self.epsilon = epsilon or backend_config.epsilon()
    self.vderiv = vderiv
    self.name = name

  def _create_slots(self, var_list):
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = tf.compat.v1.constant_initializer(
          self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, 'accumulator', init)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(FAdagrad, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(
        dict(
            epsilon=tf.convert_to_tensor(
                self.epsilon, var_dtype),
            neg_lr_t=-apply_state[(var_device, var_dtype)]['lr_t'],
            zero=tf.zeros((), dtype=tf.int64)))

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(FAdagrad, self).set_weights(weights)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates an optimizer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Args:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An optimizer instance.
    """
    if 'initial_accumulator_value' not in config:
      config['initial_accumulator_value'] = 0.1
    if 'lr' in config:
      config['learning_rate'] = config.pop('lr')
    return cls(**config)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    acc = self.get_slot(var, 'accumulator')
#oscar FAdagrad    
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
#oscar FAdagrad
    return tf.raw_ops.ResourceApplyAdagradV2(
        var=var.handle,
        accum=acc.handle,
        lr=coefficients['lr_t'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    acc = self.get_slot(var, 'accumulator')
#oscar FAdagrad    
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
#oscar FAdagrad
    
    return tf.raw_ops.ResourceSparseApplyAdagradV2(
        var=var.handle,
        accum=acc.handle,
        lr=coefficients['lr_t'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(FAdagrad, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'initial_accumulator_value': self._initial_accumulator_value,
        'epsilon': self.epsilon,
    })
    return config
##End FAdagrad Modification of Adagrad


##START Modification of Adadelta to get FAdadelta the fractional version of Adadelta
##Original from anaconda3/envs/tensorflow/lib/python3.7/site-packages/keras/optimizer_v2/adadelta.py
class FAdadelta(keras.optimizers.Optimizer): #optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the Adadelta algorithm.

  Adadelta optimization is a stochastic gradient descent method that is based on
  adaptive learning rate per dimension to address two drawbacks:

  - The continual decay of learning rates throughout training.
  - The need for a manually selected global learning rate.

  Adadelta is a more robust extension of Adagrad that adapts learning rates
  based on a moving window of gradient updates, instead of accumulating all
  past gradients. This way, Adadelta continues learning even when many updates
  have been done. Compared to Adagrad, in the original version of Adadelta you
  don't have to set an initial learning rate. In this version, the initial
  learning rate can be set, as in most other Keras optimizers.

  Args:
    learning_rate: Initial value for the learning rate:
      either a floating point value,
      or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
      Defaults to 0.001.
      Note that `Adadelta` tends to benefit from higher initial learning rate
      values compared to other optimizers.
      To match the exact form in the original paper, use 1.0.
    rho: A `Tensor` or a floating point value. The decay rate.
    epsilon: Small floating point value used to maintain numerical stability.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"Adadelta"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm and represents
      the maximum norm of each parameter;
      `"clipvalue"` (float) clips gradient by value and represents the
      maximum absolute value of each parameter.

  Reference:
    - [Zeiler, 2012](http://arxiv.org/abs/1212.5701)
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               rho=0.95,
               epsilon=1e-7,
               name='FAdadelta',
               vderiv=1.0,
               **kwargs):
    super(FAdadelta, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('rho', rho)
    self.epsilon = epsilon or backend_config.epsilon()
    self.vderiv = vderiv
    self.name = name

  def _create_slots(self, var_list):
    # Separate for-loops to respect the ordering of slot variables from v1.
    for v in var_list:
      self.add_slot(v, 'accum_grad')
    for v in var_list:
      self.add_slot(v, 'accum_var')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(FAdadelta, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(
        dict(
            epsilon=tf.convert_to_tensor(
                self.epsilon, var_dtype),
            rho=tf.identity(self._get_hyper('rho', var_dtype))))

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(FAdadelta, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
    
#oscar FAdadelta
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
#oscar FAdadelta   
    return tf.raw_ops.ResourceApplyAdadelta(
        var=var.handle,
        accum=accum_grad.handle,
        accum_update=accum_var.handle,
        lr=coefficients['lr_t'],
        rho=coefficients['rho'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
#oscar FAdadelta   
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
#oscar FAdadelta     
    return tf.raw_ops.ResourceSparseApplyAdadelta(
        var=var.handle,
        accum=accum_grad.handle,
        accum_update=accum_var.handle,
        lr=coefficients['lr_t'],
        rho=coefficients['rho'],
        epsilon=coefficients['epsilon'],
        grad=grad,
        indices=indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(FAdadelta, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._initial_decay,
        'rho': self._serialize_hyperparameter('rho'),
        'epsilon': self.epsilon,
    })
    return config
##END FAdadelta modification of Adadelta

##START modification of RMSProp to get FRMSProp the fractional version of RMSProp
class FRMSprop(keras.optimizers.Optimizer):
  r"""Optimizer that implements the RMSprop algorithm.

  The gist of RMSprop is to:

  - Maintain a moving (discounted) average of the square of gradients
  - Divide the gradient by the root of this average

  This implementation of RMSprop uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance.

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use. The
      learning rate. Defaults to 0.001.
    rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
    momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    centered: Boolean. If `True`, gradients are normalized by the estimated
      variance of the gradient; if False, by the uncentered second moment.
      Setting this to `True` may help with training, but is slightly more
      expensive in terms of computation and memory. Defaults to `False`.
    name: Optional name prefix for the operations created when applying
      gradients. Defaults to `"RMSprop"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Note that in the dense implementation of this algorithm, variables and their
  corresponding accumulators (momentum, gradient moving average, square
  gradient moving average) will be updated even if the gradient is zero
  (i.e. accumulators will decay, momentum will be applied). The sparse
  implementation (used when the gradient is an `IndexedSlices` object,
  typically because of `tf.gather` or an embedding lookup in the forward pass)
  will not update variable slices or their accumulators unless those slices
  were used in the forward pass (nor is there an "eventual" correction to
  account for these omitted updates). This leads to more efficient updates for
  large embedding lookup tables (where most of the slices are not accessed in
  a particular graph execution), but differs from the published algorithm.

  Usage:

  >>> opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2) / 2.0    # d(loss) / d(var1) = var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> var1.numpy()
  9.683772

  Reference:
    - [Hinton, 2012](
      http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               rho=0.9,
               momentum=0.0,
               epsilon=1e-7,
               centered=False,
               name="FRMSprop",
               vderiv=1.0,
               **kwargs):
    """Construct a new FRMSprop optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.001.
      rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
      momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      centered: Boolean. If `True`, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to `True` may help with training, but is slightly more
        expensive in terms of computation and memory. Defaults to `False`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSprop".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate`, `decay`, `momentum`, and
    `epsilon` can each be a callable that takes no arguments and returns the
    actual value to use. This can be useful for changing these values across
    different invocations of optimizer functions.
    @end_compatibility
    """
    
    super(FRMSprop, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("rho", rho)
    self.vderiv = vderiv
    self.name = name

    self._momentum = False
    if isinstance(momentum, tf.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError(f"`momentum` must be between [0, 1]. Received: "
                       f"momentum={momentum} (of type {type(momentum)}).")
    self._set_hyper("momentum", momentum)

    self.epsilon = epsilon or backend_config.epsilon()
    self.centered = centered

  def _set_hyper(self, name, value):
        if isinstance(value, (int, float, tf.Variable)):
            self._hyper[name] = value
        elif callable(value):
            self._hyper[name] = value()
        else:
            raise ValueError('Invalid value type for hyperparameter:', value)
        
  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, "rms")
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")
    if self.centered:
      for var in var_list:
        self.add_slot(var, "mg")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(FRMSprop, self)._prepare_local(var_device, var_dtype, apply_state)

    rho = tf.identity(self._get_hyper("rho", var_dtype))
    apply_state[(var_device, var_dtype)].update(
        dict(
            neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
            epsilon=tf.convert_to_tensor(
                self.epsilon, var_dtype),
            rho=rho,
            momentum=tf.identity(self._get_hyper("momentum", var_dtype)),
            one_minus_rho=1. - rho))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
##oscar FRMSProp
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
##oscar FRMSProp

    rms = self.get_slot(var, "rms")
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      if self.centered:
        mg = self.get_slot(var, "mg")
        return tf.raw_ops.ResourceApplyCenteredRMSProp(
            var=var.handle,
            mg=mg.handle,
            ms=rms.handle,
            mom=mom.handle,
            lr=coefficients["lr_t"],
            rho=coefficients["rho"],
            momentum=coefficients["momentum"],
            epsilon=coefficients["epsilon"],
            grad=grad,
            use_locking=self._use_locking)
      else:
        return tf.raw_ops.ResourceApplyRMSProp(
            var=var.handle,
            ms=rms.handle,
            mom=mom.handle,
            lr=coefficients["lr_t"],
            rho=coefficients["rho"],
            momentum=coefficients["momentum"],
            epsilon=coefficients["epsilon"],
            grad=grad,
            use_locking=self._use_locking)
    else:
      rms_t = (coefficients["rho"] * rms +
               coefficients["one_minus_rho"] * tf.square(grad))
      rms_t = tf.compat.v1.assign(rms, rms_t, use_locking=self._use_locking)
      denom_t = rms_t
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_t = coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
        mg_t = tf.compat.v1.assign(mg, mg_t, use_locking=self._use_locking)
        denom_t = rms_t - tf.square(mg_t)
      var_t = var - coefficients["lr_t"] * grad / (
          tf.sqrt(denom_t) + coefficients["epsilon"])
      return tf.compat.v1.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
##oscar FRMSProp
    potencia = tf.pow(tf.abs(var) + tf.constant(0.000001, tf.float32 ) , tf.constant(1 - self.vderiv, tf.float32 ) ) /math.gamma(2 - self.vderiv)
    grad = tf.multiply( grad , potencia)
##oscar FRMSProp
    rms = self.get_slot(var, "rms")
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      if self.centered:
        mg = self.get_slot(var, "mg")
        return tf.raw_ops.ResourceSparseApplyCenteredRMSProp(
            var=var.handle,
            mg=mg.handle,
            ms=rms.handle,
            mom=mom.handle,
            lr=coefficients["lr_t"],
            rho=coefficients["rho"],
            momentum=coefficients["momentum"],
            epsilon=coefficients["epsilon"],
            grad=grad,
            indices=indices,
            use_locking=self._use_locking)
      else:
        return tf.raw_ops.ResourceSparseApplyRMSProp(
            var=var.handle,
            ms=rms.handle,
            mom=mom.handle,
            lr=coefficients["lr_t"],
            rho=coefficients["rho"],
            momentum=coefficients["momentum"],
            epsilon=coefficients["epsilon"],
            grad=grad,
            indices=indices,
            use_locking=self._use_locking)
    else:
      rms_scaled_g_values = (grad * grad) * coefficients["one_minus_rho"]
      rms_t = tf.compat.v1.assign(rms, rms * coefficients["rho"],
                               use_locking=self._use_locking)
      with tf.control_dependencies([rms_t]):
        rms_t = self._resource_scatter_add(rms, indices, rms_scaled_g_values)
        rms_slice = tf.gather(rms_t, indices)
      denom_slice = rms_slice
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_scaled_g_values = grad * coefficients["one_minus_rho"]
        mg_t = tf.compat.v1.assign(mg, mg * coefficients["rho"],
                                use_locking=self._use_locking)
        with tf.control_dependencies([mg_t]):
          mg_t = self._resource_scatter_add(mg, indices, mg_scaled_g_values)
          mg_slice = tf.gather(mg_t, indices)
          denom_slice = rms_slice - tf.square(mg_slice)
      var_update = self._resource_scatter_add(
          var, indices, coefficients["neg_lr_t"] * grad / (
              tf.sqrt(denom_slice) + coefficients["epsilon"]))
      if self.centered:
        return tf.group(*[var_update, rms_t, mg_t])
      return tf.group(*[var_update, rms_t])

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(FRMSprop, self).set_weights(weights)

  def get_config(self):
    config = super(FRMSprop, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._initial_decay,
        "rho": self._serialize_hyperparameter("rho"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "epsilon": self.epsilon,
        "centered": self.centered,
    })
    return config

##END FRMSProp modification of RMSProp


## Simple test. Several fractional optimizers. Let vderiv = 1.0 to match the classical non-fractional version, see lines 40-45

# seed(1)
# # to make output stable across runs
# np.random.seed(42)
# tf.random.set_seed(42)

#opt = tf.keras.optimizers.SGD(learning_rate=0.1)
#opt = FSGD(learning_rate=0.1)
#opt = FSGDP(learning_rate=0.1, momentum=0.1)
#opt = FAdam(learning_rate=0.1)
#opt = FAdagrad(learning_rate=0.1)
# opt = FAdadelta(learning_rate=0.1)
# opt = FRMSprop(learning_rate=0.1, vderiv=1.0)

# var = tf.Variable(1.0)
# loss = lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
# step_count = opt.minimize(loss, [var] ) #.numpy()
# # Step is `- learning_rate * grad`
# print( var.numpy() )