""" Where the methods that do the heavy lifting in zooming live

For the time being, just supports the mandelbrot set while basic development takes place
"""

import tensorflow as tf
import numpy as np

from scipy.sparse import coo_matrix


def build_sparse_render_step(center=(-1.75, 0), characteristic_length=3, resolution=(1080, 1920)):
    """ Builds and returns the spare step and placeholders

    Args:
      resolution: resolution to compute at - [x_res, y_res]

    Returns:
      sparse_step: op that runs an iteration - tf.op


    """
    #aspect_ratio = resolution[0] / float(resolution[1])

    # size of the field of view
    x_fov = characteristic_length * np.sqrt(resolution[0]) / np.sqrt(resolution[1])
    y_fov = characteristic_length * np.sqrt(resolution[0]) / np.sqrt(resolution[1])

    # bounding box coordinates in each dimension
    x_bbox = (center[0] - x_fov / 2, center[0] + x_fov / 2)
    y_bbox = (center[1] - y_fov / 2, center[1] + y_fov / 2)

    x_mesh, y_mesh = np.meshgrid(np.linspace(*x_bbox, num=resolution[0]), np.linspace(*y_bbox, num=resolution[1]), indexing='ij')
    _z_mesh = x_mesh + 1j * y_mesh

    _z_mesh_coo = coo_matrix(_z_mesh)
    _z_idxs = np.stack([_z_mesh_coo.rows, _z_mesh_coo.cols], axis=1)
    _z_values = _z_mesh_coo.data.astype(np.complex64)

    z_curr = tf.sparse_reorder(tf.SparseTensor(_z_idxs, _z_values, shape=resolution, name='z_curr'))
    c_mesh = tf.sparse_reorder(tf.SparseTensor(_z_idxs, _z_values, shape=resolution, name='c_mesh'))

    _zeros_coo = coo_matrix(np.zeros(resolution))
    _zeros_idxs = np.stack([_zeros_coo.rows, _zeros_coo.idxs], axis=1)
    num_steps = tf.sparse_reorder(tf.SparseTensor(_zeros_idxs, _zeros_coo.data, shape=resolution, name='num_steps'))

    z_next = z_curr * z_curr + c_mesh

    _not_diverged_bool = tf.abs(z_next).values < 4
    _not_diverged_values = tf.cast(_not_diverged_bool, tf.float32)
    not_diverged = tf.SparseTensor(z_next.indices, _not_diverged_values, shape=resolution)

    assign_z = z_curr.assign(z_next)
    incremented_num_steps = num_steps + not_diverged
    assign_num_steps_idxs = num_steps.indices.assign(incremented_num_steps.indices)
    assign_num_steps_values = num_steps.values.assign(incremented_num_steps.values)

    _diverged_bool = tf.logical_not(_not_diverged_bool)

    # unclear if this is even feasible with tensorflow sparse ops right now







def build_dense_render_step(center=(-1.75, 0), characteristic_length=3, resolution=(1080, 1920)):
    """ Builds a simple but slower dense render step

    Adapted from http://turagas-ws4:8335/notebooks/projects/tf_fractal_zoom/demo_exploration%20.ipynb#

    The size of the rendered field view is calculated from the characteristic_length and resolution in the following way:
      resolution: (a, b)
      aspect ratio: (a / b)
      characteristic_length: l_0
      x_fov = l_0 * sqrt(a) / sqrt(b)
      y_fov = l_0 * sqrt(b) / sqrt(a)
      fov aspect ratio: (x_fov / y_fov) = (a / b)

    Args:
      center: (optional) coords render will be centered on
      characteristic_size: (optional) sets the characteristic size this render - float
      resolution: (optional) resolution to compute at - [x_res, y_res]

    Returns:
      step: op that performs one mandelbrot iteration - tf.op
      num_steps: the number of iterations until divergence for each element - tensor [1080, 1920]

    """
    #aspect_ratio = resolution[0] / float(resolution[1])

    # size of the field of view
    x_fov = characteristic_length * np.sqrt(resolution[0]) / np.sqrt(resolution[1])
    y_fov = characteristic_length * np.sqrt(resolution[0]) / np.sqrt(resolution[1])

    # bounding box coordinates in each dimension
    x_bbox = (center[0] - x_fov / 2, center[0] + x_fov / 2)
    y_bbox = (center[1] - y_fov / 2, center[1] + y_fov / 2)

    x_mesh, y_mesh = np.meshgrid(np.linspace(*x_bbox, num=resolution[0]), np.linspace(*y_bbox, num=resolution[1]), indexing='ij')
    _c_mesh = x_mesh + 1j * y_mesh

    # all with shape [1080, 1920]
    c_mesh = tf.constant(_c_mesh.astype(np.complex64), name='c_mesh')
    z_curr = tf.Variable(c_mesh, name='z_curr')
    # we stop iterating this when the corresponding point has diverged
    num_steps = tf.Variable(tf.zeros_like(c_mesh, tf.float32))

    # Compute the new values of z: z^2 + x
    z_next = z_curr * z_curr + c_mesh

    # Have we diverged with this new value?
    not_diverged = tf.abs(z_next) < 4

    # Operation to update the zs and the iteration count.
    #
    # Note: We keep computing zs after they diverge! This
    #       is very wasteful! There are better, if a little
    #       less simple, ways to do this.
    #
    step = tf.group(
        z_curr.assign(z_next),
        num_steps.assign_add(tf.cast(not_diverged, tf.float32))
      )
    return step, num_steps
