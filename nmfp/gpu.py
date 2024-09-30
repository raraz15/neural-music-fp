"""This module contains methods to configure GPUs for TensorFlow."""

import os

import tensorflow as tf


def choose_first_gpu() -> None:
    """By default, TensorFlow maps nearly all of the GPU memory of all GPUs
    subject to CUDA_VISIBLE_DEVICES visible to the process. In the case where
    CUDA_VISIBLE_DEVICES is not set his method will choose the first GPU
    available to the process to preserve the rest of the GPUs.
    """

    # Get the list of GPUs
    gpus = tf.config.list_physical_devices("GPU")

    # If there are GPUs, choose the first one
    if gpus:
        print("Available GPUs:")
        for gpu in gpus:
            print("Name:", gpu.name, "  Type:", gpu.device_type)
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.set_visible_devices(gpus[0], "GPU")
            # Check which GPUs are visible
            logical_gpus = tf.config.list_logical_devices("GPU")
            print("Visible GPU(s):")
            for gpu in logical_gpus:
                print("Name:", gpu.name, "  Type:", gpu.device_type)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found")


def set_gpu_memory_growth() -> None:
    """By default, TensorFlow maps nearly all of the GPU memory of all GPUs
    (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to
    more efficiently use the relatively precious GPU memory resources on the
    devices by reducing memory fragmentation. To limit TensorFlow to a specific
    set of GPUs we use the tf.config.experimental.set_visible_devices method."""

    # Get the list of GPUs
    gpus = tf.config.list_physical_devices("GPU")

    # If there are GPUs, set the memory growth for each GPU
    if gpus:
        for gpu in gpus:
            print("Name:", gpu.name, "  Type:", gpu.device_type)
        try:
            # Memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Check which GPUs are visible
            logical_gpus = tf.config.list_logical_devices("GPU")
            print("Visible GPU(s):")
            for gpu in logical_gpus:
                print("Name:", gpu.name, "  Type:", gpu.device_type)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found")


def config_gpu_memory_limit(size_MB: int) -> None:
    """Set the exact amount of memory to use on the GPU. This is useful when
    you want to train multiple models on the same GPU. This method will set
    the memory limit for the first GPU.

    Parameters
    ----------
        size_MB : int
            The amount of memory to use on the GPU in MB (Megabytes).
    """

    # Get the list of GPUs
    gpus = tf.config.list_physical_devices("GPU")

    # If there is a GPU, set the memory limit
    if gpus:
        try:
            # Restrict TensorFlow to only allocate size_MB of memory on the first GPU
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=size_MB)]
            )
            print(f"GPU memory limit of {gpus[0].name} is set to {size_MB} MB")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found")


def set_global_determinism() -> None:
    """Make the GPU operations deterministic for the price of slowed
    calculations."""

    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    print("Global determinism set")
