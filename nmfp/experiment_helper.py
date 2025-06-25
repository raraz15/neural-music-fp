import os
import yaml
import subprocess
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.summary import create_file_writer


class ExperimentHelper:
    """Experiment Helper class for conducting experiment.
    An object of this class manages:
        - initializing/restoring model checkpoints with optimizer states,
        - logging and writing loss metrics and images to Tensorboard.

    Usage:
    ------
        1. Define a new or existing (for restoring) experiment name.
        2. Construct {optimizer, model}.
        3. Get the number of steps for a single epoech from dataloader.
        4. Construct a helper class object.
        5. In your training loop, call the methods described in the below.

    Methods:
    --------
        - save_checkpoint():
            Save current model and optimizer states to checkpoint.
        - update_on_epoch_end():
            Update current epoch index, and loss metrics.
        - update_tr_loss(value, tb=True)->average_loss:
            Update loss value to return average loss within this epoch.
    """

    def __init__(
        self,
        cfg: dict,
        optimizer: tf.keras.optimizers,
        model_to_checkpoint: tf.keras.Model,
        max_to_keep: int = 1,
    ) -> None:
        """Initialize the ExperimentHelper object.

        Parameters
        ----------
            cfg : (dict)
                Config file of the model.
            optimizer : <tf.keras.optimizer>
                Assign a pre-constructed optimizer.
            model_to_checkpoint : <tf.keras.Model>
                Model to train.
            max_to_keep : (int), optional
                Maximum number of checkpoints to keep. The default is 10.
        """

        # Get the git sha and add it to the config
        # cfg["GIT_SHA"] = (
        #     subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        # )

        # Add the current time to the config
        cfg["TIME"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Save the config file
        self.cfg = cfg

        # Experiment settings
        self.checkpoint_name = self.cfg["MODEL"]["NAME"]

        # Initialize parameters
        self.epoch = 1

        # Choose the root directory for logs
        if self.cfg["MODEL"]["LOG_ROOT_DIR"]:
            _root_dir = self.cfg["MODEL"]["LOG_ROOT_DIR"]
        else:
            _root_dir = "./logs/"

        # Set the default directories
        self._log_dir = os.path.join(_root_dir, "fit", self.checkpoint_name)
        self._checkpoint_save_dir = os.path.join(
            _root_dir, "checkpoint", self.checkpoint_name
        )

        # Tensorboard writer for the train loss
        self._tr_summary_writer = create_file_writer(
            os.path.join(self._log_dir, "train")
        )

        # Track the learning rate
        self._lr_summary_writer = create_file_writer(os.path.join(self._log_dir, "lr"))

        # Logging loss and acc metrics
        self._tr_loss = K.metrics.Mean(name="train_loss")
        self._tr_loss.reset_states()

        # Assign optimizer and model to checkpoint
        self.optimizer = optimizer  # assign, not to create.
        self._model_to_checkpoint = model_to_checkpoint  # assign, not to create.

        # General Setup checkpoint and checkpoint manager
        self._checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, model=model_to_checkpoint
        )
        self.c_manager = tf.train.CheckpointManager(
            checkpoint=self._checkpoint,
            directory=self._checkpoint_save_dir,
            max_to_keep=max_to_keep,
            step_counter=self.optimizer.iterations,
        )

        self.load_checkpoint()
        self.write_lr()

    def update_on_epoch_end(self, save_checkpoint_now: bool = True) -> None:
        """Update current epoch index, and loss metrics."""

        if save_checkpoint_now:
            self.save_checkpoint()

        # Save learning rate to tensorboard
        self.write_lr()

        # Reset loss metrics
        self._tr_loss.reset_states()
        self.epoch += 1

    def update_tr_loss(self, value: float) -> float:
        """
        Parameters
        ----------
            value : (float)
                Loss value of the current train step.

        Returns
        -------
            avg_loss: (float)
                Cumulative-average training loss within current epoch after
                current iteration.
        """

        with self._tr_summary_writer.as_default():
            tf.summary.scalar("loss", value, step=self.optimizer.iterations)

        # Average the loss over the epoch
        avg_tr_loss = self._tr_loss(value)

        return avg_tr_loss

    def write_lr(self) -> None:
        """Write the learning rate to tensorboard."""

        if isinstance(
            self.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            current_lr = self.optimizer.lr(self.optimizer.iterations)
        else:
            current_lr = self.optimizer.lr

        with self._lr_summary_writer.as_default():
            tf.summary.scalar("lr", current_lr, step=self.optimizer.iterations)

    def load_checkpoint(self) -> None:
        """Try loading a saved checkpoint. If no checkpoint, initialize from
        scratch.
        """

        if self.c_manager.latest_checkpoint:
            print(
                f"-----------Restoring model from {self.c_manager.latest_checkpoint}-----------"
            )
            status = self._checkpoint.restore(self.c_manager.latest_checkpoint)
            status.expect_partial()
            self.epoch = (
                int(self.c_manager.latest_checkpoint.split(sep="ckpt-")[-1]) + 1
            )
        else:
            print("-----------Initializing model from scratch-----------")
            # Write the config file to the checkpoint directory
            os.makedirs(self._checkpoint_save_dir, exist_ok=True)
            # Save the config file
            with open(os.path.join(self._checkpoint_save_dir, "config.yaml"), "w") as f:
                yaml.dump(self.cfg, f)

    def save_checkpoint(self) -> None:
        """Save current model and optimizer states to checkpoint."""

        self.c_manager.save()
