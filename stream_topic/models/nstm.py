from datetime import datetime

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from loguru import logger
import torch.nn as nn
from ..commons.check_steps import check_dataset_steps
from ..utils.datamodule import TMDataModule
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.neural_basemodel import NeuralBaseModel
from .neural_base_models.nstm_base import NSTMBase
from optuna.integration import PyTorchLightningPruningCallback

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "NSTM"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class NSTM(BaseModel):
    """
    Neural Topic Model via Optimal Transport (NSTM). Based on the paper presented at ICLR 2021 by
    He Zhao, Dinh Phung, Viet Huynh, Trung Le, and Wray Buntine.

    This class initializes and configures the NSTM model with the specified
    hyperparameters and dataset. It inherits from the `BaseModel` class.

    Parameters
    ----------
    embed_size : int, optional
        Size of the embedding layer, by default 128.
    encoder_dim : int, optional
        Dimensionality of the encoder layer, by default 256.
    dropout : float, optional
        Dropout rate for the layers, by default 0.1.
    pretrained_WE : optional
        Pretrained word embeddings, by default None.
    train_WE : bool, optional
        Whether to train word embeddings, by default True.
    encoder_activation : callable, optional
        Activation function for the encoder, by default `nn.ReLU()`.
    batch_size : int, optional
        Batch size for training, by default 64.
    val_size : float, optional
        Proportion of the dataset to use for validation, by default 0.2.
    shuffle : bool, optional
        Whether to shuffle the dataset before splitting, by default True.
    random_state : int, optional
        Random seed for shuffling and splitting the dataset, by default 42.
    **kwargs : dict
        Additional keyword arguments to pass to the parent class constructor.

    Attributes
    ----------
    optimize : bool
        Flag indicating whether to optimize the model, by default False.
    n_topics : int or None
        Number of topics in the model, by default None.
    _status : TrainingStatus
        Current training status of the model, by default `TrainingStatus.NOT_STARTED`.
    hparams : dict
        Hyperparameters for the data module, including batch size, validation size,
        shuffling, and random state.

    Examples
    --------
    >>> NSTM = NSTM(embed_size=100, encoder_dim=200, dropout=0.2, batch_size=32)
    >>> print(NSTM.hparams)
    {'datamodule_args': {'batch_size': 32, 'val_size': 0.2, 'shuffle': True, 'random_state': 42,
                         'embeddings': False, 'bow': True, 'tf_idf': False, 'word_embeddings': False, 'min_df': 5}}
    """

    def __init__(
        self,
        embed_size: int = 128,
        encoder_dim: int = 256,
        dropout: float = 0.1,
        pretrained_WE=None,
        train_WE: bool = True,
        encoder_activation: callable = nn.ReLU(),
        batch_size=64,
        val_size=0.2,
        shuffle=True,
        random_state=42,
        **kwargs,
    ):
        """
        Initialize the NSTM model.

        Parameters
        ----------
        embed_size : int, optional
            Size of the embedding layer, by default 128.
        encoder_dim : int, optional
            Dimensionality of the encoder layer, by default 256.
        dropout : float, optional
            Dropout rate for the layers, by default 0.1.
        pretrained_WE : optional
            Pretrained word embeddings, by default None.
        train_WE : bool, optional
            Whether to train word embeddings, by default True.
        encoder_activation : callable, optional
            Activation function for the encoder, by default `nn.ReLU()`.
        batch_size : int, optional
            Batch size for training, by default 64.
        val_size : float, optional
            Proportion of the dataset to use for validation, by default 0.2.
        shuffle : bool, optional
            Whether to shuffle the dataset before splitting, by default True.
        random_state : int, optional
            Random seed for shuffling and splitting the dataset, by default 42.
        **kwargs : dict
            Additional keyword arguments to pass to the parent class constructor.
        """

        super().__init__(
            use_pretrained_embeddings=False,
            dropout=dropout,
            embed_size=embed_size,
            encoder_dim=encoder_dim,
            pretrained_WE=pretrained_WE,
            train_WE=train_WE,
            encoder_activation=encoder_activation,
        )
        self.save_hyperparameters(
            ignore=[
                "random_state",
            ]
        )

        self.hparams["datamodule_args"] = {
            "batch_size": batch_size,
            "val_size": val_size,
            "shuffle": shuffle,
            "random_state": random_state,
            "embeddings": False,
            "bow": True,
            "tf_idf": False,
            "word_embeddings": False,
            "min_df": kwargs.get("min_df", 5),
        }

        self.optimize = False
        self.n_topics = None
        self._status = TrainingStatus.NOT_STARTED

    def get_info(self):
        """
        Get information about the model.

        Returns
        -------
        dict
            Dictionary containing model information including model name,
            number of topics, embedding model name, UMAP arguments,
            K-Means arguments, and training status.
        """
        info = {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "trained": self._status.name,
        }
        return info

    def _initialize_model(self):
        """
        Initialize the neural base model.

        This method initializes the neural base model (`NeuralBaseModel`) with the given
        hyperparameters and dataset. It filters out certain hyperparameters that are
        not required by the model.

        Parameters
        ----------
        self : object
            The instance of the class that this method is a part of. This object should have
            attributes `dataset` and `hparams`.

        Attributes
        ----------
        model : NeuralBaseModel
            The initialized neural base model.
        """

        self.model = NeuralBaseModel(
            model_class=NSTMBase,
            dataset=self.dataset,
            **{
                k: v
                for k, v in self.hparams.items()
                if k not in ["datamodule_args", "max_epochs"]
            },
        )

    def _initialize_trainer(
        self,
        max_epochs,
        monitor,
        patience,
        mode,
        checkpoint_path,
        trial=None,
        **trainer_kwargs,
    ):
        """
        Initialize the PyTorch Lightning trainer.

        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs for training.
        monitor : str
            Metric to monitor for early stopping and checkpointing.
        patience : int
            Patience for early stopping.
        mode : str
            Mode for the monitored metric (min or max).
        checkpoint_path : str
            Path to save model checkpoints.
        **trainer_kwargs : dict
            Additional keyword arguments for the trainer.
        """

        logger.info(f"--- Initializing Trainer for {MODEL_NAME} ---")
        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,  # Specify the directory to save checkpoints
            filename="best_model",
        )

        model_callbacks = [
            early_stop_callback,
            checkpoint_callback,
            ModelSummary(max_depth=2),
        ]

        if self.optimize:
            model_callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor="val_loss")
            )

        # Initialize the trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=model_callbacks,
            **trainer_kwargs,
        )

    def _initialize_datamodule(
        self,
        dataset,
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to be used for training.
        """

        logger.info(f"--- Initializing Datamodule for {MODEL_NAME} ---")
        self.data_module = TMDataModule(
            batch_size=self.hparams["datamodule_args"]["batch_size"],
            shuffle=self.hparams["datamodule_args"]["shuffle"],
            val_size=self.hparams["datamodule_args"]["val_size"],
            random_state=self.hparams["datamodule_args"]["random_state"],
        )

        self.data_module.preprocess_data(
            dataset=dataset,
            **{
                k: v
                for k, v in self.hparams["datamodule_args"].items()
                if k not in ["batch_size", "shuffle", "val_size"]
            },
        )

        self.dataset = dataset

    def fit(
        self,
        dataset: TMDataset = None,
        n_topics: int = 20,
        val_size: float = 0.2,
        lr: float = 1e-04,
        lr_patience: int = 15,
        patience: int = 15,
        weight_decay: float = 1e-07,
        max_epochs: int = 100,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int = 101,
        checkpoint_path: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        trial=None,
        optimize=False,
        language = 'en',
        **kwargs,
    ):
        """
        Fits the NSTM (topic model in embedding spaces) topic model to the given dataset.

        Parameters
        ----------
        dataset : TMDataset, optional
            The dataset to train the topic model on. Defaults to None.
        n_topics : int, optional
            The number of topics to extract. Defaults to 20.
        val_size : float, optional
            The proportion of the dataset to use for validation. Defaults to 0.2.
        lr : float, optional
            The learning rate for the optimizer. Defaults to 1e-04.
        lr_patience : int, optional
            The number of epochs with no improvement after which the learning rate will be reduced. Defaults to 15.
        patience : int, optional
            The number of epochs with no improvement after which training will be stopped. Defaults to 15.
        weight_decay : float, optional
            The weight decay (L2 penalty) for the optimizer. Defaults to 1e-07.
        max_epochs : int, optional
            The maximum number of epochs to train for. Defaults to 100.
        batch_size : int, optional
            The batch size for training. Defaults to 32.
        shuffle : bool, optional
            Whether to shuffle the training data. Defaults to True.
        random_state : int, optional
            The random seed for reproducibility. Defaults to 101.
        checkpoint_path : str, optional
            The path to save model checkpoints. Defaults to "checkpoints".
        monitor : str, optional
            The metric to monitor for early stopping. Defaults to "val_loss".
        mode : str, optional
            The mode for early stopping. Defaults to "min".
        trial : optuna.Trial, optional
            The Optuna trial for hyperparameter optimization. Defaults to None.
        optimize : bool, optional
            Whether to optimize hyperparameters. Defaults to False.
        **kwargs
            Additional keyword arguments to be passed to the trainer.

        Raises
        ------
        ValueError
            If the dataset is not an instance of TMDataset or if the number of topics is less than or equal to 0.

        Examples
        --------
        >>> model = NSTM()
        >>> dataset = TMDataset(...)
        >>> model.fit(dataset, n_topics=20, val_size=0.2, lr=1e-04)
        """

        self.optimize = optimize
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        if language == 'chinese':
            check_dataset_steps(dataset, logger, MODEL_NAME, language='chinese')
        else:
            check_dataset_steps(dataset, logger, MODEL_NAME)

        self.n_topics = n_topics

        self.hparams.update(
            {
                "n_topics": n_topics,
                "lr": lr,
                "lr_patience": lr_patience,
                "patience": patience,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
            }
        )

        self.hparams["datamodule_args"].update(
            {
                "batch_size": batch_size,
                "val_size": val_size,
                "shuffle": shuffle,
                "random_state": random_state,
            }
        )

        try:

            self._status = TrainingStatus.INITIALIZED
            self._initialize_datamodule(dataset=dataset)

            self._initialize_model()

            self._initialize_trainer(
                max_epochs=self.hparams["max_epochs"],
                monitor=monitor,
                patience=patience,
                mode=mode,
                checkpoint_path=checkpoint_path,
                trial=trial,
                **kwargs,
            )

            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            self.trainer.fit(self.model, self.data_module)

        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise
        except KeyboardInterrupt:
            logger.error("Training interrupted.")
            self._status = TrainingStatus.INTERRUPTED
            raise

        if self.n_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")

        logger.info("--- Training completed successfully. ---")
        self._status = TrainingStatus.SUCCEEDED

        self.theta = (
            self.model.model.get_theta(torch.tensor(self.dataset.bow))
            .detach()
            .cpu()
            .numpy()
        )

        self.theta = self.theta / self.theta.sum(axis=1, keepdims=True)

        self.beta = self.model.model.get_beta().detach().cpu().numpy()
        self.labels = np.array(np.argmax(self.theta, axis=1))

        self.topic_dict = self.get_topic_word_dict(self.data_module.vocab)

    def get_topic_word_dict(self, vocab, num_words=100):
        """
        Get the topic-word dictionary.

        Parameters
        ----------
        vocab : list of str
            Vocabulary list corresponding to the word indices.
        num_words : int, optional
            Number of top words to retrieve for each topic, by default 100.

        Returns
        -------
        dict
            Dictionary where keys are topic indices and values are lists of tuples (word, probability).
        """

        topic_word_dict = {}
        for topic_idx, topic_dist in enumerate(self.beta):
            top_word_indices = topic_dist.argsort()[-num_words:][::-1]
            top_words_probs = [(vocab[i], topic_dist[i]) for i in top_word_indices]
            topic_word_dict[topic_idx] = top_words_probs
        return topic_word_dict

    def predict(self, dataset):
        pass

    def suggest_hyperparameters(self, trial, max_topics=100):
        self.hparams["n_topics"] = trial.suggest_int("n_topics", 1, max_topics)
        self.hparams["encoder_dim"] = trial.suggest_int("encoder_dim", 16, 512)
        self.hparams["embed_size"] = trial.suggest_int("embed_size", 16, 512)
        self.hparams["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        self.hparams["encoder_activation"] = trial.suggest_categorical(
            "encoder_activation", ["Softplus", "ReLU", "LeakyReLU", "Tanh"]
        )
        self.hparams["lr"] = trial.suggest_float("lr", 1e-5, 1e-2)
        self.hparams["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-3)

        # Map string to actual PyTorch activation function
        activation_mapping = {
            "Softplus": nn.Softplus(),
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Tanh": nn.Tanh(),
        }
        self.hparams["encoder_activation"] = activation_mapping[
            self.hparams["encoder_activation"]
        ]

        self.hparams["datamodule_args"]["batch_size"] = trial.suggest_int(
            "batch_size", 12, 512
        )

    def optimize_and_fit(
        self,
        dataset,
        min_topics=2,
        max_topics=20,
        criterion="val_loss",
        n_trials=100,
        custom_metric=None,
    ):
        """
        A new method in the child class that calls the parent class's optimize_hyperparameters method.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        min_topics : int, optional
            Minimum number of topics to evaluate, by default 2.
        max_topics : int, optional
            Maximum number of topics to evaluate, by default 20.
        criterion : str, optional
            Criterion to use for optimization ('aic', 'bic', or 'custom'), by default 'aic'.
        n_trials : int, optional
            Number of trials for optimization, by default 100.
        custom_metric : object, optional
            Custom metric object with a `score` method for evaluation, by default None.

        Returns
        -------
        dict
            Dictionary containing the best parameters and the optimal number of topics.
        """
        best_params = super().optimize_hyperparameters_neural(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion=criterion,
            n_trials=n_trials,
            custom_metric=custom_metric,
        )

        return best_params
