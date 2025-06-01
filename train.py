import os
from typing import Optional, Dict, Any, Tuple, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from tqdm import tqdm

from model import HAMAForLanguageModeling, HAMA
from matplotlib import pyplot as plt


class HAMALightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for HAMA models.

    Handles both layerwise pre-training and end-to-end training.
    Now includes support for text tasks through dimension permutation.
    """

    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-4,
            layerwise_lr: float = 1e-3,
            layerwise_epochs: int = 10,
            weight_decay: float = 0.01,
            layerwise_pretraining: bool = True,
            criterion: Optional[nn.Module] = None,
            tokenizer: Optional[Any] = None
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.layerwise_lr = layerwise_lr
        self.layerwise_epochs = layerwise_epochs
        self.weight_decay = weight_decay
        self.layerwise_pretraining = layerwise_pretraining
        self.tokenizer = tokenizer

        # Default criterion if none provided
        if criterion is None:
            # For language modeling tasks
            if isinstance(model, HAMAForLanguageModeling):
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()

        # Loss function
        self.criterion = criterion

        # Determine if we're doing a text task based on the criterion type
        self.is_text_task = isinstance(criterion, nn.CrossEntropyLoss)

        # Current training phase
        self.current_phase = "main"  # "main" or "layerwise"
        self.current_layer_idx = -1  # Only used during layerwise training

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        """Forward pass through the model"""
        # If x is a dictionary, extract the input_ids
        if isinstance(x, dict) and 'input_ids' in x:
            return self.model(x['input_ids'])
        else:
            return self.model(x)

    def configure_optimizers(self):
        """Configure optimizers and LR schedulers"""
        if self.current_phase == "layerwise":
            # Handle both direct HAMA model and wrapped model
            if isinstance(self.model, HAMAForLanguageModeling):
                params = []

                # For layer 0, include embedding parameters
                if self.current_layer_idx == 0 and hasattr(self.model, 'embedding'):
                    params.extend(self.model.embedding.parameters())

                # Add the current layer parameters - access layers directly
                params.extend(self.model.layers[self.current_layer_idx].parameters())

                optimizer = optim.AdamW(
                    params,
                    lr=self.layerwise_lr,
                    weight_decay=self.weight_decay
                )
            else:
                # Original behavior for direct HAMA model
                optimizer = optim.AdamW(
                    self.model.layers[self.current_layer_idx].parameters(),
                    lr=self.layerwise_lr,
                    weight_decay=self.weight_decay
                )
            return optimizer
        else:
            # During main training, optimize all parameters
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }

    def _unfreeze_all_layers(self):
        """Unfreeze all layers for end-to-end training"""
        # Unfreeze all parameters in the entire model
        for param in self.model.parameters():
            param.requires_grad = True

    def _main_training_step(self, batch, batch_idx):
        """Regular end-to-end training step"""
        # Extract input_ids if batch is a dictionary
        if isinstance(batch, dict) and 'input_ids' in batch:
            input_tensor = batch['input_ids']
        else:
            input_tensor = batch

        # Forward pass
        output = self(batch)  # Using our updated forward method

        # Calculate loss based on task type
        if self.is_text_task:
            loss = self._compute_text_loss(output, input_tensor)
        else:
            loss = self.criterion(output, input_tensor)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _main_validation_step(self, batch, batch_idx):
        """Regular end-to-end validation step"""
        # Extract input_ids if batch is a dictionary
        if isinstance(batch, dict) and 'input_ids' in batch:
            input_tensor = batch['input_ids']
        else:
            input_tensor = batch

        # Forward pass
        output = self(batch)

        # Calculate loss based on task type
        if self.is_text_task:
            loss = self._compute_text_loss(output, input_tensor)
        else:
            loss = self.criterion(output, input_tensor)

        if self.logger.experiment and batch_idx == 0:
            # Log output for the first batch
            if self.is_text_task:
                # For text tasks, log the output as well in the form of text
                columns = ['Input Text', 'Reconstructed Text']
                input_text = self.tokenizer.decode(input_tensor[0].cpu())
                output_text = self.tokenizer.decode(torch.argmax(output.cpu(), dim=-1)[0])
                table = wandb.Table(columns=columns, data=[[input_text, output_text]])
                self.logger.experiment.log({
                    'val_output': table
                }, commit=False)
                print(f"Input: {input_text}")
                print(f"Output: {output_text}")
            else:
                # For non-text tasks, plot the output and input
                fig, ax = plt.subplots()
                ax.plot(input_tensor[0].cpu().numpy(), label='Input')
                ax.plot(output[0].cpu().numpy(), label='Output')
                ax.legend()
                ax.set_title('Input vs Output')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                self.logger.experiment.log({
                    'val_output': fig
                }, commit=False)
                plt.close(fig)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def _compute_text_loss(self, logits, targets):
        """
        Compute loss for text tasks using dimension permutation.

        This method handles the dimensional requirements for CrossEntropyLoss
        by permuting dimensions rather than reshaping.

        Args:
            logits: Model output [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]

        Returns:
            loss: CrossEntropy loss
        """
        # Permute dimensions to match CrossEntropyLoss expectations
        # From: [batch_size, seq_len, vocab_size]
        # To:   [batch_size, vocab_size, seq_len]
        logits = logits.permute(0, 2, 1)

        # Calculate loss - CrossEntropyLoss expects:
        # - Input: [N, C, d1, d2, ...] where C is the number of classes
        # - Target: [N, d1, d2, ...] where N is the batch size
        return self.criterion(logits, targets)

    def _freeze_layers(self, up_to_layer: int):
        """Freeze all layers except the specified one"""
        # Handle both direct HAMA model and wrapped model
        if isinstance(self.model, HAMAForLanguageModeling):
            # Freeze/unfreeze embedding based on layer index
            if hasattr(self.model, 'embedding'):
                for param in self.model.embedding.parameters():
                    param.requires_grad = (up_to_layer == 0)  # Only for first layer

            # Freeze base model layers - access layers directly
            for i, layer in enumerate(self.model.layers):
                for param in layer.parameters():
                    param.requires_grad = (i == up_to_layer)

            # Always freeze LM head during layerwise training
            if hasattr(self.model, 'lm_head'):
                for param in self.model.lm_head.parameters():
                    param.requires_grad = False
        else:
            # Original behavior for direct HAMA model
            for i, layer in enumerate(self.model.layers):
                for param in layer.parameters():
                    param.requires_grad = (i == up_to_layer)

    def _layerwise_training_step(self, batch, batch_idx):
        """Layer-wise pre-training step"""
        # Extract input_ids if batch is a dictionary
        if isinstance(batch, dict) and 'input_ids' in batch:
            x = batch['input_ids']
        else:
            x = batch

        # Handle both direct HAMA model and wrapped model for layerwise training
        if isinstance(self.model, HAMAForLanguageModeling):
            # Embed input if we're using a language model
            embed = self.model.embedding(x)
            for prev_idx in range(self.current_layer_idx):
                embed = self.model.layers[prev_idx].encode(embed)  # Access layers directly
            for prev_idx in reversed(range(self.current_layer_idx)):
                embed = self.model.layers[prev_idx].decode(embed)  # Access layers directly

            # Forward pass through current layer - access layers directly
            output = self.model.lm_head(self.model.layers[self.current_layer_idx](embed))
        else:
            y = x
            for prev_idx in range(self.current_layer_idx):
                y = self.model.layers[prev_idx].encode(y)
            for prev_idx in reversed(range(self.current_layer_idx)):
                y = self.model.layers[prev_idx].decode(y)

            # Forward pass through current layer
            output = self.model.layers[self.current_layer_idx](y)

        # Compute loss (for text tasks, this should be a reconstruction loss during layerwise training)
        if self.is_text_task:
            loss = self._compute_text_loss(output, x)
        else:
            loss = self.criterion(output, x)
        self.log(f'layer_{self.current_layer_idx}_train_loss', loss, prog_bar=True)
        return loss

    def _layerwise_validation_step(self, batch, batch_idx):
        """Layer-wise pre-training validation step"""
        # Extract input_ids if batch is a dictionary
        if isinstance(batch, dict) and 'input_ids' in batch:
            x = batch['input_ids']
        else:
            x = batch

        # Handle both direct HAMA model and wrapped model
        if isinstance(self.model, HAMAForLanguageModeling):
            # Embed input if we're using a language model
            embed = self.model.embedding(x)
            for prev_idx in range(self.current_layer_idx):
                embed = self.model.layers[prev_idx].encode(embed)  # Access layers directly
            for prev_idx in reversed(range(self.current_layer_idx)):
                embed = self.model.layers[prev_idx].decode(embed)

            # Forward pass through current layer - access layers directly
            output = self.model.lm_head(self.model.layers[self.current_layer_idx](embed))
        else:
            y = x
            for prev_idx in range(self.current_layer_idx):
                y = self.model.layers[prev_idx].encode(y)
            for prev_idx in reversed(range(self.current_layer_idx)):
                y = self.model.layers[prev_idx].decode(y)

            # Forward pass through current layer
            output = self.model.layers[self.current_layer_idx](y)

        if self.logger.experiment and batch_idx == 0:
            # Log output for the first batch
            if self.is_text_task:
                # For text tasks, log the output as well in the form of text
                columns = ['Input Text', 'Reconstructed Text']
                input_text = self.tokenizer.decode(x[0].cpu())
                output_text = self.tokenizer.decode(torch.argmax(output.cpu(), dim=-1)[0])
                table = wandb.Table(columns=columns, data=[[input_text, output_text]])
                self.logger.experiment.log({
                    f'layer_{self.current_layer_idx}_val_output': table
                }, commit=False)
                print(f"Input: {input_text}")
                print(f"Output: {output_text}")
            else:
                # For non-text tasks, plot the output and input
                fig, ax = plt.subplots()
                ax.plot(x[0].cpu().numpy(), label='Input')
                ax.plot(output[0].cpu().numpy(), label='Output')
                ax.legend()
                ax.set_title('Input vs Output')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                self.logger.experiment.log({
                    f'layer_{self.current_layer_idx}_val_output': fig
                }, commit=False)
                plt.close(fig)

        # Compute loss
        if self.is_text_task:
            loss = self._compute_text_loss(output, x)
        else:
            loss = self.criterion(output, x)
        self.log(f'layer_{self.current_layer_idx}_val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    # Update these methods to handle both model types
    def training_step(self, batch, batch_idx):
        """Training step logic"""
        if self.current_phase == "layerwise":
            return self._layerwise_training_step(batch, batch_idx)
        else:
            return self._main_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        """Validation step logic"""
        if self.current_phase == "layerwise":
            return self._layerwise_validation_step(batch, batch_idx)
        else:
            return self._main_validation_step(batch, batch_idx)

    def do_layerwise_pretraining(self, trainer: pl.Trainer, datamodule: pl.LightningDataModule) -> None:
        """
        Perform layerwise pre-training before the main training.

        This version is compatible with older PyTorch Lightning versions.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer
            datamodule (pl.LightningDataModule): Data module
        """
        if not self.layerwise_pretraining:
            return

        original_max_epochs = trainer.max_epochs
        original_callbacks = trainer.callbacks

        # We'll use a simplified set of callbacks for layerwise training
        layerwise_callbacks = [cb for cb in original_callbacks
                               if not isinstance(cb, (EarlyStopping, ModelCheckpoint))]

        # Determine number of layers based on model type - access layers directly
        num_layers = len(self.model.layers)

        # Get trainer version-compatible parameters
        trainer_kwargs = {
            'max_epochs': self.layerwise_epochs,
            'callbacks': layerwise_callbacks,
            'logger': trainer.logger,
        }

        # Handle version differences for accelerator and devices
        if hasattr(trainer, 'accelerator'):
            if isinstance(trainer.accelerator, str):
                trainer_kwargs['accelerator'] = trainer.accelerator

        # For older versions, check for gpus, tpu_cores, etc.
        if hasattr(trainer, 'gpus'):
            trainer_kwargs['gpus'] = trainer.gpus
        if hasattr(trainer, 'tpu_cores'):
            trainer_kwargs['tpu_cores'] = trainer.tpu_cores

        # For newer versions that use devices
        if hasattr(trainer, 'devices'):
            trainer_kwargs['devices'] = trainer.devices

        # Strategy parameter (if available)
        if hasattr(trainer, 'strategy'):
            trainer_kwargs['strategy'] = trainer.strategy

        # Precision parameter
        if hasattr(trainer, 'precision'):
            trainer_kwargs['precision'] = trainer.precision

        for layer_idx in range(num_layers):
            print(f"\nPre-training layer {layer_idx}")

            # Set up for this layer
            self.current_phase = "layerwise"
            self.current_layer_idx = layer_idx
            self._freeze_layers(layer_idx)

            # Configure a temporary trainer for this layer with version-compatible parameters
            layer_trainer = pl.Trainer(**trainer_kwargs)

            # Train this layer
            layer_trainer.fit(self, datamodule=datamodule)

            print(f"Completed pre-training for layer {layer_idx}")

        # Reset for main training
        self.current_phase = "main"
        self.current_layer_idx = -1
        self._unfreeze_all_layers()

        # Reconfigure optimizer for main training
        self.trainer = trainer  # To ensure we can access the trainer in configure_optimizers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Generate encodings/representations"""
        return self.model.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Generate decodings/reconstructions"""
        return self.model.decode(x)


class HAMADataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for HAMA training.

    Args:
        dataset (Dataset): Full dataset
        batch_size (int): Batch size
        val_split (float): Validation split ratio
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 32,
            val_split: float = 0.2,
            num_workers: int = 4,
            seed: int = 42
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Prepare datasets"""
        if self.train_dataset is None or self.val_dataset is None:
            # Calculate split sizes
            val_size = int(len(self.dataset) * self.val_split)
            train_size = len(self.dataset) - val_size

            # Split dataset
            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )

    def train_dataloader(self):
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def create_hama_for_text(
        input_dim,
        embedding_dim,
        num_heads,
        dropout,
        num_layers,
        transformer_layers,
        initial_num_nodes,
        initial_partition_length,
        initial_n_mask,
        num_nodes_scaling_factor,
        partition_length_scaling_factor,
        n_mask_scaling_factor,
        vocab_size,
        compression_factor=1,
        use_masking=False,
):
    """
    Create a HAMA model configured for text tasks.

    Returns:
        HAMAForLanguageModeling: HAMA model wrapped for language modeling
    """
    # Create base HAMA model without embedding
    hama_base = HAMA(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers,
        transformer_layers=transformer_layers,
        initial_num_nodes=initial_num_nodes,
        initial_partition_length=initial_partition_length,
        initial_n_mask=initial_n_mask,
        num_nodes_scaling_factor=num_nodes_scaling_factor,
        partition_length_scaling_factor=partition_length_scaling_factor,
        n_mask_scaling_factor=n_mask_scaling_factor,
        compression_factor=compression_factor,
        use_masking=use_masking,
        # No external embedding here
    )

    # Wrap with language modeling components
    return HAMAForLanguageModeling(
        hama_base=hama_base,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim
    )


def train_hama_model(
        model: nn.Module,
        dataset: Dataset,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
        learning_rate: float = 1e-4,
        layerwise_lr: float = 1e-3,
        layerwise_epochs: int = 10,
        num_epochs: int = 100,
        text_task: bool = False,
        vocab_size: int = None,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        layerwise_pretraining: bool = True,
        early_stopping_patience: int = 10,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        strategy: str = "auto",
        precision: str = "16-mixed"
) -> Dict[str, Any]:
    """
    Train a HAMA model using PyTorch Lightning.

    For text tasks, the model should either:
    1. Already be a HAMAForLanguageModeling instance
    2. Be a base HAMA model that will be wrapped with HAMAForLanguageModeling
    """

    # Create criterion based on task type
    criterion = nn.CrossEntropyLoss() if text_task else nn.MSELoss()

    # Create Lightning module
    pl_model = HAMALightningModule(
        model=model,
        learning_rate=learning_rate,
        layerwise_lr=layerwise_lr,
        layerwise_epochs=layerwise_epochs,
        layerwise_pretraining=layerwise_pretraining,
        criterion=criterion
    )

    # Create data module
    data_module = HAMADataModule(
        dataset=dataset,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers
    )

    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min'
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='hama-{epoch:02d}-{val_loss:.6f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Set up logger
    logger = WandbLogger(project="hama-training") if use_wandb else True

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=False  # For better performance
    )

    # Perform layerwise pre-training if requested
    if layerwise_pretraining:
        pl_model.do_layerwise_pretraining(trainer, data_module)

    # Train model
    trainer.fit(pl_model, datamodule=data_module)

    return {
        "model": pl_model,
        "trainer": trainer,
        "best_model_path": trainer.checkpoint_callback.best_model_path,
        "best_model_score": trainer.checkpoint_callback.best_model_score
    }