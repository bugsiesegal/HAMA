import os
import time
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


class HAMATrainer:
    """
    Trainer class for HAMA models.

    Handles layerwise pre-training and end-to-end training, validation,
    checkpointing, and logging.

    Args:
        model (nn.Module): HAMA model instance
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        learning_rate (float): Initial learning rate
        device (torch.device): Device to train on
        checkpoint_dir (str): Directory to save checkpoints
        use_wandb (bool): Whether to use Weights & Biases logging
        layerwise_epochs (int): Number of epochs for layerwise pre-training
        layerwise_lr (float): Learning rate for layerwise pre-training
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            learning_rate: float = 1e-4,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint_dir: str = "checkpoints",
            use_wandb: bool = False,
            layerwise_epochs: int = 10,
            layerwise_lr: float = 1e-3
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.learning_rate = learning_rate
        self.layerwise_epochs = layerwise_epochs
        self.layerwise_lr = layerwise_lr

        # Initialize optimizers and schedulers
        self._init_optimizer_scheduler()

        # Loss function (MSE for reconstruction)
        self.criterion = nn.MSELoss()

        # Early stopping
        self.early_stopping = EarlyStopping(patience=10)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_state = {
            'best_val_loss': float('inf'),
            'current_epoch': 0,
            'no_improvement_count': 0
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_state': self.training_state
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_state = checkpoint['training_state']

    def train_epoch(self) -> Dict[str, float]:
        """
        Train model for one epoch.

        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        with tqdm(total=num_batches, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Update metrics
                total_loss += loss.item()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.criterion(output, batch)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}

    def _init_optimizer_scheduler(self):
        """Initialize optimizer and scheduler for current training phase"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def _freeze_layers(self, up_to_layer: int):
        """Freeze all layers except the specified one"""
        for i, layer in enumerate(self.model.layers):
            for param in layer.parameters():
                param.requires_grad = (i == up_to_layer)

    def _unfreeze_all_layers(self):
        """Unfreeze all layers for end-to-end training"""
        for layer in self.model.layers:
            for param in layer.parameters():
                param.requires_grad = True

    def train_layer(self, layer_idx: int) -> Dict[str, float]:
        """Train a single layer"""
        self._freeze_layers(layer_idx)

        # Initialize layer-specific optimizer with higher learning rate
        self.optimizer = optim.AdamW(
            self.model.layers[layer_idx].parameters(),
            lr=self.layerwise_lr,
            weight_decay=0.01
        )

        history = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(self.layerwise_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = len(self.train_loader)

            with tqdm(total=num_batches, desc=f'Layer {layer_idx}, Epoch {epoch + 1}') as pbar:
                for batch in self.train_loader:
                    batch = batch.to(self.device)

                    x = batch
                    for prev_idx in range(layer_idx):
                        x = self.model.layers[prev_idx].encode(x)

                    output = self.model.layers[layer_idx](x)

                    # Reconstruction loss
                    loss = self.criterion(output, x)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    total_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = total_loss / num_batches

            # Validation phase
            val_metrics = self.validate_layer(layer_idx)

            if self.use_wandb:
                wandb.log({
                    f'layer_{layer_idx}_train_loss': avg_train_loss,
                    f'layer_{layer_idx}_val_loss': val_metrics['val_loss']
                })

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['val_loss'])

        return history

    def validate_layer(self, layer_idx: int) -> Dict[str, float]:
        """Validate a single layer"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                # Forward pass through all layers up to current
                x = batch
                for prev_idx in range(layer_idx):
                    x = self.model.layers[prev_idx].encode(x)

                output = self.model.layers[layer_idx](x)

                loss = self.criterion(output, x)
                total_loss += loss.item()

        return {'val_loss': total_loss / num_batches}

    def train(
            self,
            num_epochs: int,
            checkpoint_frequency: int = 5,
            early_stopping: bool = True,
            skip_layerwise: bool = False
    ) -> Dict[str, Any]:
        """
        Train model for specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train
            checkpoint_frequency (int): How often to save checkpoints
            early_stopping (bool): Whether to use early stopping

        Returns:
            Dict[str, Any]: Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        # Layerwise pre-training
        if not skip_layerwise:
            print("\nStarting layerwise pre-training...")
            for layer_idx in range(len(self.model.layers)):
                print(f"\nPre-training layer {layer_idx}")
                layer_history = self.train_layer(layer_idx)
                if self.use_wandb:
                    wandb.log({
                        f'layer_{layer_idx}_final_train_loss': layer_history['train_loss'][-1],
                        f'layer_{layer_idx}_final_val_loss': layer_history['val_loss'][-1]
                    })

            # Reset optimizer and scheduler for end-to-end training
            self._unfreeze_all_layers()
            self._init_optimizer_scheduler()
            print("\nStarting end-to-end training...")

        # Initialize W&B if requested
        if self.use_wandb:
            wandb.watch(self.model)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate()

            # Update learning rate scheduler
            self.scheduler.step(val_metrics['val_loss'])

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Log metrics
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"Epoch time: {epoch_time:.2f}s")

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Save checkpoint if this is the best model so far
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(is_best=True)
                self.training_state['no_improvement_count'] = 0
            else:
                self.training_state['no_improvement_count'] += 1

            # Regular checkpoint saving
            if (epoch + 1) % checkpoint_frequency == 0:
                self.save_checkpoint()

            # Early stopping check
            if early_stopping and self.early_stopping(val_metrics['val_loss']):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        return history

    def predict(
            self,
            input_data: torch.Tensor,
            batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate predictions using the trained model.

        Args:
            input_data (torch.Tensor): Input data
            batch_size (Optional[int]): Batch size for prediction

        Returns:
            torch.Tensor: Model predictions
        """
        self.model.eval()

        if batch_size is None:
            # If no batch size specified, process all at once
            with torch.no_grad():
                input_data = input_data.to(self.device)
                predictions = self.model(input_data)
            return predictions.cpu()

        # Process in batches
        predictions = []
        num_samples = len(input_data)

        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i + batch_size].to(self.device)
            with torch.no_grad():
                batch_predictions = self.model(batch)
            predictions.append(batch_predictions.cpu())

        return torch.cat(predictions, dim=0)

    def encodings(
            self,
            input_data: torch.Tensor,
            batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate representations using the trained model.

        Args:
            input_data (torch.Tensor): Input data
            batch_size (Optional[int]): Batch size for prediction

        Returns:
            torch.Tensor: Model representations
        """
        self.model.eval()

        if batch_size is None:
            # If no batch size specified, process all at once
            with torch.no_grad():
                input_data = input_data.to(self.device)
                representations = self.model.encode(input_data)
            return representations.cpu()

        # Process in batches
        representations = []
        num_samples = len(input_data)

        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i + batch_size].to(self.device)
            with torch.no_grad():
                batch_representations = self.model.encode(batch)
            representations.append(batch_representations.cpu())

        return torch.cat(representations, dim=0)

    def decodings(
            self,
            input_data: torch.Tensor,
            batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate reconstructions using the trained model.

        Args:
            input_data (torch.Tensor): Input data
            batch_size (Optional[int]): Batch size for prediction

        Returns:
            torch.Tensor: Model reconstructions
        """
        self.model.eval()

        if batch_size is None:
            # If no batch size specified, process all at once
            with torch.no_grad():
                input_data = input_data.to(self.device)
                reconstructions = self.model.decode(input_data)
            return reconstructions.cpu()

        # Process in batches
        reconstructions = []
        num_samples = len(input_data)

        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i + batch_size].to(self.device)
            with torch.no_grad():
                batch_reconstructions = self.model.decode(batch)
            reconstructions.append(batch_reconstructions.cpu())

        return torch.cat(reconstructions, dim=0)


def prepare_dataloaders(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        val_split: float = 0.2,
        num_workers: int = 4,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation dataloaders.

    Args:
        dataset (Dataset): Full dataset
        batch_size (int): Batch size
        val_split (float): Validation split ratio
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders
    """
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
