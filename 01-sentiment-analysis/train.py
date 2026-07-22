import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

from dataset import IMDBConfig, IMDBData
from model import IMDBModelLP, IMDBModelLPPackedSeq

import logging
logger = logging.getLogger("imdb.trainer")


class CleanMetricsLogger(Callback):
    def __init__(self, 
                 is_val_set=True, 
                 print_every_n_epochs=1,
                print_save_notification=False,
                 skip_first_n_epochs=0
                 ):
        super().__init__()
        self.is_val_set = is_val_set
        self.print_every_n_epochs = print_every_n_epochs
        self.print_save_notification = print_save_notification
        self.skip_first_n_epochs = skip_first_n_epochs
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        # Track history for plotting later
        for key in self.history.keys():
            if key in m: self.history[key].append(m[key].item())
            
        epoch = trainer.current_epoch + 1
        if epoch % self.print_every_n_epochs == 0:
            t_loss, t_acc = m.get('train_loss'), m.get('train_acc')
            msg = f"Epoch {epoch:3d} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}"
            if self.is_val_set:
                v_loss, v_acc = m.get('val_loss'), m.get('val_acc')
                if v_loss is not None and v_acc is not None:
                    msg += f" | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}"
            print(msg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.current_epoch < self.skip_first_n_epochs:
            return
        if self.print_save_notification:
            self._save_notification(trainer)

    def _save_notification(self, trainer):
        print(f"--- Model saved at epoch {trainer.current_epoch} ---")
        m = trainer.callback_metrics
        t_loss, t_acc = m.get('train_loss'), m.get('train_acc')
        msg = f"--- Saved Model Metrics -> Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}"
        if self.is_val_set:
            v_loss, v_acc = m.get('val_loss'), m.get('val_acc')
            if v_loss is not None and v_acc is not None:
                msg += f" | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}"
        msg += " ---"
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        msg += f" | LR: {lr:.6e}"
        print(msg)

    def on_train_end(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                if callback.wait_count >= callback.patience:
                     print(f"\n🛑 Early Stopping triggered at epoch {trainer.current_epoch}")
                     break

class TrainerHighLevel:

    def __init__(self, 
                config_class = IMDBConfig,    # Inject class reference
                 data_class = IMDBData,      # Inject class reference
                 model_class = IMDBModelLPPackedSeq,     # Inject class reference
                 config: IMDBConfig = None,  # Optional: Directly pass a config instance
                 run_name: str = 'base_config',          # Optional: Name for the run 
                 config_file: str = 'base_config.yaml',
                 config_dir: str = 'configs/',
                 data_limit: int = 1000,
                 device: str = "auto",  # Default to auto-select device
                 print_every_n_epochs: int = None,
                 print_save_notification: bool = False,
                 skip_first_n_epochs: int = 0,
                 gradient_clip_val: float = 1.0):
        
        # (0) Set device
        self.device = device
        logger.debug(f"===TRAINER INITIALIZATION===")
        logger.debug(f"Device set to: {self.device}")
        
        # (1) Load config
        if config is not None and config_file is not None: 
            raise ValueError("Provide either config or config_file, not both.")

        # self.config_dir = Path(config_dir)
        # self.config_file = config_file
        
        logger.debug(f"===CONFIG INITIALIZATION===")
        
        if config is not None:
            logger.debug(f"Using provided config instance. Run name: {run_name}")
            self.config = config
            self.run_name = run_name
        else:
            logger.debug(f"Config name: {config_file}")
            self.config = config_class.from_yaml(Path(config_dir) / config_file)
        
        if print_every_n_epochs is not None:
            self.print_every_n_epochs = print_every_n_epochs
        else:
            self.print_every_n_epochs = max(1, self.config.epochs // 10)

        # (2) Create data and loaders
        self.data = data_class(config=self.config, data_limit=data_limit)
        self.is_val_set = self.data.val_split > 0.0
        self.train_loader, self.val_loader = self.data.get_loaders()

        # (3) Create model
        self.model_class = model_class
        self.model = model_class( 
                            config=self.config,
                            data=self.data,
                            )
        
        # (4) Create trainer
        logger.debug(f"===TRAINER INITIALIZATION===")
        self.print_save_notification = print_save_notification
        self.skip_first_n_epochs = skip_first_n_epochs
        self.gradient_clip_val = gradient_clip_val
        self.trainer = self._setup_trainer()
        
    def _setup_trainer(self):
        metrics_callback = CleanMetricsLogger(print_every_n_epochs=self.print_every_n_epochs, 
                                              is_val_set=self.is_val_set,
                                              skip_first_n_epochs=self.skip_first_n_epochs,
                                              print_save_notification=self.print_save_notification)
        logger.debug(f"Metrics callback created with print_every_n_epochs: {self.print_every_n_epochs}, is_val_set: {self.is_val_set}, skip_first_n_epochs: {self.skip_first_n_epochs}")

        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.monitor_metric,
            # dirpath=str(Path(self.config.checkpoint_dir) / self.config_file.replace('.yaml', '')),
            dirpath=str(Path(self.config.checkpoint_dir) / self.run_name),
            filename='best-model',
            save_top_k=1,
            mode="min"
        )
        logger.debug(f"Checkpoint callback created with monitor: {self.config.monitor_metric}, filename: {checkpoint_callback.filename}, save_top_k: {checkpoint_callback.save_top_k}, mode: {checkpoint_callback.mode}")
        
        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=self.config.patience,
            mode='min',
            verbose=True
        )
        logger.debug(f"Early stopping callback created with monitor: {self.config.monitor_metric}, patience: {self.config.patience}, mode: {'min'}, verbose: True")
        
        csv_logger = CSVLogger(
            save_dir="logs/", 
            # name=self.config_file
            name=self.run_name
        )

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            # Use the specified device if provided, otherwise let PyTorch Lightning auto-select
            accelerator=self.device if self.device else "auto",  
            logger=csv_logger, # Use CSV logger to log metrics
            enable_checkpointing=True, # Enable checkpointing to save the best model
            enable_model_summary=False, # Disable model summary to reduce verbosity
            enable_progress_bar=False, # Disable progress bar to reduce verbosity

            callbacks=[metrics_callback, checkpoint_callback, early_stop_callback],
            gradient_clip_val=self.gradient_clip_val
        )
        trainer_log_msg = f"Trainer created with max_epochs: {self.config.epochs}"
        trainer_log_msg += f", accelerator: {self.device if self.device else 'auto'}"
        trainer_log_msg += f", logger: {csv_logger is not None}"
        trainer_log_msg += f", enable_checkpointing: True"
        logger.debug(trainer_log_msg)

        return trainer
    
    def fit(self):
        device = self.trainer.strategy.root_device
        print(f"🚀 Using hardware accelerator: {device}")

        if self.is_val_set:
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
        else:
            self.trainer.fit(self.model, self.train_loader)

        best_path = self.trainer.checkpoint_callback.best_model_path
        if best_path:
            print(f"\n✅ Training finished. Loading best model State")
            
            best_score = self.trainer.checkpoint_callback.best_model_score
            if best_score is not None:
                best_score = best_score.item()
                monitor_metric = self.config.monitor_metric
                
                # Fetch history recorded by our custom CleanMetricsLogger callback
                history = None
                for cb in self.trainer.callbacks:
                    if isinstance(cb, CleanMetricsLogger):
                        history = cb.history
                        break
                
                if history and monitor_metric in history and len(history[monitor_metric]) > 0:
                    # Find the epoch index that matches the best logged metric score
                    monitored_values = history[monitor_metric]
                    best_idx = min(range(len(monitored_values)), key=lambda i: abs(monitored_values[i] - best_score))
                    best_epoch = best_idx + 1
                    
                    print(f"\n🏆 Best Model Metrics (from Epoch {best_epoch}):")
                    train_loss = history['train_loss'][best_idx]
                    train_acc = history['train_acc'][best_idx]
                    print(f"├─ Train Loss: {train_loss:.4f}")
                    print(f"├─ Train Acc:  {train_acc:.4f}")
                    
                    if self.is_val_set:
                        val_loss = history['val_loss'][best_idx]
                        val_acc = history['val_acc'][best_idx]
                        print(f"├─ Val Loss:   {val_loss:.4f}")
                        print(f"└─ Val Acc:    {val_acc:.4f}")
                    # print(f"📂 Loaded best checkpoint from: {best_path}\n")

                    # --- NEW: Call the logging function here ---
                    best_metrics = {
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss if self.is_val_set else None,
                        'val_acc': val_acc if self.is_val_set else None
                    }
                    self.save_summary_to_csv(best_epoch, best_metrics)

            # self.model = self.model_class.load_from_checkpoint(best_path)
            self.model = self.model_class.load_from_checkpoint(
                best_path,
                config=self.config,
                data=self.data
            )
        else:
            print("\n⚠️ No checkpoint found. Proceeding with last epoch state.")

    def plot_training_curves(self):
        h = self.trainer.callbacks[0].history
        epochs = range(1, len(h['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, h['train_loss'], label='Train Loss')
        if self.is_val_set: plt.plot(epochs, h['val_loss'], label='Val Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, h['train_acc'], label='Train Acc')
        if self.is_val_set: plt.plot(epochs, h['val_acc'], label='Val Acc')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.show()

    def generate(self, **kwargs):
        """Pass generation request to the model's internal generator"""
        return self.model.generate(**kwargs)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load a specific checkpoint into self.model.
        If checkpoint_path is None, try to load the best model found by the trainer.
        """
        if checkpoint_path is None:
            checkpoint_path = self.trainer.checkpoint_callback.best_model_path

        if not checkpoint_path or not Path(checkpoint_path).exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return

        print(f"📂 Loading model from {checkpoint_path}")
        self.model = self.model_class.load_from_checkpoint(checkpoint_path)
        self.model.eval()

    # Inside TrainerHighLevel.fit()
    def save_summary_to_csv(self, best_epoch, metrics):
        summary_file = "logging.csv"
        data = {
            "timestamp": [pd.Timestamp.now()],
            # "config": [self.config_file],
            "config": [self.run_name],
            "epoch": [best_epoch],
            **{k: [v] for k, v in metrics.items()}
        }
        df = pd.DataFrame(data)
        df.to_csv(summary_file, mode='a', header=not os.path.exists(summary_file), index=False)
