# Torch and PyTorch Lightning imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Standard library imports
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# Config, dataset and model imports
from src.machine_translation.config import TatoebaConfig
from src.machine_translation.dataset import TatoebaData
from src.machine_translation.model import TatoebaModelPackedSeq

# Logging setup
import logging
logger = logging.getLogger("tatoeba.trainer")


class CleanMetricsLogger(Callback):

    """
    A custom PyTorch Lightning Callback subclass for logging metrics with 3 callbacks
    1) Optionally logs a message when a model checkpoint is saved (on_save_checkpoint).
    2) Prints metrics every n epochs (on_train_epoch_end).
    3) Logs a message if early stopping is triggered (on_train_end).

    It also saves history of metrics for future plotting and analysis. 
    """

    def __init__(self, 
                 print_every_n_epochs=1,
                print_save_notification=False
                 ):
        super().__init__()

        # Store the print frequency and save notification flag
        self.print_every_n_epochs = print_every_n_epochs
        self.print_save_notification = print_save_notification

        # Initialize a custom history dictionary to store metrics for plotting
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_bleu': []}

    def on_train_epoch_end(self, trainer, pl_module):

        # Retrieve the metrics from the trainer's callback metrics
        m = trainer.callback_metrics

        # Track history for plotting later
        for key in self.history.keys():
            if key in m: self.history[key].append(m[key].item())

        # Print metrics every n epochs based on the specified frequency
        epoch = trainer.current_epoch + 1
        if epoch % self.print_every_n_epochs == 0:
            t_loss, t_acc = m.get('train_loss'), m.get('train_acc')
            msg = f"Epoch {epoch:3d} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}"

            v_loss, v_acc, v_bleu = m.get('val_loss'), m.get('val_acc'), m.get('val_bleu')
            if v_loss is not None:
                msg += f" | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}"
            if v_bleu is not None:
                    msg += f" | Val BLEU: {v_bleu:.4f}"
            print(msg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Log a message when a model checkpoint is saved.
        Only log if print_save_notification is True.
        """

        if self.print_save_notification:
            self._save_notification(trainer)

    def _save_notification(self, trainer):
        """
        helper function to print a message when a model checkpoint is saved.
        """
        print(f"--- Model saved at epoch {trainer.current_epoch} ---")
        m = trainer.callback_metrics
        t_loss, t_acc = m.get('train_loss'), m.get('train_acc')
        msg = f"--- Saved Model Metrics -> Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}"
        v_loss, v_acc = m.get('val_loss'), m.get('val_acc')
        if v_loss is not None and v_acc is not None:
            msg += f" | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}"
        msg += " ---"
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        msg += f" | LR: {lr:.6e}"
        print(msg)

    def on_train_end(self, trainer, pl_module):
        """
        Log a message if early stopping was triggered.
        """
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                if callback.wait_count >= callback.patience:
                     print(f"\n🛑 Early Stopping triggered at epoch {trainer.current_epoch}")
                     break

class ExperimentRunnerWithCustomLogging:

    """
    A wrapper class for running experiments with custom logging in PyTorch Lightning.
    1) Initializes the config, data, model, and trainer.
    2) Provides a fit method to train the model, logs metrics every n epochs and best model metrics.
    3) Provides a method to plot training curves.
    4) Provides a method to load the best model checkpoint after training.
    5) Saves the best metrics to a CSV file for future reference.
    """

    def __init__(self, 
                 config_class = TatoebaConfig, # Inject class reference
                 data_class = TatoebaData, # Inject class reference
                 model_class = TatoebaModelPackedSeq, # Inject class reference

                 seed: int = 42, # Optional: Set a global random seed in PyTorch Lightning

                 config: TatoebaConfig = None, # Optional: Directly pass a config instance
                 run_name: str = None, # Optional: Name for the run 
                 config_file: str = None,
                 config_dir: str = 'configs/',

                 data_limit: int = None, # Optional: Limit the number of data samples for quick testing
                 device: str = "auto",  # Default to auto-select device by PyTorch Lightning

                 print_every_n_epochs: int = 1, # Optional: Frequency of printing metrics during training
                 print_save_notification: bool = False, # Optional: Whether to print when a model checkpoint is saved
                 compute_bleu: bool = False  # Optional: Whether to compute BLEU score after training    
                 ):
        
        # (1) Set device parameter
        logger.debug(f"===TRAINER INITIALIZATION===")
        self.device = device
        logger.debug(f"Device set to: {self.device}")

        # (2) Set random seed for reproducibility if provided
        self.seed = seed
        pl.seed_everything(seed, workers=True)
        logger.debug(f"Random seed in Lightning set to: {self.seed}")

        # (3) Load config
        logger.debug(f"===CONFIG INITIALIZATION===")
        if config is not None and config_file is not None: 
            raise ValueError("Provide either config or config_file, not both.")
        
        if config is not None:
            logger.debug(f"Using provided config instance. Run name: {run_name}")
            self.config = config
            self.run_name = config.__class__.__name__ if run_name is None else run_name
        else:
            logger.debug(f"Config name: {config_file}")
            self.config = config_class.from_yaml(Path(config_dir) / config_file)
            self.run_name = Path(config_file).stem if run_name is None else run_name

        # (4) Set print frequency for metrics
        self.print_every_n_epochs = print_every_n_epochs

        # (5) Create data. Loaders are created within Lightning built-in lifecycle
        self.data = data_class(config=self.config, 
                               data_limit=data_limit,
                               seed=self.seed)

        # FIX: The model architecture depends on vocab_size/pad_id.
        # We must manually prepare data and setup the datamodule to load the tokenizer.
        self.data.prepare_data()
        self.data.setup()

        # (3) Create model
        self.model_class = model_class
        self.model = model_class( 
                            config=self.config,
                            tokenizer=self.data.tokenizer
                            )
        
        # (4) Create trainer
        logger.debug(f"===TRAINER INITIALIZATION===")
        self.print_save_notification = print_save_notification
        self.compute_bleu = compute_bleu
        self.gradient_clip_val = getattr(self.config, 'gradient_clip_val', 1.0)  # Default to 1.0 if not specified in config
        self.trainer = self._setup_trainer()
        
    def _setup_trainer(self):

        # Get callbacks based on scheduler type
        callbacks = self._get_callbacks()
        logger.debug(f"LR Scheduler type: {self.config.scheduler_type}.")
        logger.debug(f"Callbacks set for trainer: {[type(cb).__name__ for cb in callbacks]}")

        # Create the PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=self.device,  
            logger=False, # Disable default logger to reduce verbosity
            enable_checkpointing=True, # Enable checkpointing to save the best model
            enable_model_summary=False, # Disable model summary to reduce verbosity
            enable_progress_bar=False, # Disable progress bar to reduce verbosity
            callbacks=callbacks,
            gradient_clip_val=self.gradient_clip_val
        )
        trainer_log_msg = f"Trainer created with max_epochs: {self.config.epochs}"
        trainer_log_msg += f", accelerator: {self.device if self.device else 'auto'}"
        trainer_log_msg += f", native logger: False"
        trainer_log_msg += f", enable_checkpointing: True"
        logger.debug(trainer_log_msg)

        return trainer

    def _get_callbacks(self):

        # (1) Create metrics callback for logging training and validation metrics
        metrics_callback = CleanMetricsLogger(print_every_n_epochs=self.print_every_n_epochs, 
                                              print_save_notification=self.print_save_notification)
        logger.debug(f"Metrics callback created with print_every_n_epochs: {self.print_every_n_epochs}")

        # (2) Create checkpoint callback to save the best model based on the monitored metric
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.monitor_metric,
            dirpath=str(Path(self.config.checkpoint_dir) / self.run_name),
            filename='best-model',
            save_top_k=1,
            mode="min"
        )
        message = f"Checkpoint callback created with monitor: {self.config.monitor_metric}"
        message += f", filename: {checkpoint_callback.filename}"
        message += f", save_top_k: {checkpoint_callback.save_top_k}, mode: {checkpoint_callback.mode}"
        logger.debug(message)

        # (3) Create early stopping callback to stop training if the monitored metric doesn't improve
        early_stop_callback = EarlyStopping(
            monitor=self.config.monitor_metric,
            patience=self.config.patience,
            mode='min',
            verbose=False # Set to False to avoid duplicate native prints
        )
        message = f"Early stopping callback created with monitor: {self.config.monitor_metric}"
        message += f", patience: {self.config.patience}, mode: {'min'}, verbose: False"
        logger.debug(message)

        callbacks = [metrics_callback, checkpoint_callback]
        # Add early stopping callback only for certain scheduler types
        if self.config.scheduler_type in [TatoebaModelPackedSeq.LR_SCHEDULER_PLATEAU, 
                                          TatoebaModelPackedSeq.LR_SCHEDULER_NONE]:
            callbacks.append(early_stop_callback)

        return callbacks
            
    def fit(self, load_best_model=True):

        # (1) Print the ACTUAL device being used for training
        device = self.trainer.strategy.root_device
        print(f"🚀 Using hardware accelerator: {device}")

        # (2) Fit the model using PyTorch Lightning's Trainer with the provided data module
        self.trainer.fit(self.model, datamodule=self.data)

        # (2) RUN BLEU ONCE on the best model
        if self.compute_bleu:
            print("\n🏆 Computing BLEU score for the best model...")
            # This tells Lightning to reload the best checkpoint and run test_step 
            # using the validation dataloader
            self.trainer.test(
                model=self.model, 
                dataloaders=self.data.val_dataloader(), 
                ckpt_path='best',
                verbose=False  # This hides the "strange table"
            )

        # (4) After training,  print the best metrics and optionally load the best model checkpoint
        self._get_best_metrics(load_best_model=load_best_model)

    def _get_best_metrics(self, load_best_model=True):
        """
        Load the best model checkpoint after training and print the best metrics.
        """
        best_path = self.trainer.checkpoint_callback.best_model_path
        if best_path:
            print(f"\n✅ Training finished. Extract best model metrics.")
            
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
                    

                    val_loss = history['val_loss'][best_idx]
                    val_acc = history['val_acc'][best_idx]
                    # val_bleu = history['val_bleu'][best_idx]
                    print(f"├─ Val Loss:   {val_loss:.4f}")
                    print(f"├─ Val Acc:    {val_acc:.4f}")
                    # print(f"└─ Val BLEU:   {val_bleu:.4f}")
                    # print(f"📂 Loaded best checkpoint from: {best_path}\n")

                    # NEW: Get the BLEU score from the test run results
                    # (logged as 'final_bleu' in our model.py test_step)
                    final_bleu = self.trainer.callback_metrics.get('final_bleu')
                    if final_bleu is not None:
                        print(f"└─ Val BLEU:   {final_bleu:.4f}")

                    # --- NEW: Call the logging function here ---
                    best_metrics = {
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_bleu': final_bleu.item() if final_bleu else 0.0
                    }

                    # Save the best metrics to a CSV file for future reference
                    self._save_summary_to_csv(best_epoch, best_metrics)

            # self.model = self.model_class.load_from_checkpoint(best_path)
            if load_best_model:
                print(f"\n📂 Loading best model from checkpoint: {best_path}")
                self.model = self.model_class.load_from_checkpoint(
                    best_path,
                    config=self.config,
                    tokenizer=self.data.tokenizer
                )
        else:
            print("\n⚠️ No checkpoint found. Proceeding with last epoch state.")
            
    def plot_training_curves(self):
        h = self.trainer.callbacks[0].history
        epochs = range(1, len(h['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, h['train_loss'], label='Train Loss')
        plt.plot(epochs, h['val_loss'], label='Val Loss')
        plt.title('Loss Over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, h['train_acc'], label='Train Acc')
        plt.plot(epochs, h['val_acc'], label='Val Acc')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.show()

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load a model checkpoint. If no path is provided, load the best model checkpoint from the trainer.
        """
        if checkpoint_path is None:
            checkpoint_path = self.trainer.checkpoint_callback.best_model_path

        if not checkpoint_path or not Path(checkpoint_path).exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return

        print(f"📂 Loading model from {checkpoint_path}")
        # Update this to match the robust loading at line 297
        self.model = self.model_class.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
            tokenizer=self.data.tokenizer
        )
        self.model.eval()

    def _save_summary_to_csv(self, best_epoch, metrics):

        # Specify the path for the summary CSV file
        summary_file = Path("logs/logging.csv")
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a DataFrame with the best metrics and save it to a CSV file
        data = {
            "timestamp": [pd.Timestamp.now()],
            "config": [self.config.__class__.__name__],
            "run_name": [self.run_name],
            "epoch": [best_epoch],
            "train_loss": [metrics['train_loss']],
            "train_acc": [metrics['train_acc']],
            "val_loss": [metrics['val_loss']],
            "val_acc": [metrics['val_acc']],
            "val_bleu": [metrics['val_bleu']]
        }
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file, appending if it already exists
        df.to_csv(summary_file, 
                  mode='a',
                  # Write header only if the file does not exist to avoid duplicate headers 
                  header=not os.path.exists(summary_file), 
                  index=False)
