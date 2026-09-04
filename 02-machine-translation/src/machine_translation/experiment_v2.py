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
import time
import sacrebleu
from tqdm import tqdm


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
        self.history = {'train_loss': [], 
                        'val_loss': [], 
                        'train_tokens_per_second': []}

        # Initialize a start time for tracking training duration
        self.start_time = None
        self.epoch_start_time = None

    def on_train_start(self, trainer, pl_module):
        # Record the start time when training begins
        self.start_time = time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        elapsed_time = time.perf_counter() - self.epoch_start_time
        train_tokens = metrics.get("train_tokens")

        tokens_per_second = None
        if train_tokens is not None and elapsed_time > 0:
            tokens_per_second = train_tokens.item() / elapsed_time

        for key in ("train_loss", "val_loss"):
            value = metrics.get(key)
            if value is not None:
                self.history[key].append(value.item())

        if tokens_per_second is not None:
            self.history["train_tokens_per_second"].append(tokens_per_second)

        epoch = trainer.current_epoch + 1

        if epoch % self.print_every_n_epochs == 0:
            train_loss = metrics.get("train_loss")
            val_loss = metrics.get("val_loss")

            message = f"Epoch {epoch:3d}"

            if train_loss is not None:
                message += f" | Train Loss: {train_loss.item():.4f}"

            if val_loss is not None:
                message += f" | Val Loss: {val_loss.item():.4f}"

            if tokens_per_second is not None:
                message += f" | Tokens/s: {tokens_per_second:,.1f}"

            message += f" | Elapsed Time: {elapsed_time:.0f}s"
            print(message)

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
        t_loss = m.get('train_loss')
        msg = f"--- Saved Model Metrics -> Train Loss: {t_loss:.4f}"
        v_loss = m.get('val_loss')
        if v_loss is not None:
            msg += f" | Val Loss: {v_loss:.4f}"
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
                    print(f"├─ Train Loss: {train_loss:.4f}")

                    val_loss = history['val_loss'][best_idx]
                    print(f"├─ Val Loss:   {val_loss:.4f}")

                    # --- NEW: Call the logging function here ---
                    best_metrics = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
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

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, h['train_loss'], label='Train Loss')
        plt.plot(epochs, h['val_loss'], label='Val Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
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
            "val_loss": [metrics['val_loss']],
        }
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file, appending if it already exists
        df.to_csv(summary_file, 
                  mode='a',
                  # Write header only if the file does not exist to avoid duplicate headers 
                  header=not os.path.exists(summary_file), 
                  index=False)

    def evaluate_ppl(self, dataloader=None):
        """
        Compute token-level perplexity over the validation dataset.

        Perplexity is calculated from the total negative log-likelihood
        divided by the total number of non-padding target tokens.
        """
        if dataloader is None:
            dataloader = self.data.val_dataloader()

        was_training = self.model.training
        self.model.eval()

        total_nll = 0.0
        total_tokens = 0

        with torch.inference_mode():
            for batch in dataloader:
                encoder_input_ids = batch["encoder_input_ids"].to(self.model.device)
                encoder_attention_mask = batch["encoder_attention_mask"].to(self.model.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.model.device)
                decoder_labels = batch["decoder_labels"].to(self.model.device)

                logits = self.model(
                    encoder_input_ids,
                    encoder_attention_mask,
                    decoder_input_ids,
                )

                # Sum losses over valid target tokens in this batch.
                batch_nll = torch.nn.functional.cross_entropy(
                    logits,
                    decoder_labels,
                    ignore_index=self.model.pad_id,
                    reduction="sum",
                )

                batch_tokens = (decoder_labels != self.model.pad_id).sum()

                total_nll += batch_nll.item()
                total_tokens += batch_tokens.item()

        if was_training:
            self.model.train()

        if total_tokens == 0:
            raise ValueError("Cannot compute perplexity: no non-padding target tokens found.")

        average_nll = total_nll / total_tokens
        perplexity = torch.exp(torch.tensor(average_nll)).item()

        return {
            "perplexity": perplexity,
            "average_nll": average_nll,
            "tokens": total_tokens,
        }

    def decode(
        self,
        split: str = "val",             # 'val' or 'test'
        beam_size: int = 5,
        max_samples: int = None,
        compute_bleu: bool = True,
        max_decoding_time_step: int = 50,
    ):
        """Decodes the validation or test split using Beam Search,
        prints sample translations, and computes the corpus BLEU score.
        """
        # 1. Select dataset split
        raw_data = self.data.val_data if split == "val" else self.data.test_data
        if max_samples:
            raw_data = raw_data[:max_samples]

        self.model.eval()
        hypotheses_text = []
        references_text = []

        print(f"\n🔍 Decoding {len(raw_data)} sentences from '{split}' set (beam_size={beam_size})...")

        for sample in tqdm(raw_data, desc="Decoding"):
            src_text = sample["source_text"]
            tgt_text = sample["target_text"]

            hyps = self.model.beam_search(
                src_text,
                beam_size=beam_size,
                max_decoding_time_step=max_decoding_time_step
            )

            top_hyp_text = hyps[0].text
            hypotheses_text.append(top_hyp_text)
            references_text.append(tgt_text)

        # 2. Print a few sample predictions
        print("\n--- Sample Translations ---")
        for i in range(min(5, len(hypotheses_text))):
            print(f"Source:    {raw_data[i]['source_text']}")
            print(f"Target:    {references_text[i]}")
            print(f"Predicted: {hypotheses_text[i]}")
            print("-" * 30)

        # 3. Compute SacreBLEU
        results = {"hypotheses": hypotheses_text, "references": references_text}
        if compute_bleu:
            # sacrebleu expects references as a list of reference streams: [refs]
            bleu = sacrebleu.corpus_bleu(hypotheses_text, [references_text])
            results["bleu"] = bleu.score
            print(f"\n📊 Corpus BLEU score: {bleu.score:.2f}")

        return results
