# Torch and PyTorch Lightning imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.text import BLEUScore

# Standard library imports
from pathlib import Path

# Config and dataset import
from src.machine_translation.config import TatoebaConfig
from src.machine_translation.dataset import TatoebaData

# Logging setup
import logging
logger = logging.getLogger("tatoeba.model")

class TatoebaModelPackedSeq(pl.LightningModule):

    # Scheduler Constants
    LR_SCHEDULER_PLATEAU = "plateau"
    LR_SCHEDULER_COSINE = "cosine"
    LR_SCHEDULER_ONE_CYCLE = "one_cycle"
    LR_SCHEDULER_NONE = "none"

    """
    A PyTorch Lightning model for the Tatoeba dataset using a GRU-based architecture.
    1) We use packing of sequences in Encoder but not in Decoder. Rather we set 
       `ignore_index=self.pad_id` in the loss function.
    2) We use Layer Normalization after the GRU outputs to stabilize training.
    3) We use a custom weight initialization scheme for the model parameters.
    4) We use AdamW optimizer, a standard choice for NLP tasks.
    5) We support multiple learning rate schedulers: 
       ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, or no scheduler.
    6) We use torchmetrics for accuracy and BLEU score computation (BLEU is not yet implemented).

    """

    def __init__(self, 
                 config: TatoebaConfig,
                 tokenizer: TatoebaData.tokenizer,
                 ):
        
        super().__init__()

        # Save hyperparameters so we can load the model from checkpoint later
        self.save_hyperparameters(ignore=['tokenizer'])

        logger.debug(f"===MODEL CREATION===")

        # Create a config object to hold model hyperparameters
        self.config = config

        # Setup vocabulary size and padding index for the model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = tokenizer.token_to_id(TatoebaData.PAD_TOKEN)
        self.bos_id = tokenizer.token_to_id(TatoebaData.BOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(TatoebaData.EOS_TOKEN)
        logger.debug(f"Model initialized with vocab_size: {self.vocab_size:,}, pad_id: {self.pad_id}")
        
        # Log the model hyperparameters for debugging: hidden_dim, num_layers, embedding_dim, dropout
        log_message = f"Model hyperparameters: hidden_dim={self.config.hidden_dim}"
        log_message += f", num_layers={self.config.num_layers}"
        log_message += f", embedding_dim={self.config.embedding_dim}"
        log_message += f", dropout={self.config.dropout}"
        logger.debug(log_message)
        

        # Model Layers
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, 
                                  embedding_dim=self.config.embedding_dim,
                                  # Add padding_idx to ignore the padding token during training
                                  padding_idx=self.pad_id) 
        logger.debug(f"Embedding layer created with embedding_dim: {self.config.embedding_dim}")
        logger.debug(f"Embedding layer dimensions: {self.embed.weight.shape}")
        
        # Initialize encoder and decoder GRU layers with the specified hyperparameters
        self.encoder = nn.GRU(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0
        )
        message = f"Encoder created with input_size: {self.config.embedding_dim}"
        message += f", hidden_size: {self.config.hidden_dim}"
        message += f", num_layers: {self.config.num_layers}"
        message += f", dropout: {self.config.dropout if self.config.num_layers > 1 else 0.0}"
        logger.debug(message)

        self.decoder = nn.GRU(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0
        )
        message = f"Decoder created with input_size: {self.config.embedding_dim}"
        message += f", hidden_size: {self.config.hidden_dim}"
        message += f", num_layers: {self.config.num_layers}"
        message += f", dropout: {self.config.dropout if self.config.num_layers > 1 else 0.0}"
        logger.debug(message)

        
        # Add Layer Normalization
        self.ln = nn.LayerNorm(self.config.hidden_dim)
        logger.debug(f"LayerNorm created with hidden_dim: {self.config.hidden_dim}")

        # Output Linear layer mapping hidden_dim to vocabulary size
        self.fc = nn.Linear(self.config.hidden_dim, self.vocab_size)
        message = f"Linear layer created with input_dim: {self.config.hidden_dim}"
        message += f", output_dim: {self.vocab_size:,}"
        logger.debug(message)
        
        # Ignore padding index in loss computation
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)  

        # Weight initialization
        self._init_weights()

        # Initialize Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=self.vocab_size, 
            ignore_index=self.pad_id
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=self.vocab_size, 
            ignore_index=self.pad_id
        )
        # n_gram=4 is standard for BLEU
        self.val_bleu = BLEUScore(n_gram=4)

    ####################################################
    # Essential methods for PyTorch Lightning
    ####################################################

    def forward(self, 
                encoder_input_ids, 
                encoder_attention_mask, 
                decoder_input_ids):

        logger.debug(f"=== forward() call ===")
        logger.debug(f"---Input shapes---")
        logger.debug(f"Encoder input_ids shape: {encoder_input_ids.shape}")
        logger.debug(f"Encoder attention_mask shape: {encoder_attention_mask.shape}")
        logger.debug(f"Decoder input_ids shape: {decoder_input_ids.shape}")

        # (1) Compute embeddings for the source and target sequences
        logger.debug(f"---Embeddings shapes---")
        src_embeddings = self.embed(encoder_input_ids) # [batch_size, seq_length, embedding_dim]
        logger.debug(f"(1) Embeddings output shape: {src_embeddings.shape}")
        tgt_embeddings = self.embed(decoder_input_ids) # [batch_size, seq_length, embedding_dim]
        logger.debug(f"(1) Decoder embeddings output shape: {tgt_embeddings.shape}")

        # (2) Prepare lengths for packing source sequences ONLY
        # Lengths must be a 1D CPU int64 tensor
        src_lengths = encoder_attention_mask.sum(dim=1).cpu()
        logger.debug(f"---Packing sequences---")
        logger.debug(f"(2) Attention mask shape: {encoder_attention_mask.shape}")
        logger.debug(f"(2) Lengths tensor: {src_lengths[:5]}")
        
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            src_embeddings, 
            src_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )

        # (3) Pass embeddings through the Encoder
        # outputs: [batch_size, seq_length, hidden_dim]
        # last_hidden_state: [num_layers, batch_size, hidden_dim]
        encoder_packed_outputs, encoder_last_hidden_state = self.encoder(packed_embeddings)
        logger.debug(f"---GRU outputs---")
        logger.debug(f"(3) Hidden dimension: {self.config.hidden_dim}")
        logger.debug(f"(3) Encoder output shape: {encoder_packed_outputs.data.shape} Type: {type(encoder_packed_outputs)}")
        logger.debug(f"(3) Encoder last hidden state shape: {encoder_last_hidden_state.shape} Type: {type(encoder_last_hidden_state)}")

        # (4) Pass last hidden state to the Decoder 
        # last hidden state is a REGULAR tensor, not a packed one
        decoder_outputs, _ = self.decoder(tgt_embeddings, encoder_last_hidden_state)
        logger.debug(f"---Decoder outputs---")
        logger.debug(f"(4) Decoder output shape: {decoder_outputs.shape} Type: {type(decoder_outputs)}")


        # (5) Apply Layer Normalization to the Decoder outputs
        decoder_outputs = self.ln(decoder_outputs)
        logger.debug(f"---LayerNorm outputs---")
        logger.debug(f"(5) LayerNorm output shape: {decoder_outputs.shape}")

        # (6) Map the final hidden state to category logits
        logits = self.fc(decoder_outputs) # [batch_size, seq_length, vocab_size]
        logger.debug(f"---Logits output---")
        logger.debug(f"(6) Logits output shape: {logits.shape}")

        # (7) Permute logits to match the expected shape for CrossEntropyLoss
        logits = logits.permute(0, 2, 1) # [batch_size, vocab_size, seq_length]
        logger.debug(f"---Permuted logits output---")
        logger.debug(f"(7) Permuted logits output shape: {logits.shape}")

        return logits

    def training_step(self, batch, batch_idx):

        # Extract input_ids, attention_mask, and labels from the batch
        encoder_input_ids = batch["encoder_input_ids"]
        encoder_attention_mask = batch["encoder_attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_labels = batch["decoder_labels"]
        
        # Pass the inputs to the forward method to get logits
        logits = self(encoder_input_ids, encoder_attention_mask, decoder_input_ids)

        # Compute the loss using the logits and the true labels
        loss = self.loss_fn(logits, decoder_labels)
        
        # Calculate accuracy using torchmetrics
        self.train_acc(logits, decoder_labels)

        # Log the training loss and accuracy for monitoring (Lightning)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        # Extract input_ids, attention_mask, and labels from the batch
        encoder_input_ids = batch["encoder_input_ids"]
        encoder_attention_mask = batch["encoder_attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_labels = batch["decoder_labels"]
        
        # Pass the inputs to the forward method to get logits
        logits = self(encoder_input_ids, encoder_attention_mask, decoder_input_ids)

        # Compute the loss using the logits and the true labels
        loss = self.loss_fn(logits, decoder_labels)
        
        # Calculate accuracy using torchmetrics
        self.val_acc(logits, decoder_labels)

        # BLEU Score: Generate sequences and decode to strings
        generated_ids = self.generate(batch["encoder_input_ids"], batch["encoder_attention_mask"])
        preds = self.tokenizer.decode_batch(generated_ids.tolist(), skip_special_tokens=True)
        # Wrap each target in a list for BLEUScore requirements
        targets = [[t] for t in self.tokenizer.decode_batch(batch["decoder_labels"].tolist(), skip_special_tokens=True)]
        self.val_bleu(preds, targets)

        # Log the validation loss and accuracy for monitoring (Lightning)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_bleu", self.val_bleu, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Determine the scheduler type from the configuration
        sched_type = self.config.scheduler_type

        if sched_type == self.LR_SCHEDULER_PLATEAU:
            return self._configure_plateau_lr(optimizer)
        elif sched_type == self.LR_SCHEDULER_COSINE:
            return self._configure_cosine_lr(optimizer)
        elif sched_type == self.LR_SCHEDULER_ONE_CYCLE:
            return self._configure_one_cycle_lr(optimizer)
        elif sched_type == self.LR_SCHEDULER_NONE:
            return {"optimizer": optimizer}
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

    ####################################################
    # Helper methods for weight initialization and 
    # learning rate scheduling
    ####################################################

    @torch.no_grad()
    def generate(self, encoder_input_ids, encoder_attention_mask, max_length=None):
        """Greedy decoding for validation/inference."""
        batch_size = encoder_input_ids.size(0)
        max_length = max_length or self.config.max_seq_length
        
        # 1) Encode once
        src_embeddings = self.embed(encoder_input_ids)
        src_lengths = encoder_attention_mask.sum(dim=1).cpu()
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            src_embeddings, src_lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed_embeddings)
        
        # 2) Recursive decoding
        decoder_input_ids = torch.full((batch_size, 1), self.bos_id, device=self.device)
        all_tokens = []
        
        for _ in range(max_length):
            tgt_embeddings = self.embed(decoder_input_ids)
            outputs, hidden = self.decoder(tgt_embeddings, hidden)
            logits = self.fc(self.ln(outputs))
            next_token = torch.argmax(logits, dim=-1)
            all_tokens.append(next_token)
            decoder_input_ids = next_token
            # Stop if all sequences in batch reached EOS
            if (next_token == self.eos_id).all():
                break
                
        return torch.cat(all_tokens, dim=1)

    def translate(self, text: str, max_length: int = None):
        """
        A high-level utility to translate a single raw string.
        It handles tokenization, moving tensors to the correct device,
        optimized inference, and decoding.
        """
        # Ensure model is in eval mode
        self.eval()
        
        with torch.no_grad():
            # 1. Tokenize input string
            encoding = self.tokenizer.encode(text)
            # Add batch dimension [1, seq_len] and move to model device
            input_ids = torch.tensor([encoding.ids], device=self.device)
            mask = torch.tensor([encoding.attention_mask], device=self.device)

            # 2. Call the optimized generate method
            # This uses the hidden-state passing logic we reviewed
            output_ids = self.generate(input_ids, mask, max_length=max_length)

            # 3. Decode tokens back to a human-readable string
            # output_ids[0] to remove the batch dimension
            prediction = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        return prediction

    def _init_weights(self):
        """
        Custom weight initialization for the model.
        - GRU weights are initialized orthogonally.
        - Embedding weights are initialized from a normal distribution.
        - LayerNorm weights are initialized to 1.0.
        - All other weights are initialized using Kaiming normal initialization.
        - Biases are initialized to 0.0.
        """
        logger.debug(f"Custom weight initialization...")
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'encoder' in name or 'decoder' in name:
                    nn.init.orthogonal_(param)
                elif 'embed' in name:
                    std = 1 / torch.sqrt(torch.tensor(self.config.embedding_dim, dtype=torch.float32))
                    nn.init.normal_(param, std=std.item())
                    # CRITICAL: Re-zero the padding vector after manual init
                    if self.embed.padding_idx is not None:
                        with torch.no_grad():
                            param[self.embed.padding_idx].fill_(0.0)
                elif 'ln' in name:
                    # LayerNorm scale parameters are 1D; initialize to 1.0
                    nn.init.constant_(param, 1.0)
                else:
                    nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def _configure_plateau_lr(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(self.config, "scheduler_factor", 0.5),
            patience=getattr(self.config, "scheduler_patience", 1),
            min_lr=getattr(self.config, "min_lr", 1e-6),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": getattr(self.config, "monitor_metric", "val_loss"),
            },
        }

    def _configure_cosine_lr(self, optimizer):
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(
            total_steps * (getattr(self.config, "warmup_epochs", 1) / max(1, self.config.epochs))
        )
        
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max(1, total_steps - warmup_steps), 
            eta_min=getattr(self.config, "min_lr", 1e-6)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_sched, cosine_sched], 
            milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _configure_one_cycle_lr(self, optimizer):
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Use 'step' for OneCycleLR
            },
        }