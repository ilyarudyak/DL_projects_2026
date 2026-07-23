from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from dataset import IMDBConfig, IMDBData

import logging
logger = logging.getLogger("imdb.model")


class IMDBModelLP(pl.LightningModule):

    def __init__(self, config,
                 data: IMDBData = None,
                 num_classes: int = 2):
        
        super().__init__()

        logger.debug(f"===MODEL CREATION===")

        # Setup data
        self.data = data
        self.vocab_size = data.vocab_size
        self.num_classes = num_classes
        logger.debug(f"Model initialized with vocab_size: {self.vocab_size}")
        
        # Save hyperparameters so we can load the model from checkpoint later
        # Ignore 'data' and do not store it in the checkpoint
        self.save_hyperparameters(ignore=['data']) 
        
        # Create a config object to hold model hyperparameters
        self.config = config
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
                                  padding_idx=self.data.PAD_ID) 
        logger.debug(f"Embedding layer created with embedding_dim: {self.config.embedding_dim}")
        logger.debug(f"Embedding layer dimensions: {self.embed.weight.shape}")
        
        # Unidirectional GRU
        self.gru = nn.GRU(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0
        )
        logger.debug(f"GRU layer created with hidden_dim: {self.config.hidden_dim}, num_layers: {self.config.num_layers}, dropout: {self.config.dropout}")
        
        # Add Layer Normalization
        self.ln = nn.LayerNorm(self.config.hidden_dim)
        logger.debug(f"LayerNorm created with hidden_dim: {self.config.hidden_dim}")

        # Output Linear layer mapping to class logits instead of vocabulary size
        self.fc = nn.Linear(self.config.hidden_dim, self.num_classes)
        logger.debug(f"Output Linear layer created mapping hidden_dim: {self.config.hidden_dim} to num_classes: {self.num_classes}")
        
        # Cross Entropy Loss for classification
        self.loss_fn = nn.CrossEntropyLoss()

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        logger.debug(f"Custom weight initialization...")
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
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

    def forward(self, x):
        logger.debug(f"===FORWARD PASS===")
        logger.debug(f"---Input shapes---")
        # Log the size of a batch size, sequence length, and embedding dimension
        logger.debug(f"(1) Batch size: {x.size(0)}, Sequence length: {x.size(1)}, Embedding dimension: {self.config.embedding_dim}")
        # x: [batch_size, seq_length]
        logger.debug(f"(1) Forward pass input shape (x): {x.shape}")

        # (1) Extract embeddings for the input sequence
        logger.debug(f"---Embeddings shapes---")
        embeddings = self.embed(x)                  # [batch_size, seq_length, embedding_dim]
        logger.debug(f"(1) Embeddings output shape: {embeddings.shape}")
        logger.debug(f"---Embeddings padding---")
        self._check_embediing_padding(x, embeddings)

        # (2) Pass embeddings through the GRU
        # outputs: [batch_size, seq_length, hidden_dim]
        # last_hidden_state: [num_layers, batch_size, hidden_dim]
        outputs, last_hidden_state = self.gru(embeddings)
        logger.debug(f"---GRU outputs---")
        logger.debug(f"(2) Hidden dimension: {self.config.hidden_dim}")
        logger.debug(f"(2) GRU output shape: {outputs.shape}")
        logger.debug(f"(2) GRU last hidden state shape: {last_hidden_state.shape}")
        # Print some values from outputs and final_outputs for debugging
        logger.debug(f"(2) Sample GRU outputs (first batch, last timestep):\n{outputs[0, -1, :5]}")
        logger.debug(f"(2) Sample final outputs (first batch):\n{last_hidden_state[1, 0, :5]}")

        # (2.1) Apply Layer Normalization to the GRU outputs
        outputs = self.ln(outputs)
        logger.debug(f"---LayerNorm outputs---")
        logger.debug(f"(2.1) LayerNorm output shape: {outputs.shape}")

        # (2.2) Extract final timestep hidden state for sequence classification
        final_outputs = outputs[:, -1, :]          # [batch_size, hidden_dim]
        logger.debug(f"---Final timestep outputs---")
        logger.debug(f"(2.2) Final timestep shape: {final_outputs.shape}")

        # (3) Map the final hidden state to category logits
        logits = self.fc(final_outputs)            # [batch_size, num_classes]
        logger.debug(f"---Logits output---")
        logger.debug(f"(3) Logits output shape: {logits.shape}")

        return logits
    
    def _check_embediing_padding(self, x, embeddings):
        # --- DIAGNOSTIC LOGGING ---
        # Find where padding tokens are in the input
        is_pad = (x == self.data.PAD_ID)
        if is_pad.any():
            # Grab all embeddings that correspond to padding tokens
            pad_embeddings = embeddings[is_pad]
            # If padding_idx is working, the max absolute value should be exactly 0.0
            max_val = pad_embeddings.abs().max().item()
            logger.debug(f"(DIAGNOSTIC) Max value in padding embeddings: {max_val:.6f}")
            if max_val == 0:
                logger.debug("(DIAGNOSTIC) SUCCESS: Padding tokens are zeroed out by Embedding layer.")
            else:
                logger.debug("(DIAGNOSTIC) WARNING: Padding tokens are NOT zero. (Check _init_weights)")

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["labels"]
        logits = self(x) # [batch_size, num_classes]
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        y = batch["labels"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # NEW: Retrieve the exact total number of steps/batches across all epochs
        total_steps = self.trainer.estimated_stepping_batches

        # Setup OneCycleLR scheduler for robust training rates
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            # total_steps=self.trainer.max_epochs,
            total_steps=total_steps, # Use estimated total steps for OneCycleLR
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # "interval": "epoch",
                "interval": "step",  # Use 'step' for OneCycleLR
            },
        }


class IMDBModelLPV2(pl.LightningModule):

    def __init__(self, config,
                 data: IMDBData = None,
                 num_classes: int = 2):
        
        super().__init__()
        logger.debug(f"===MODEL CREATION===")

        self.data = data
        self.vocab_size = data.vocab_size
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=['data']) 
        self.config = config

        # Layers
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, 
                                  embedding_dim=self.config.embedding_dim,
                                  padding_idx=self.data.PAD_ID) 
        
        self.gru = nn.GRU(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0
        )
        
        self.ln = nn.LayerNorm(self.config.hidden_dim)
        self.fc = nn.Linear(self.config.hidden_dim, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                elif 'embed' in name:
                    std = 1 / torch.sqrt(torch.tensor(self.config.embedding_dim, dtype=torch.float32))
                    nn.init.normal_(param, std=std.item())
                    if self.embed.padding_idx is not None:
                        with torch.no_grad():
                            param[self.embed.padding_idx].fill_(0.0)
                elif 'ln' in name:
                    nn.init.constant_(param, 1.0)
                else:
                    nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, attention_mask=None):
        # (1) Extract embeddings [batch_size, seq_length, embedding_dim]
        embeddings = self.embed(x)                  

        # (2) Pass padded sequences directly to GRU
        # outputs shape: [batch_size, seq_length, hidden_dim]
        outputs, _ = self.gru(embeddings)

        # (3) Apply Layer Normalization to the outputs
        outputs = self.ln(outputs)

        # (4) Slice out the hidden state at the last REAL token of each review
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)  # True length of each sequence [batch_size]
            lengths = torch.clamp(lengths, min=1) # Ensure clamp to bounds if empty review
            device = outputs.device
            # Dynamically index: outputs[batch_index, true_length - 1]
            final_outputs = outputs[torch.arange(outputs.size(0)), lengths.to(device) - 1]
        else:
            # Fallback if forward is called without mask (e.g. initial testing/summary)
            final_outputs = outputs[:, -1, :]          

        # (5) Logits mapping
        logits = self.fc(final_outputs)            
        return logits

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch.get("attention_mask") # Extract attention mask!
        y = batch["labels"]
        
        logits = self(x, attention_mask=attention_mask) 
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        attention_mask = batch.get("attention_mask") # Extract attention mask!
        y = batch["labels"]
        
        logits = self(x, attention_mask=attention_mask)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
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
                "interval": "step",
            },
        }


class IMDBModelLPPackedSeq(pl.LightningModule):

    """
    A PyTorch Lightning model for sentiment analysis using a GRU-based architecture.
    This model includes an embedding layer, a GRU layer, layer normalization, 
    and a fully connected output layer. 
    It is designed to classify movie reviews as positive or negative.
    Version 2 includes packing and unpacking of sequences for variable-length input handling.
    """

    def __init__(self, config,
                 data: IMDBData = None,
                 num_classes: int = 2):
        
        super().__init__()

        logger.debug(f"===MODEL CREATION===")

        # Setup data
        self.data = data
        self.vocab_size = data.vocab_size
        self.num_classes = num_classes
        logger.debug(f"Model initialized with vocab_size: {self.vocab_size}")
        
        # Save hyperparameters so we can load the model from checkpoint later
        # Ignore 'data' and do not store it in the checkpoint
        self.save_hyperparameters(ignore=['data']) 
        
        # Create a config object to hold model hyperparameters
        self.config = config
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
                                  padding_idx=self.data.PAD_ID) 
        logger.debug(f"Embedding layer created with embedding_dim: {self.config.embedding_dim}")
        logger.debug(f"Embedding layer dimensions: {self.embed.weight.shape}")
        
        # Unidirectional GRU
        self.gru = nn.GRU(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0
        )
        logger.debug(f"GRU layer created with hidden_dim: {self.config.hidden_dim}, num_layers: {self.config.num_layers}, dropout: {self.config.dropout}")
        
        # Add Layer Normalization
        self.ln = nn.LayerNorm(self.config.hidden_dim)
        logger.debug(f"LayerNorm created with hidden_dim: {self.config.hidden_dim}")

        # Output Linear layer mapping to class logits instead of vocabulary size
        self.fc = nn.Linear(self.config.hidden_dim, self.num_classes)
        logger.debug(f"Output Linear layer created mapping hidden_dim: {self.config.hidden_dim} to num_classes: {self.num_classes}")
        
        # Cross Entropy Loss for classification
        self.loss_fn = nn.CrossEntropyLoss()

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        logger.debug(f"Custom weight initialization...")
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
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

    def forward(self, x, attention_mask=None):
        logger.debug(f"===FORWARD PASS (PACKED)===")
        logger.debug(f"---Input shapes---")
        # Log the size of a batch size, sequence length, and embedding dimension
        logger.debug(f"Batch size: {x.size(0)}, Sequence length: {x.size(1)}, Embedding dimension: {self.config.embedding_dim}")
        # x: [batch_size, seq_length]
        logger.debug(f"Forward pass input shape (x): {x.shape}")

        # (1) Extract embeddings for the input sequence
        logger.debug(f"---Embeddings shapes---")
        embeddings = self.embed(x)                  # [batch_size, seq_length, embedding_dim]
        logger.debug(f"(1) Embeddings output shape: {embeddings.shape}")

        # logger.debug(f"---Embeddings padding---")
        # self._check_embediing_padding(x, embeddings)

      # (2) Prepare lengths and Pack
        # Lengths must be a 1D CPU int64 tensor
        lengths = attention_mask.sum(dim=1).cpu()
        logger.debug(f"---Packing sequences---")
        logger.debug(f"(2) Attention mask shape: {attention_mask.shape}")
        logger.debug(f"(2) Lengths tensor: {lengths[:5]}")
        
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )

        # (3) Pass embeddings through the GRU
        # outputs: [batch_size, seq_length, hidden_dim]
        # last_hidden_state: [num_layers, batch_size, hidden_dim]
        packed_outputs, last_hidden_state = self.gru(packed_embeddings)
        logger.debug(f"---GRU outputs---")
        logger.debug(f"(3) Hidden dimension: {self.config.hidden_dim}")
        logger.debug(f"(3) GRU output shape: {packed_outputs.data.shape}")
        logger.debug(f"(3) GRU last hidden state shape: {last_hidden_state.shape}")
        # Print some values from outputs and final_outputs for debugging
        logger.debug(f"(3) Sample GRU outputs (first batch, last timestep):\n{packed_outputs.data[0, :5]}")
        logger.debug(f"(3) Sample final outputs (first batch):\n{last_hidden_state[1, 0, :5]}")

        # (4) Unpack so we can apply LayerNorm to the sequence
        # outputs: [batch_size, seq_length, hidden_dim] (padded with zeros)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        logger.debug(f"---Unpacked GRU outputs---")
        logger.debug(f"(4) Unpacked GRU output shape: {outputs.shape}")

        # (2.1) Apply Layer Normalization to the GRU outputs
        outputs = self.ln(outputs)
        logger.debug(f"---LayerNorm outputs---")
        logger.debug(f"(2.1) LayerNorm output shape: {outputs.shape}")

        # (6) Extract the hidden state at the REAL end of each sequence
        # We use lengths-1 to get the index of the last valid token
        device = outputs.device
        final_outputs = outputs[torch.arange(outputs.size(0)), lengths.to(device) - 1]
        logger.debug(f"---Final timestep outputs---")
        logger.debug(f"(6) Final timestep shape: {final_outputs.shape}")

        # (7) Map the final hidden state to category logits
        logits = self.fc(final_outputs)            # [batch_size, num_classes]
        logger.debug(f"---Logits output---")
        logger.debug(f"(7) Logits output shape: {logits.shape}")

        return logits
    
    def _check_embediing_padding(self, x, embeddings):
        # --- DIAGNOSTIC LOGGING ---
        # Find where padding tokens are in the input
        is_pad = (x == self.data.PAD_ID)
        if is_pad.any():
            # Grab all embeddings that correspond to padding tokens
            pad_embeddings = embeddings[is_pad]
            # If padding_idx is working, the max absolute value should be exactly 0.0
            max_val = pad_embeddings.abs().max().item()
            logger.debug(f"(DIAGNOSTIC) Max value in padding embeddings: {max_val:.6f}")
            if max_val == 0:
                logger.debug("(DIAGNOSTIC) SUCCESS: Padding tokens are zeroed out by Embedding layer.")
            else:
                logger.debug("(DIAGNOSTIC) WARNING: Padding tokens are NOT zero. (Check _init_weights)")

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        y = batch["labels"]
        
        # Pass both ids and mask to forward
        logits = self(input_ids, attention_mask) 
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        y = batch["labels"]
        
        # Pass both ids and mask to forward
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # NEW: Retrieve the exact total number of steps/batches across all epochs
        total_steps = self.trainer.estimated_stepping_batches

        # Setup OneCycleLR scheduler for robust training rates
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps, # Use estimated total steps for OneCycleLR
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Use 'step' for OneCycleLR
            },
        }