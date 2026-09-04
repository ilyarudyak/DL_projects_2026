# Torch and PyTorch Lightning imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import tokenizers
import torch.nn.functional as F

# Standard library imports
from pathlib import Path
from typing import List, NamedTuple

# Config and dataset import
from src.machine_translation.config import TatoebaConfig
from src.machine_translation.dataset import TatoebaData

# Logging setup
import logging
logger = logging.getLogger("tatoeba.model")


class Hypothesis(NamedTuple):
    value: List[int]       # decoded token IDs (without BOS/EOS)
    score: float           # log-likelihood score
    text: str = ""         # decoded string (optional convenience)

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

    """

    def __init__(self, 
                 config: TatoebaConfig,
                 tokenizer: tokenizers.Tokenizer
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
        

        # (1) Initialize an embedding layer of size [vocab_size, embedding_dim]
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, 
                                  embedding_dim=self.config.embedding_dim,
                                  # Add padding_idx to ignore the padding token during training
                                  padding_idx=self.pad_id) 
        logger.debug(f"Embedding layer dimensions: {self.embed.weight.shape}")
        
        # (2) Initialize encoder and decoder GRU layers with the specified hyperparameters
        # NOT a bidirectional GRU in the current implementation
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
        # logger.debug(f"LayerNorm created with hidden_dim: {self.config.hidden_dim}")

        # (3) Initialize Output Linear layer mapping hidden_dim to vocabulary size
        self.fc = nn.Linear(self.config.hidden_dim, self.vocab_size)
        message = f"Linear layer created with input_dim: {self.config.hidden_dim}"
        message += f", output_dim: {self.vocab_size:,}"
        logger.debug(message)
        
        # (4) Initialize CrossEntropyLoss loss function with padding index ignored
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)  

        # Weight initialization
        self._init_weights()

        # Initialize Metrics
        # self.train_acc = torchmetrics.Accuracy(
        #     task="multiclass", 
        #     num_classes=self.vocab_size, 
        #     ignore_index=self.pad_id
        # )
        # self.val_acc = torchmetrics.Accuracy(
        #     task="multiclass", 
        #     num_classes=self.vocab_size, 
        #     ignore_index=self.pad_id
        # )

    ####################################################
    # Essential methods for PyTorch Lightning
    ####################################################

    def forward(self, 
                encoder_input_ids, 
                encoder_attention_mask, 
                decoder_input_ids):

        logger.debug(f"=== forward() call ===")
        logger.debug(f"---Input shapes---")
        logger.debug(f"Encoder input_ids shape: {encoder_input_ids.shape}") # [batch_size, seq_length]
        logger.debug(f"Encoder input_ids type: {type(encoder_input_ids)}")
        logger.debug(f"Encode input ids: {encoder_input_ids}")
        logger.debug(f"Encoder attention_mask shape: {encoder_attention_mask.shape}") # [batch_size, seq_length]
        logger.debug(f"Decoder input_ids shape: {decoder_input_ids.shape}") # [batch_size, seq_length]

        # (1) Compute embeddings for the source and target sequences
        logger.debug(f"---Embeddings shapes---")
        src_embeddings = self.embed(encoder_input_ids) # [batch_size, seq_length, embedding_dim]
        logger.debug(f"(1) Encoder embeddings output shape: {src_embeddings.shape}")
        tgt_embeddings = self.embed(decoder_input_ids) # [batch_size, seq_length, embedding_dim]
        logger.debug(f"(1) Decoder embeddings output shape: {tgt_embeddings.shape}")

        # (2) Prepare lengths and pack the source embeddings for the GRU encoder
        # Lengths must be a 1D CPU int64 tensor
        src_lengths = encoder_attention_mask.sum(dim=1).cpu()
        logger.debug(f"---Packing sequences---")
        # logger.debug(f"(2) Attention mask shape: {encoder_attention_mask.shape}")
        logger.debug(f"(2) Lengths tensor: {src_lengths}")
        
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            src_embeddings, 
            src_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        logger.debug(f"(2) Packed embeddings data shape: {packed_embeddings.data.shape} Type: {type(packed_embeddings)}")

        # (3) Pass PACKED embeddings through the Encoder
        # outputs: [batch_size, seq_length, hidden_dim]
        # last_hidden_state: [num_layers, batch_size, hidden_dim]
        encoder_packed_outputs, encoder_last_hidden_state = self.encoder(packed_embeddings)
        logger.debug(f"---GRU outputs---")
        logger.debug(f"(3) Hidden dimension: {self.config.hidden_dim}")
        logger.debug(f"(3) Encoder output shape: {encoder_packed_outputs.data.shape} Type: {type(encoder_packed_outputs)}")
        logger.debug(f"(3) Encoder last hidden state shape: {encoder_last_hidden_state.shape} Type: {type(encoder_last_hidden_state)}")

        # (4) Pass last hidden state to the Decoder 
        # last hidden state is a REGULAR tensor, not a packed one
        # decoder_output: [batch_size, seq_length, hidden_dim]
        decoder_outputs, _ = self.decoder(tgt_embeddings, encoder_last_hidden_state)
        logger.debug(f"---Decoder outputs---")
        logger.debug(f"(4) Decoder output shape: {decoder_outputs.shape} Type: {type(decoder_outputs)}")

        # (5) Apply Layer Normalization to the Decoder outputs
        decoder_outputs = self.ln(decoder_outputs)
        logger.debug(f"---LayerNorm outputs---")
        logger.debug(f"(5) LayerNorm output shape: {decoder_outputs.shape}")

        # (6) Map the Decoder outputs to the vocabulary logits
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
        
        # Count the number of target tokens (non-padding) for speed calculation
        target_tokens = (decoder_labels != self.pad_id).sum()

        # Log the training loss and accuracy for monitoring (Lightning)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_tokens", target_tokens, on_step=False, on_epoch=True, reduce_fx="sum")

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

        # Log the validation loss and accuracy for monitoring (Lightning)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
        # logger.debug(f"Custom weight initialization...")
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

    @torch.no_grad()
    def beam_search(
        self,
        src_text_or_ids: str | torch.Tensor,
        beam_size: int = 5,
        max_decoding_time_step: int = 50,
    ) -> List[Hypothesis]:
        """Given a single source sentence (string or 1D/2D Tensor of IDs),
        perform beam search and yield hypotheses sorted by score descending.
        """
        was_training = self.training
        self.eval()

        # 1. Encode source text into hidden state
        if isinstance(src_text_or_ids, str):
            encoding = self.tokenizer.encode(src_text_or_ids)
            encoder_input_ids = torch.tensor([encoding.ids], device=self.device)
            mask = torch.tensor([encoding.attention_mask], device=self.device)
        else:
            encoder_input_ids = src_text_or_ids.to(self.device)
            if encoder_input_ids.ndim == 1:
                encoder_input_ids = encoder_input_ids.unsqueeze(0)
            mask = (encoder_input_ids != self.pad_id).long()

        src_lengths = mask.sum(dim=1).cpu()
        src_embeddings = self.embed(encoder_input_ids)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            src_embeddings, src_lengths, batch_first=True, enforce_sorted=False
        )
        # encoder_last_hidden: [num_layers, 1, hidden_dim]
        _, encoder_last_hidden = self.encoder(packed_embeddings)

        # 2. Initialize hypotheses
        # hypotheses holds token ID lists; scores holds log probabilities
        hypotheses = [[self.bos_id]]
        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device)
        completed_hypotheses: List[Hypothesis] = []

        # Current hidden state: [num_layers, hyp_num, hidden_dim]
        h_tm1 = encoder_last_hidden  # starts with shape [num_layers, 1, hidden_dim]

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            # Last predicted token for each live beam: [hyp_num, 1]
            y_tm1 = torch.tensor([[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.embed(y_tm1)  # [hyp_num, 1, embed_dim]

            # Step decoder: output [hyp_num, 1, hidden_dim], h_t [num_layers, hyp_num, hidden_dim]
            decoder_outputs, h_t = self.decoder(y_t_embed, h_tm1)
            decoder_outputs = self.ln(decoder_outputs)
            logits = self.fc(decoder_outputs).squeeze(1)  # [hyp_num, vocab_size]
            log_p_t = F.log_softmax(logits, dim=-1)        # [hyp_num, vocab_size]

            # Calculate cumulative scores for all beam x vocab expansions
            live_hyp_num = beam_size - len(completed_hypotheses)
            continuing_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            
            # Top candidate branches
            top_cand_scores, top_cand_pos = torch.topk(continuing_hyp_scores, k=min(live_hyp_num * 2, continuing_hyp_scores.size(0)))

            prev_hyp_ids = torch.div(top_cand_pos, self.vocab_size, rounding_mode='floor')
            hyp_token_ids = top_cand_pos % self.vocab_size

            new_hypotheses = []
            live_hyp_indices = []
            new_hyp_scores = []

            for prev_idx, token_id, cand_score in zip(prev_hyp_ids, hyp_token_ids, top_cand_scores):
                prev_idx = prev_idx.item()
                token_id = token_id.item()
                cand_score = cand_score.item()

                new_seq = hypotheses[prev_idx] + [token_id]

                if token_id == self.eos_id:
                    # Strip leading BOS and trailing EOS
                    clean_ids = new_seq[1:-1]
                    completed_hypotheses.append(
                        Hypothesis(
                            value=clean_ids,
                            score=cand_score,
                            text=self.tokenizer.decode(clean_ids, skip_special_tokens=True)
                        )
                    )
                    if len(completed_hypotheses) == beam_size:
                        break
                else:
                    if len(new_hypotheses) < live_hyp_num:
                        new_hypotheses.append(new_seq)
                        live_hyp_indices.append(prev_idx)
                        new_hyp_scores.append(cand_score)

            if len(completed_hypotheses) == beam_size or len(new_hypotheses) == 0:
                break

            # Select hidden states corresponding to surviving hypotheses
            live_hyp_indices = torch.tensor(live_hyp_indices, dtype=torch.long, device=self.device)
            # h_t is [num_layers, hyp_num, hidden_dim]; slice along dim=1 (batch/hyp dimension)
            h_tm1 = h_t[:, live_hyp_indices, :]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        # Fallback if no hypothesis reached EOS
        if len(completed_hypotheses) == 0:
            clean_ids = hypotheses[0][1:]
            completed_hypotheses.append(
                Hypothesis(
                    value=clean_ids,
                    score=hyp_scores[0].item(),
                    text=self.tokenizer.decode(clean_ids, skip_special_tokens=True)
                )
            )

        completed_hypotheses.sort(key=lambda h: h.score, reverse=True)

        if was_training:
            self.train()

        return completed_hypotheses