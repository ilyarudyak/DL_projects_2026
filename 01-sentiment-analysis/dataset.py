from fileinput import filename
import random
from scipy import datasets
from sympy import sequence
import yaml
import numpy as np

from collections import Counter

import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

import spacy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
import tokenizers

import os
import pandas as pd
from datetime import datetime

logger = logging.getLogger("imdb.dataset")

@dataclass
class IMDBConfig:

    """
    Configuration class for the IMDB sentiment analysis dataset and model.
    Parameters are the same as in file "base_config.yaml".
    """

    ###########################################################################
    # 1. Dataset & Paths
    ###########################################################################

    data_path: str = "datasets/imdb"
    checkpoint_dir: str = "checkpoints/"
    tokenizer_path: str = "datasets/tokenizers/"  # Path to save/load the trained BPE tokenizer

    max_seq_length: int = 500          # Geron: 500
    pretrained_tokenizer: str = "gpt2" # Type of pretrained tokenizer to use 

    max_vocab_size: int = 1000         # We DO NOT use this with a pre-trained tokenizer

    ###########################################################################
    # 2. Model Hyperparameters 
    ###########################################################################

    hidden_dim: int = 64              # Geron: 64
    num_layers: int = 2                # Geron: 2
    embedding_dim: int = 128           # Geron: 128
    dropout: float = 0.4               # Geron: 0.2

    ###########################################################################
    # 3. Training Hyperparameters
    ###########################################################################

    batch_size: int = 512              # Geron: 256
    learning_rate: float = 0.001       # Geron: torch.optim.NAdam(lr=0.001)
    epochs: int = 15                   # Geron: 10
    weight_decay: float = 0.01         # Geron: NOT used

    ###########################################################################
    # 4. Training Flow
    ###########################################################################

    patience: int = 3                  #
    gradient_clip_val: float = 1.0
    monitor_metric: str = 'val_loss'

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))
        

class IMDBData:

    # EOS_TOKEN = "\n"  # End-of-sequence token for names
    DATA_DIR = Path(".")

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    TEXT = "text"
    LABEL = "label"

    PAD_TOKEN = "<pad>"
    PAD_TOKEN_BERT = "[PAD]"
    PAD_ID = 0
    UNK_TOKEN = "<unk>"
    UNK_TOKEN_BERT = "[UNK]"


    END_OF_WORD_TOKEN = "</w>"
    TOKENIZER_FILE = "tokenizer.json"

    INPUT_IDS = "input_ids"
    LABELS = "labels"
    ATTENTION_MASK = "attention_mask"

    GPT2_TOKENIZER = "gpt2"
    BERT_TOKENIZER = "bert-base-uncased"
    
    def __init__(self, 
                 config: IMDBConfig, 
                 train_split: float = 0.8,
                 val_split: float = 0.2,
                 with_padding: bool = True,
                 train_tokenizer: bool = False,
                 data_limit: int = 1000,
                 seed: int = 42):

        # Setup configuration and debug mode
        self.config: IMDBConfig = config
        self.seed: int = seed

        # Load the dataset from Hugging Face datasets library
        self.imdb_data = self._load_dataset(limit=data_limit)

        # Setup train/val split ratios
        self.train_split: float = train_split
        self.val_split: float = val_split
        logger.debug(f"===DATASET SPLIT===")
        logger.debug(f"Train/val split ratios: {self.train_split}/{self.val_split}")
        self.imdb_train_set, self.imdb_val_set = self._split_dataset()

        # Create BPE tokenizer and train it on the training set reviews
        self.with_padding: bool = with_padding
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
        self.train_reviews = [review[self.TEXT].lower() for review in self.imdb_train_set]

        self.bpe_tokenizer = None
        self.bbpe_tokenizer = None
        if train_tokenizer:
            # self.bpe_tokenizer = self._create_bpe_tokenizer()
            self.bpe_tokenizer = self._create_bpe_tokenizer_v2()
            self.bbpe_tokenizer = self._create_bbpe_tokenizer()
        elif self.config.pretrained_tokenizer == self.GPT2_TOKENIZER:
            self.bbpe_tokenizer = self._load_gpt2_tokenizer()
        elif self.config.pretrained_tokenizer == self.BERT_TOKENIZER:
            self.bbpe_tokenizer = self._load_bert_tokenizer()
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.config.pretrained_tokenizer}")

    @property
    def vocab_size(self) -> int:
        """Returns the total number of tokens in the vocabulary."""
        if self.bbpe_tokenizer is not None:
            return self.bbpe_tokenizer.get_vocab_size()
        return 0

    def _load_dataset(self, limit: int = 1000):
        # Load the IMDB dataset using the Hugging Face datasets library

        # Setup a data directory for the dataset
        data_dir = self.DATA_DIR / Path(self.config.data_path)

        # 1. Load the full dataset (will use cache if available)
        imdb_data = load_dataset(
            "stanfordnlp/imdb", 
            cache_dir=data_dir,
            download_mode="reuse_dataset_if_exists" 
        )

        # 2. Shuffle and limit the dataset
        if limit is not None:
            imdb_data['train'] = imdb_data['train'].shuffle(seed=self.seed).select(range(min(limit, len(imdb_data['train']))))
            imdb_data['test'] = imdb_data['test'].shuffle(seed=self.seed).select(range(min(limit // 2, len(imdb_data['test']))))

        logger.debug(f"===DATASET LOADED===")
        logger.debug(f"Dataset splits: {imdb_data.keys()}")
        logger.debug(f"Number of training samples (original dataset): {len(imdb_data[self.TRAIN])}")
        logger.debug(f"Number of test samples (original dataset): {len(imdb_data[self.TEST])}")

        return imdb_data
    
    def _split_dataset(self):

        # Split the dataset into training and validation sets based on the specified ratios
        split = self.imdb_data[self.TRAIN].train_test_split(train_size=self.train_split, seed=self.seed)
        imdb_train_set, imdb_valid_set = split[self.TRAIN], split[self.TEST]

        logger.debug(f"Number of training samples (after split): {len(imdb_train_set)}")
        logger.debug(f"Number of validation samples (after split): {len(imdb_valid_set)}")

        # We currently do NOT use a test set
        # imdb_test_set = self.imdb_data[self.TEST]

        return imdb_train_set, imdb_valid_set
    
    def _create_bpe_tokenizer(self):

        logger.debug(f"===BPE TOKENIZER TRAINING===")

        # Create a BPE model tokenizer using the Hugging Face tokenizers library
        bpe_model = tokenizers.models.BPE(unk_token=self.UNK_TOKEN)

        # Setup a pre-tokenizer to split the text into words based on whitespace
        bpe_tokenizer = tokenizers.Tokenizer(bpe_model)
        bpe_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        # Create a BPE trainer to train the tokenizer on the training set reviews
        bpe_trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=self.config.max_vocab_size,
            special_tokens=self.special_tokens)
        logger.debug(f"Training BPE tokenizer with vocab size: {self.config.max_vocab_size} and special tokens: {self.special_tokens}")
        
        # Create a list of all reviews in the training set, converted to lowercase
        # train_reviews = [review["text"].lower() for review in self.imdb_train_set]
        # logger.debug(f"Number of training reviews: {len(train_reviews)}")

        # MAIN step: Train the BPE tokenizer on the training set reviews
        bpe_tokenizer.train_from_iterator(self.train_reviews, bpe_trainer)
        logger.debug(f"BPE tokenizer training completed. Vocabulary size: {len(bpe_tokenizer.get_vocab())}")

        # Add the PAD token to the tokenizer's vocabulary and set its ID to 0
        if self.with_padding:
            bpe_tokenizer.enable_padding(pad_id=self.PAD_ID, pad_token=self.PAD_TOKEN)
            bpe_tokenizer.enable_truncation(max_length=self.config.max_seq_length)

        return bpe_tokenizer
    
    def _create_bpe_tokenizer_v2(self):
        """
        1. The Suffix Approach (Replicating your </w>). This is the most common way 
        to fix the "inappropriate space" issue while still using the Whitespace() 
        pre-tokenizer. You tell the BPE model to automatically append a suffix 
        to every word it receives from the pre-tokenizer. 
        Original approach with suffix (Sennrich 2016 paper).
        """

        logger.debug(f"===BPE TOKENIZER TRAINING===")

        # Create a BPE model tokenizer using the Hugging Face tokenizers library
        bpe_model = tokenizers.models.BPE(
            unk_token=self.UNK_TOKEN,
            end_of_word_suffix=self.END_OF_WORD_TOKEN)  # Add the end-of-word suffix to the BPE model

        # Setup a pre-tokenizer to split the text into words based on whitespace
        bpe_tokenizer = tokenizers.Tokenizer(bpe_model)
        bpe_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

        # Create a BPE trainer to train the tokenizer on the training set reviews
        bpe_trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=self.config.max_vocab_size,
            special_tokens=self.special_tokens,
            end_of_word_suffix=self.END_OF_WORD_TOKEN)  # Add the end-of-word suffix to the BPE trainer
        logger.debug(f"Training BPE tokenizer with vocab size: {self.config.max_vocab_size} and special tokens: {self.special_tokens}")
        
        # Add a Decoder so that .decode() knows what to do with the suffix
        bpe_tokenizer.decoder = tokenizers.decoders.BPEDecoder(suffix=self.END_OF_WORD_TOKEN)
        logger.debug(f"Decoder added to BPE tokenizer with suffix '{self.END_OF_WORD_TOKEN}'")

        # Create a list of all reviews in the training set, converted to lowercase
        # train_reviews = [review["text"].lower() for review in self.imdb_train_set]
        # logger.debug(f"Number of training reviews: {len(train_reviews)}")

        # MAIN step: Train the BPE tokenizer on the training set reviews
        bpe_tokenizer.train_from_iterator(self.train_reviews, bpe_trainer)
        logger.debug(f"BPE tokenizer training completed. Vocabulary size: {len(bpe_tokenizer.get_vocab())}")

        # Add the PAD token to the tokenizer's vocabulary and set its ID to 0
        if self.with_padding:
            bpe_tokenizer.enable_padding(pad_id=self.PAD_ID, pad_token=self.PAD_TOKEN)
            bpe_tokenizer.enable_truncation(max_length=self.config.max_seq_length)

        return bpe_tokenizer
    
    def _create_bbpe_tokenizer(self):

        logger.debug(f"===BBPE TOKENIZER TRAINING===")
        # Create a BBPE model tokenizer using the Hugging Face tokenizers library
        bbpe_model = tokenizers.models.BPE(unk_token=self.UNK_TOKEN)

        # Setup a pre-tokenizer to split the text into bytes using ByteLevel
        bbpe_tokenizer = tokenizers.Tokenizer(bbpe_model)

        # Setup a pre-tokenizer to split the text into bytes using ByteLevel
        bbpe_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()

        # Create a BBPE trainer to train the tokenizer on the training set reviews
        bbpe_trainer = tokenizers.trainers.BpeTrainer(vocab_size=self.config.max_vocab_size,
                                                      special_tokens=self.special_tokens)
        logger.debug(f"Training BBPE tokenizer with vocab size: {self.config.max_vocab_size} and special tokens: {self.special_tokens}")
        
        # Train the BBPE tokenizer on the training set reviews
        bbpe_tokenizer.train_from_iterator(self.train_reviews, bbpe_trainer)
        logger.debug(f"BBPE tokenizer training completed. Vocabulary size: {len(bbpe_tokenizer.get_vocab())}")

        return bbpe_tokenizer
    
    def _load_tokenizer_from_file_or_download(self, tokenizer_name):
        """
        Load a low-level tokenizer from a local tokenizer.json file.
        Download and save it when the file does not exist.
        """

        logger.debug(f"===LOADING PRETRAINED TOKENIZER ({tokenizer_name})===")

        tokenizer_path = Path(self.config.tokenizer_path) / tokenizer_name
        base_save_path = self.DATA_DIR / tokenizer_path
        vocab_file = base_save_path / self.TOKENIZER_FILE

        if vocab_file.exists():
            logger.debug(f"Using local tokenizer found at: {tokenizer_path / self.TOKENIZER_FILE}")
            tokenizer = tokenizers.Tokenizer.from_file(str(vocab_file))
        else:
            logger.debug(f"Pretrained tokenizer not found locally. Downloading '{tokenizer_name}'...")
            tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)
            base_save_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(vocab_file))
            logger.debug(f"Downloaded tokenizer saved to {vocab_file}")

        # Log the tokenizer's vocabulary size for debugging purposes
        logger.debug(f"Tokenizer '{tokenizer_name}' loaded. Vocabulary size: {tokenizer.get_vocab_size()}")

        return tokenizer
    
    def _load_gpt2_tokenizer(self):
        """
        Loads a pretrained BBPE tokenizer (e.g., GPT-2).
        If not found in the local directory, it downloads it and saves a copy.
        """
        # 1. Load the pretrained GPT-2 tokenizer from a local file or download it
        tokenizer = self._load_tokenizer_from_file_or_download(self.GPT2_TOKENIZER)

        # 2. Configure Padding/Truncation 
        if self.with_padding:
            # Note: GPT-2 doesn't have a PAD token by default. 
            # Since we are fine-tuning/training, we can register your PAD_TOKEN.
            tokenizer.add_special_tokens(self.special_tokens)
            # This is just a temporary fix!!!
            # Set the PAD_ID to the ACTUAL ID of the PAD_TOKEN in the tokenizer's vocabulary
            self.PAD_ID = tokenizer.token_to_id(self.PAD_TOKEN)

            logger.debug(f"<pad> token ID: {self.PAD_ID}")
            logger.debug(f"<unk> token ID: {tokenizer.token_to_id(self.UNK_TOKEN)}")
            
            tokenizer.enable_padding(pad_id=self.PAD_ID, pad_token=self.PAD_TOKEN)
            tokenizer.enable_truncation(max_length=self.config.max_seq_length)

        return tokenizer
    
    def _load_bert_tokenizer(self):
        """
        Loads a pretrained BERT tokenizer.
        """

        # 1. Load the pretrained BERT tokenizer from a local file or download it
        tokenizer = self._load_tokenizer_from_file_or_download(self.BERT_TOKENIZER)

        # 2. Configure Padding/Truncation 
        if self.with_padding:

            # Set the PAD_ID to the ACTUAL ID of the PAD_TOKEN in the tokenizer's vocabulary
            self.PAD_ID = tokenizer.token_to_id(self.PAD_TOKEN_BERT)

            # Log the PAD and UNK token IDs for debugging purposes
            unk_id = tokenizer.token_to_id(self.UNK_TOKEN_BERT)
            logger.debug(f"BERT vocabulary size: {tokenizer.get_vocab_size()}")
            logger.debug(f"{self.PAD_TOKEN_BERT} token ID: {self.PAD_ID}")
            logger.debug(f"{self.UNK_TOKEN_BERT} token ID: {unk_id}")
            logger.debug(f"Token at ID 0: {tokenizer.id_to_token(0)}")

            # Enable padding and truncation for the BERT tokenizer
            tokenizer.enable_padding(pad_id=self.PAD_ID, pad_token=self.PAD_TOKEN_BERT)
            tokenizer.enable_truncation(max_length=self.config.max_seq_length)

        return tokenizer
    
    def get_loaders(self):

        def collate_fn(batch):

            logger.debug(f"===COLLATE_FN===")
            # Log the shape of the batch before processing
            logger.debug(f"Batch type: {type(batch)}, Batch size: {len(batch)}")
            # Log the type and shape of the first item in the batch
            logger.debug(f"First item type: {type(batch[0])}, Keys: {batch[0].keys()}")
            logger.debug(f"First item text length: {len(batch[0][self.TEXT])}, First item label: {batch[0][self.LABEL]}")

            # 1. Extract texts and labels from the list of dicts
            texts = [item[self.TEXT] for item in batch]
            labels = [item[self.LABEL] for item in batch]
            
            # 2. Tokenize the batch using your BBPE tokenizer
            # Since you used .enable_padding() and .enable_truncation() 
            # in _load_pretrained_bbpe_tokenizer, this handles everything.
            # Returns a list of Encoding objects with the following attributes: 
            # ids, attention_mask, type_ids, tokens, offsets, word_ids
            encodings = self.bbpe_tokenizer.encode_batch(texts) # Returns a list of Encoding objects

            # Log the shape of the encodings after tokenization
            logger.debug(f"Encodings type: {type(encodings)}, Number of encodings: {len(encodings)}")
            # Log the type and length of the first encoding in the batch
            logger.debug(f"First encoding type: {type(encodings[0])}")
            # Log the type and shape

            # Log the length of the first encoding's and its label
            logger.debug(f"First encoding Length: {len(encodings[0].ids)} First Label: {labels[0]}")
            
            # 3. Convert to tensors
            # Unpack an Encoding object to get the ids, attention_mask and labels. 
            input_ids = torch.tensor([e.ids for e in encodings], dtype=torch.long)
            attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        # Create dataloaders for each split
        batch_size = self.config.batch_size
        train_loader = torch.utils.data.DataLoader(self.imdb_train_set, 
                                                   batch_size=batch_size, 
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(self.imdb_val_set, 
                                                 batch_size=batch_size, 
                                                 shuffle=False, 
                                                 collate_fn=collate_fn)

        logger.debug(f"===DATA LOADERS CREATED===")
        logger.debug(f"Train loader: {len(train_loader)} batches, {len(self.imdb_train_set)} samples")
        logger.debug(f"Validation loader: {len(val_loader)} batches, {len(self.imdb_val_set)} samples")

        return train_loader, val_loader
    
