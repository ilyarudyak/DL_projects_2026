# Torch and PyTorch Lightning imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

# Standard library imports
from pathlib import Path

# Hugging Face datasets library import
from datasets import load_dataset
import tokenizers

# Config import
from src.machine_translation.config import TatoebaConfig

# Logging setup
import logging
logger = logging.getLogger("tatoeba.dataset")


class TatoebaData(pl.LightningDataModule):

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(self, 
                 config: TatoebaConfig = None,
                 train_size: float = 0.8,
                 data_limit: int = None,
                 seed: int = 42):
        super().__init__()

        logger.debug(f"=== DATASET CREATION ===")

        # Store the configuration and set up paths
        self.config: TatoebaConfig = config
        self.data_path: Path = Path(config.data_path)

        # Initialize dataset attributes
        self.train_size: float = train_size
        self.data_limit: int = data_limit
        self.seed: int = seed
        self.train_data: torch.utils.data.Dataset = None
        self.val_data: torch.utils.data.Dataset = None
        self.test_data: torch.utils.data.Dataset = None

        # Initialize tokenizer attribute
        self.tokenizer: tokenizers.Tokenizer = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id(self.PAD_TOKEN)

    @property
    def bos_id(self):
        return self.tokenizer.token_to_id(self.BOS_TOKEN)

    @property
    def eos_id(self):
        return self.tokenizer.token_to_id(self.EOS_TOKEN)

    def prepare_data(self):
        """
        Global setup (GPU 0 / Process 0 only):
        Download data to disk and train/save tokenizer to disk.
        """
        logger.debug(f"=== prepare_data() call ===")

        # Load the dataset and split it FROM Hugging Face
        train_data, _, _ = self._load_and_split_data()
        logger.debug(f"Loaded ONLY train data: {len(train_data):,}")

        # Train and save the tokenizer if it doesn't already exist
        self._train_and_save_tokenizer(train_data)

    def _load_and_split_data(self):
        """
        Per-process setup (Runs on ALL GPUs / processes):
        Loads datasets from disk cache into RAM and sets up splits.
        """

        # Logging the data loading and splitting process
        logger.debug(f"=== _load_and_split_data() call ===")
        cache_path = Path(self.data_path)
        # Check if HF cache files or directory exist
        if cache_path.exists() and any(cache_path.iterdir()):
            logger.debug(f"Loading Tatoeba dataset from local disk cache: {cache_path}")
        else:
            logger.debug(f"Local cache not found at {cache_path}. Downloading Tatoeba dataset from Hugging Face...")

        # Load the original validation and test sets from the Hugging Face datasets library
        nmt_original_valid_set, nmt_test_set = load_dataset(
            path="ageron/tatoeba_mt_train", 
            name="eng-spa",
            split=["validation", "test"],
            cache_dir=str(self.data_path),
            download_mode="reuse_dataset_if_exists")

        # Keep a deterministic subset for quick experiments.
        if self.data_limit is not None:
            nmt_original_valid_set = (
                nmt_original_valid_set
                .shuffle(seed=self.seed)
                .select(range(min(self.data_limit, len(nmt_original_valid_set))))
            )

        # Split the original validation set into training and validation sets based on the specified train_size
        split = nmt_original_valid_set.train_test_split(train_size=self.train_size, seed=self.seed)

        # Store the resulting splits in the dataset attributes
        train_data = split["train"]
        val_data = split["test"]
        test_data = nmt_test_set

        return train_data, val_data, test_data

    def setup(self, stage=None):
        """
        Per-process setup (Runs on ALL GPUs / processes):
        Loads datasets from disk cache into RAM and sets up splits.
        """
        logger.debug(f"=== setup() call ===")

        # Load and split the dataset FROM disk if not already done in other stages
        # Setup attributes in these processes.
        if self.train_data is None:
            train_data, val_data, test_data = self._load_and_split_data()
            self.train_data = train_data
            self.val_data = val_data
            self.test_data = test_data

        # Log the sizes of the loaded datasets
        message = f"Loaded Tatoeba dataset: train={len(self.train_data):,}"
        message += f", validation={len(self.val_data):,}, test={len(self.test_data):,}"
        logger.debug(message)

        # Load the tokenizer from disk if it exists, otherwise train a new one
        self._load_tokenizer_from_disk()

    def _train_tokenizer(self, train_data):
        """Train one shared BPE tokenizer on English and Spanish training text."""

        def training_texts():
            """Yield all source and target texts from the training dataset for tokenizer training."""
            for entry in train_data:
                yield entry["source_text"]
                yield entry["target_text"]

        logger.debug(f"=== _train_tokenizer() call ===")
        message = f"Training BPE tokenizer with max_vocab_size={self.config.max_vocab_size:,}"
        message += f" and max_seq_length={self.config.max_seq_length}..."
        logger.debug(message)

        # Train a BPE tokenizer from scratch using the training texts
        tokenizer_model = tokenizers.models.BPE(unk_token="<unk>")
        tokenizer = tokenizers.Tokenizer(tokenizer_model)
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tokenizer_trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=self.config.max_vocab_size,
            special_tokens=[self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN],
        )
        tokenizer.train_from_iterator(training_texts(), trainer=tokenizer_trainer)

        # Enable padding and truncation for the tokenizer
        pad_id = tokenizer.token_to_id(self.PAD_TOKEN)
        tokenizer.enable_padding(pad_id=pad_id, pad_token=self.PAD_TOKEN)
        tokenizer.enable_truncation(max_length=self.config.max_seq_length)

        logger.debug(f"BPE tokenizer trained: vocab_size={tokenizer.get_vocab_size():,}, pad_id={pad_id}")

        return tokenizer

    def _tokenizer_file(self) -> Path:

        if self.data_limit is not None:
            filename = f"tokenizer_limit_{self.data_limit}.json"
        else:
            filename = f"tokenizer.json"

        return Path(self.config.tokenizer_path) / filename

    def _train_and_save_tokenizer(self, train_data):
        """Train the tokenizer and save it to disk."""
        logger.debug(f"=== _train_and_save_tokenizer() call ===")
        tokenizer_file = self._tokenizer_file()
        if tokenizer_file.exists():
            logger.debug(f"Tokenizer already exists at {tokenizer_file}. Skipping training.")
            return
        
        Path(self.config.tokenizer_path).mkdir(parents=True, exist_ok=True)
        tokenizer = self._train_tokenizer(train_data)
        tokenizer.save(str(tokenizer_file))

    def _load_tokenizer_from_disk(self):
        """
        Load the tokenizer from disk.
        Does not train a new tokenizer in a process except Process 0.
        """
        tokenizer_file = self._tokenizer_file()
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Tokenizer was not prepared: {tokenizer_file}. "
                "Run prepare_data() before setup()."
            )

        # Load the tokenizer from the saved file and assign it to the tokenizer attribute
        # We do assign to attribite in setup processes
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))
        logger.debug(f"=== _load_tokenizer_from_disk() call ===")
        logger.debug(f"Loaded tokenizer from {tokenizer_file}.")
        logger.debug(f"Vocab size: {self.tokenizer.get_vocab_size():,}")

    def _collate_fn(self, batch):

        logger.debug(f"=== _collate_fn() call ===")
        # Debug batch type and keys
        logger.debug(f"Batch type: {type(batch)}")
        logger.debug(f"Batch keys: {list(batch[0].keys())}")
        # Print examples of source and target texts from the batch
        entry = batch[0]
        logger.debug(f"Batch example source_text: {entry['source_text']}")
        logger.debug(f"Batch example target_text: {entry['target_text']}")

        # Extract source texts from the batch
        src_texts = [entry['source_text'] for entry in batch]
        logger.debug(f"Batch source_texts length: {len(src_texts)}") 
        logger.debug(f"Batch example source_text: {src_texts[0]}")

        # Prepare target texts with BOS and EOS tokens
        tgt_texts = [f"{self.BOS_TOKEN} {entry['target_text']} {self.EOS_TOKEN}" for entry in batch]
        logger.debug(f"Batch target_texts length: {len(tgt_texts)}")
        logger.debug(f"Batch example target_text: {tgt_texts[0]}")

        # Tokenize source and target sequences once
        src_encodings = self.tokenizer.encode_batch(src_texts) # [batch_size, seq_length]
        # Print shape and example of source encodings
        logger.debug(f"Source encodings length: {len(src_encodings)}")
        logger.debug(f"Source encoding example encodings: {src_encodings[0]}")
        logger.debug(f"Source encoding example ids: {src_encodings[0].ids}")
        tgt_encodings = self.tokenizer.encode_batch(tgt_texts) # [batch_size, seq_length]
        logger.debug(f"Target encodings length: {len(tgt_encodings)}")
        logger.debug(f"Target encoding example encodings: {tgt_encodings[0]}")
        logger.debug(f"Target encoding example ids: {tgt_encodings[0].ids}")


        # Extract input IDs and attention masks from the encodings and convert them to PyTorch tensors
        src_ids = torch.tensor([enc.ids for enc in src_encodings], dtype=torch.long)
        src_mask = torch.tensor([enc.attention_mask for enc in src_encodings], dtype=torch.long)
        # Extract target IDs and attention masks from the encodings and convert them to PyTorch tensors
        tgt_ids = torch.tensor([enc.ids for enc in tgt_encodings], dtype=torch.long)
        tgt_mask = torch.tensor([enc.attention_mask for enc in tgt_encodings], dtype=torch.long)

        # Return a dictionary as expected by Pytorch Lightning
        return {
            "encoder_input_ids": src_ids,
            "encoder_attention_mask": src_mask,
            "decoder_input_ids": tgt_ids[:, :-1],    # Drops </s> (Starts with <s>) [batch_size, seq_length-1]
            "decoder_attention_mask": tgt_mask[:, :-1], # 
            "decoder_labels": tgt_ids[:, 1:],                 # Drops <s> (Ends with </s>) [batch_size, seq_length-1]
        }
                
    def train_dataloader(self):
        """
        Return a DataLoader for the training dataset.
        Each batch is a dictionary with keys:
        dict_keys(['source_text', 'target_text', 'source_lang', 'target_lang'])

        """
        return torch.utils.data.DataLoader(self.train_data, 
                                           batch_size=self.config.batch_size, 
                                           collate_fn=self._collate_fn,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, 
                                           batch_size=self.config.batch_size, 
                                           collate_fn=self._collate_fn,
                                           num_workers=4,
                                           pin_memory=True
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, 
                                           batch_size=self.config.batch_size, 
                                           collate_fn=self._collate_fn,
                                           num_workers=4,
                                           pin_memory=True)
