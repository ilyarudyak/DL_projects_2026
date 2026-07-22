# Sentiment Analysis Project

## 01 Brief Overview

## 02 IMDB Dataset

The `IMDBData` class in `dataset.py` handles dataset loading, dataset splitting, tokenization setup, and batch processing for sentiment analysis models.

- **Loading dataset**: Loads the IMDB movie reviews dataset via Hugging Face `datasets.load_dataset("stanfordnlp/imdb")` and caches it locally under `datasets/imdb`. 
    - The `imdb_data` object is a **Hugging Face `DatasetDict`**, and its individual splits (like `imdb_data['train']`) are **Hugging Face `Dataset`** objects. It is not a native `torch.utils.data.Dataset` class, but it is **fully compatible** with PyTorch. Hugging Face `Dataset` objects implement the Python "Map-style dataset" protocol (they have `__len__` and `__getitem__`).
    - `IMDBData` class supports dataset size limiting (`data_limit`), which we use to create a toy dataset for quick experimentation. To create it we use `.shuffle()` and `.select()` methods from the Hugging Face `Dataset` API. They return the same type of object (Hugging Face `Dataset`) and are highly efficient.

- **Splitting into train and validation sets**: Uses `train_test_split` (**Hugging Face `Dataset`** method) to split the Hugging Face training split into distinct training (`imdb_train_set`) and validation (`imdb_val_set`) subsets according to specified ratios (e.g., 80% train / 20% val). We use a fixed seed (`seed=42`) for reproducibility.

## Loaders and custom `collate_fn`

- **On-the-fly Encoding**: Instead of encoding the entire dataset upfront (which would consume significant RAM and create huge static tensors), we use a custom `collate_fn`. This performs "lazy" tokenization for each batch as it is requested by the training loop, keeping the memory footprint low.

- **Input/Output** of the `collate_fn`: 
    - **Input**: A list of items (dictionaries) precisely from the Hugging Face `Dataset`, where each item contains raw `"text"` and a numerical `"label"`.
    - **Output**: A dictionary of consolidated PyTorch `torch.long` tensors: `input_ids`, `attention_mask`, and `labels`.

- **The `encode_batch` Method**: We use the `tokenizers` library's `encode_batch` method to process the entire batch of text strings simultaneously. This is highly efficient as it leverages the underlying Rust implementation for parallelized tokenization.

- **The `Encoding` Object & Attention Mask**:
    - `encode_batch` returns a list of `Encoding` objects. While these objects contain various metadata (like `offsets` and `type_ids`), we specifically extract `ids` and `attention_mask`.
    - **Attention Mask**: This is a binary sequence where `1` represents a real token and `0` represents a `<pad>` token. 
    - **Usage**: The mask is passed alongside the `input_ids` to the model. It instructs the architecture (specifically self-attention layers in Transformers or handling variable lengths in RNNs) to ignore the padding tokens during computation. This ensures that the final sentiment prediction is based solely on the actual review content and not the "empty" padding added to make batches uniform in length.

### Pretrained Tokenizers

- **Supported Tokenizers**: Utilizes two primary pretrained tokenizers—GPT-2 (`"gpt2"`) and BERT (`"bert-base-uncased"`)—while also maintaining functionality to train custom BPE / Byte-level BPE (BBPE) tokenizers on the training reviews.
- **Hugging Face `tokenizers` Library**: Built directly on the low-level, Rust-backed Hugging Face `tokenizers` library (`tokenizers.Tokenizer`) rather than `transformers.AutoTokenizer`. Downloads tokenizers on first run and saves them locally as `tokenizer.json` files for fast offline loading.
- **Special symbols and padding**: Enforces padding (`enable_padding`) up to `max_seq_length` (500 tokens) and sequence truncation (`enable_truncation`). For GPT-2 (which lacks a default pad token), dynamically registers custom `<pad>` and `<unk>` tokens; for BERT, aligns with standard `[PAD]` and `[UNK]` token IDs.

- **BERT vs. GPT-2 Tokenizer Advantages for Sentiment Analysis** (the comparison of training results for these tokenizers is described later on):
  - **Subword Algorithm**: BERT uses WordPiece on lowercased text (`bert-base-uncased`), which cleanly isolates subword suffixes/prefixes (e.g., `##ing`), whereas GPT-2 uses Byte-level BPE (BBPE) tailored for continuous byte sequence generation.
  - **Vocabulary Efficiency**: BERT’s smaller vocabulary (~30,522 tokens vs. GPT-2’s ~50,257 tokens) results in a smaller embedding matrix with less parameter bloat, which is better suited for classification on standard text corpora.
  - **Classification Alignment**: BERT natively provides classification-oriented special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`), avoiding potential edge cases caused by manually adding padding tokens to generative models like GPT-2.

## 03 Model Architectures

- 

## 04 Optimizers and Learning Rate Schedulers

## 05 Training and Evaluation

## 06 Further Experiments