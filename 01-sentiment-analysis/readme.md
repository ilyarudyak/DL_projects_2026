# Sentiment Analysis Project

## 01 Brief Overview

- The goal of this project is to implement a sentiment analysis model for movie reviews using the IMDB dataset. The model will classify reviews as either positive or negative based on the text content. The project is partially based on the book "Hands-On Machine Learning with PyTorch" by Aurélien Géron.

- **IMDB Dataset**: We build a custom dataset class `IMDBData` in `dataset.py` to handle loading, splitting, and tokenizing the IMDB movie reviews dataset. We use 2 pre-trained tokenizers (GPT-2 and BERT) using the Hugging Face `tokenizers` library. The dataset is split into training and validation sets, and we implement a custom `collate_fn` for efficient batch processing using the `tokenizers` library's `encode_batch` method. 

- **Model Architecture**: The main architecture is a 2-layer sequence-to-vector unidirectional GRU model. We use padding and truncation to some fixed sequence length (500 tokens in our case). We extract the last real token representation from the GRU output using the attention mask. We use packed sequences.

- **Training and Evaluation**: We use almost the same model hyperparameters as Geron. We first trained our model on MacBook M3 and then on Colab GPU. First results on Colab were encouraging, but when we tried to choose the best tokenizers over 3 runs with different seeds, we noticed high variance in the results. We reduced the batch size from 512 to 128 and got much better and more stable results. Finally, they are at least on par with the book.

```
Geron's results:
Epoch 1/10,                      train loss: 0.6795, train metric: 57.05%, valid metric: 55.84%
Epoch 2/10,                      train loss: 0.6118, train metric: 67.74%, valid metric: 59.58%
Epoch 3/10,                      train loss: 0.4613, train metric: 78.65%, valid metric: 80.68%
Epoch 4/10,                      train loss: 0.3391, train metric: 85.62%, valid metric: 83.52%
Epoch 5/10,                      train loss: 0.2547, train metric: 89.93%, valid metric: 84.66%
Epoch 6/10,                      train loss: 0.2894, train metric: 88.05%, valid metric: 81.98%
Epoch 7/10,                      train loss: 0.1837, train metric: 93.08%, valid metric: 84.26%
Epoch 8/10,                      train loss: 0.1181, train metric: 96.04%, valid metric: 82.80%
Epoch 9/10,                      train loss: 0.0648, train metric: 98.22%, valid metric: 83.44%
Epoch 10/10,                      train loss: 0.0488, train metric: 98.80%, valid metric: 83.30%

Our best results (BERT tokenizer, batch size 128):
Epoch   1 | Train Loss: 0.8025 | Train Acc: 0.5226 | Val Loss: 0.6903 | Val Acc: 0.5696
Epoch   2 | Train Loss: 0.6807 | Train Acc: 0.5761 | Val Loss: 0.6596 | Val Acc: 0.6048
Epoch   3 | Train Loss: 0.6116 | Train Acc: 0.6625 | Val Loss: 0.5986 | Val Acc: 0.6788
Epoch   4 | Train Loss: 0.4747 | Train Acc: 0.7811 | Val Loss: 0.6280 | Val Acc: 0.7550
Epoch   5 | Train Loss: 0.4033 | Train Acc: 0.8385 | Val Loss: 0.4690 | Val Acc: 0.8222
Epoch   6 | Train Loss: 0.2900 | Train Acc: 0.8953 | Val Loss: 0.5182 | Val Acc: 0.8212
Epoch   7 | Train Loss: 0.2314 | Train Acc: 0.9136 | Val Loss: 0.3721 | Val Acc: 0.8620
Epoch   8 | Train Loss: 0.1767 | Train Acc: 0.9384 | Val Loss: 0.4509 | Val Acc: 0.8352
Epoch   9 | Train Loss: 0.1524 | Train Acc: 0.9467 | Val Loss: 0.4727 | Val Acc: 0.8600
Epoch  10 | Train Loss: 0.1099 | Train Acc: 0.9635 | Val Loss: 0.5505 | Val Acc: 0.8506

🛑 Early Stopping triggered at epoch 10

✅ Training finished. Loading best model State

🏆 Best Model Metrics (from Epoch 7):
├─ Train Loss: 0.2314
├─ Train Acc:  0.9136
├─ Val Loss:   0.3721
└─ Val Acc:    0.8620

```

## 02 IMDB Dataset

- The `IMDBData` class in `dataset.py` handles dataset loading, dataset splitting, tokenization setup, and batch processing for sentiment analysis models.

- **Loading dataset**: Loads the IMDB movie reviews dataset via Hugging Face `datasets.load_dataset("stanfordnlp/imdb")` and caches it locally under `datasets/imdb`. 
    - The `imdb_data` object is a **Hugging Face `DatasetDict`**, and its individual splits (like `imdb_data['train']`) are **Hugging Face `Dataset`** objects. It is not a native `torch.utils.data.Dataset` class, but it is **fully compatible** with PyTorch. Hugging Face `Dataset` objects implement the Python "Map-style dataset" protocol (they have `__len__` and `__getitem__`).
    - `IMDBData` class supports dataset size limiting (`data_limit`), which we use to create a toy dataset for quick experimentation. To create it we use `.shuffle()` and `.select()` methods from the Hugging Face `Dataset` API. They return the same type of object (Hugging Face `Dataset`) and are highly efficient.

- **Splitting into train and validation sets**: Uses `train_test_split` (**Hugging Face `Dataset`** method) to split the Hugging Face training split into distinct training (`imdb_train_set`) and validation (`imdb_val_set`) subsets according to specified ratios (e.g., 80% train / 20% val). We use a fixed seed (`seed=42`) for reproducibility.

### Loaders and custom `collate_fn`

- **On-the-fly Encoding**: Instead of encoding the entire dataset upfront (which would consume significant RAM and create huge static tensors), we use a custom `collate_fn`. This performs "lazy" tokenization for each batch as it is requested by the training loop, keeping the memory footprint low.

- **Input/Output** of the `collate_fn`: 
    - **Input**: A list of dictionaries from the Hugging Face `Dataset`, where each item contains review `"text"` and a numerical `"label"`.
    - **Output**: A dictionary of PyTorch `torch.long` tensors: `input_ids`, `attention_mask`, and `labels`.

- **The `encode_batch` Method**: We use the `tokenizers` library's `encode_batch` method to process the entire batch of text strings simultaneously. This is highly efficient as it leverages the underlying Rust implementation for parallelized tokenization.

- **The `Encoding` Object & Attention Mask**:
    - `encode_batch` returns a list of `Encoding` objects. While these objects contain various metadata (like `offsets` and `type_ids`), we specifically extract `ids` and `attention_mask`.
    - **Attention Mask**: A binary tensor where `1` represents a real token and `0` represents a `<pad>` token.
    - **Dynamic Selection of "Summary" States**: In our architectures (like `IMDBModelLPV2` and `IMDBModelLPPackedSeq`), the mask is not just for ignoring padding during training. We use it to calculate the true length of each review (`lengths = attention_mask.sum(dim=1)`). 
    - **Why this matters**: Since movie reviews vary in length, simply taking the hidden state at the final index (e.g., the 500th token) would often return a zeroed vector or "empty" information from padding. By using the attention mask, we can dynamically slice the GRU output at the **last real token** of each specific review, ensuring the classification head receives the actual summarized representation of the text.

### Pretrained Tokenizers

- **Supported Tokenizers**: Utilizes two primary pretrained tokenizers—GPT-2 (`"gpt2"`) and BERT (`"bert-base-uncased"`)—while also maintaining functionality to train custom BPE / Byte-level BPE (BBPE) tokenizers on the training reviews.

- **Hugging Face `tokenizers` Library**: Built directly on the low-level, Rust-backed Hugging Face `tokenizers` library (`tokenizers.Tokenizer`) rather than `transformers.AutoTokenizer`. This is efficient, has minimal overhead, and allows for more direct control over padding and truncation.

- **Special symbols and padding**: Enforces padding (`enable_padding`) up to `max_seq_length` (500 tokens) and sequence truncation (`enable_truncation`). For GPT-2 (which lacks a default pad token), dynamically registers custom `<pad>` and `<unk>` tokens; for BERT, aligns with standard `[PAD]` and `[UNK]` token IDs.

- **BERT vs. GPT-2 Tokenizer Advantages for Sentiment Analysis** (the comparison of training results for these tokenizers is described later on):
  - **Subword Algorithm**: BERT uses WordPiece on lowercased text (`bert-base-uncased`), which cleanly isolates subword suffixes/prefixes (e.g., `##ing`), whereas GPT-2 uses Byte-level BPE (BBPE) tailored for continuous byte sequence generation.
  - **Vocabulary Efficiency**: BERT’s smaller vocabulary (~30,522 tokens vs. GPT-2’s ~50,257 tokens) results in a smaller embedding matrix with less parameter bloat, which is better suited for classification on standard text corpora.
  - **Classification Alignment**: BERT natively provides classification-oriented special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`), avoiding potential edge cases caused by manually adding padding tokens to generative models like GPT-2.

## 03 Model Architectures

- The main architecture is a 2-layer sequence-to-vector unidirectional GRU model. We use padding and truncation to some fixed sequence length (e.g., 500 tokens). We extract the last real token representation from the GRU output using the attention mask. We use packed sequences.

- **Sequence-to-vector model**: The input sequence passes through an embedding layer and a 2-layer **unidirectional GRU**, followed by Layer Normalization and an Output Linear layer that maps the *final* hidden state to class logits.

- **Padding in Embedding layer**: To handle varied review lengths within a batch, we use `padding_idx=self.data.PAD_ID` in the `nn.Embedding` layer. This ensures that the embedding vector for the padding token is initialized to zero and remains zero during training, preventing it from contributing to the gradients. 

- **Extraction of the last REAL token representation**: Since movie reviews are padded to a fixed length (e.g., 500 tokens), the "final" hidden state of a standard RNN would often represent empty padding. 
    - In `IMDBModelLPV2` and `IMDBModelLPPackedSeq`, we use the **attention mask** to calculate the true length of each sequence: `lengths = attention_mask.sum(dim=1)`. 
    - We then dynamically slice the GRU output at the specific index of the last real token (`lengths - 1`) for each review in the batch, ensuring the classifier receives the actual review summary.

- **`num_classes = 2` and `CrossEntropyLoss`**: Although the task is binary (positive/negative), we use `num_classes = 2` and the standard `nn.CrossEntropyLoss` rather than a single output node with `BCEWithLogitsLoss`. This approach is easier to understand and generalize on multi-class classification tasks, for example, if we have 5-star review ratings instead of just positive/negative. 

- **Packed sequence**: In `model.py`, we utilize `torch.nn.utils.rnn.pack_padded_sequence`. By "packing" the embeddings using the calculated lengths, we tell the GRU to **skip computation for padding tokens**. This significantly improves training efficiency and ensures that the hidden state is only updated by real text data and prevents corruption from padding tokens.

## 04 Optimizers and Learning Rate Schedulers

### 01 Optimizer

- We use the **AdamW** optimizer, which is a variant of Adam that decouples weight decay from the gradient update. This is a default for almost all modern deep learning models (especially Transformers, CNNs, and language models). We use it with a standard `weight_decay=0.01`.



### 02 Learning Rate Scheduler

### 03 Batch Size

## 05 Training and Evaluation

### 01 The first training run

- We first trained our model on MacBook M3 with the the same hyperparameters as in Geron's book:  

```yaml
# 2. Model Hyperparameters

# Model hyperparameters: in order of significance
hidden_dim: 64               # Geron: 64
num_layers: 2                # Geron: 2
embedding_dim: 128           # Geron: 128
dropout: 0.4                 # Geron: 0.2 CHANGED! 0.2 -> 0.4
```

- We got the following results:

```
🏆 Best Model Metrics (from Epoch 4):
├─ Train Loss: 0.4422
├─ Train Acc:  0.7954
├─ Val Loss:   0.4601
└─ Val Acc:    0.7874
```

### 02 The second training run

- The second training run was performed on a Colab GPU. Initial results were better than the first run but still not quite like in the book. 

```
🏆 Best Model Metrics (from Epoch 6):
├─ Train Loss: 0.2184
├─ Train Acc:  0.9148
├─ Val Loss:   0.6198
└─ Val Acc:    0.8208
```

### 03 Adding BERT tokenizer

- We identified a possible reason for the lower accuracy compared to the book: we were using GPT-2 tokenizer, which is not optimal for classification tasks, in particular it has a larger vocabulary (~50,257 tokens vs. BERT’s ~30,522 tokens). We added BERT tokenizer and trained the model with it. The results were not better than with GPT-2 tokenizer, but closer to the oners in the book.

```
GPT-2 tokenizer:
🏆 Best Model Metrics (from Epoch 5):
├─ Train Loss: 0.3459
├─ Train Acc:  0.8519
├─ Val Loss:   0.4887
└─ Val Acc:    0.8340

🏆 Best Model Metrics (from Epoch 8):
├─ Train Loss: 0.2511
├─ Train Acc:  0.9040
├─ Val Loss:   0.4985
└─ Val Acc:    0.8204
```

- But then we decided to choose between GPT-2 and BERT tokenizers based on the results of 3 run with fixed seed and the same hyperparameters. The results were kind of discouraging and pointed towards high variance in the results. The results are shown below:

```
GPT-2 tokenizer:

=== Training with seed 42 ===
🏆 Best Model Metrics (from Epoch 5):
├─ Train Loss: 0.3279
├─ Train Acc:  0.8594
├─ Val Loss:   0.5161
└─ Val Acc:    0.7764

=== Training with seed 43 ===
🏆 Best Model Metrics (from Epoch 4):
├─ Train Loss: 0.5347
├─ Train Acc:  0.7308
├─ Val Loss:   0.6351
└─ Val Acc:    0.6772

=== Training with seed 44 ===
🏆 Best Model Metrics (from Epoch 6):
├─ Train Loss: 0.2352
├─ Train Acc:  0.9079
├─ Val Loss:   0.5726
└─ Val Acc:    0.7720

```

### 04 Reducing the batch size

- We reduced the batch size from 512 to 128 and suddently the results were much better and more stable. Finally, they are at least on par with the book. It also shows that BERT is, in fact, maybe a better choice for classification tasks than GPT-2. The results are shown below (best out of 3 runs for each tokenizer):

```
GPT-2 tokenizer:
=== Training with seed 42 ===
🚀 Using hardware accelerator: cuda:0
Epoch   1 | Train Loss: 0.8323 | Train Acc: 0.5099 | Val Loss: 0.7048 | Val Acc: 0.5432
Epoch   2 | Train Loss: 0.6927 | Train Acc: 0.5607 | Val Loss: 0.6728 | Val Acc: 0.5852
Epoch   3 | Train Loss: 0.6217 | Train Acc: 0.6473 | Val Loss: 0.5945 | Val Acc: 0.6806
Epoch   4 | Train Loss: 0.4608 | Train Acc: 0.7931 | Val Loss: 0.5668 | Val Acc: 0.7176
Epoch   5 | Train Loss: 0.3615 | Train Acc: 0.8557 | Val Loss: 0.5069 | Val Acc: 0.8006
Epoch   6 | Train Loss: 0.2466 | Train Acc: 0.9064 | Val Loss: 0.4548 | Val Acc: 0.8322
Epoch   7 | Train Loss: 0.1697 | Train Acc: 0.9406 | Val Loss: 0.5950 | Val Acc: 0.8422
Epoch   8 | Train Loss: 0.1442 | Train Acc: 0.9506 | Val Loss: 0.4895 | Val Acc: 0.8394
Epoch   9 | Train Loss: 0.1004 | Train Acc: 0.9679 | Val Loss: 0.5685 | Val Acc: 0.8300

🛑 Early Stopping triggered at epoch 9

✅ Training finished. Loading best model State

🏆 Best Model Metrics (from Epoch 6):
├─ Train Loss: 0.2466
├─ Train Acc:  0.9064
├─ Val Loss:   0.4548
└─ Val Acc:    0.8322

BERT tokenizer:
=== Training with seed 44 ===
🚀 Using hardware accelerator: cuda:0
Epoch   1 | Train Loss: 0.8025 | Train Acc: 0.5226 | Val Loss: 0.6903 | Val Acc: 0.5696
Epoch   2 | Train Loss: 0.6807 | Train Acc: 0.5761 | Val Loss: 0.6596 | Val Acc: 0.6048
Epoch   3 | Train Loss: 0.6116 | Train Acc: 0.6625 | Val Loss: 0.5986 | Val Acc: 0.6788
Epoch   4 | Train Loss: 0.4747 | Train Acc: 0.7811 | Val Loss: 0.6280 | Val Acc: 0.7550
Epoch   5 | Train Loss: 0.4033 | Train Acc: 0.8385 | Val Loss: 0.4690 | Val Acc: 0.8222
Epoch   6 | Train Loss: 0.2900 | Train Acc: 0.8953 | Val Loss: 0.5182 | Val Acc: 0.8212
Epoch   7 | Train Loss: 0.2314 | Train Acc: 0.9136 | Val Loss: 0.3721 | Val Acc: 0.8620
Epoch   8 | Train Loss: 0.1767 | Train Acc: 0.9384 | Val Loss: 0.4509 | Val Acc: 0.8352
Epoch   9 | Train Loss: 0.1524 | Train Acc: 0.9467 | Val Loss: 0.4727 | Val Acc: 0.8600
Epoch  10 | Train Loss: 0.1099 | Train Acc: 0.9635 | Val Loss: 0.5505 | Val Acc: 0.8506
...
├─ Train Loss: 0.2314
├─ Train Acc:  0.9136
├─ Val Loss:   0.3721
└─ Val Acc:    0.8620

```

## 06 Further Experiments

- **1. Learning Rate Scheduler and Optimizer**: We may try to reproduce Geron's results by using the same learning rate scheduler and optimizer as in the book. 

- **2. Batch Size**: We definitely need to experiment with different batch sizes since they have a drastic influence on training stability and model performance.

- **3. Bidirectional GRU**: we may explore the bidirectional GRU to capture context from both directions in the sequence. But Geron got the same exact results with bidirectional GRU as with unidirectional GRU, so we may not get much of an improvement.

- **4. Sequence Length**: We computed some statistics on the length of the sequences for our 2 tokenizers (GPT-2 and BERT). The results are very close for both of them. `max_seq_length: 500`, which is used in the book, is probably a good choice but we may experiment with different values to see if we can get better results. The statistics are shown below:  

```
GPT-2
  Minimum: 11
  Maximum: 3097
  Mean:    300.3
  Median:  222.0
  Longer than 500: 14.3%

BERT
  Minimum: 13
  Maximum: 3127
  Mean:    314.5
  Median:  233.0
  Longer than 500: 15.7%
```

- **5. Pre-trained Embeddings**: Currently, we initialize our `nn.Embedding` from scratch with a normal distribution in `model.py`. We may load pre-trained **GloVe** or **FastText** vectors into our embedding layer. The size of Embedding matrix is `vocab_size x embedding_dim` (e.g., 30,522 x 128 for BERT tokenizer), so it is pretty big, which can be one of the reasons behind instability of the training. 

- **6. Addressing Model Instability (SWA / Label Smoothing)**: We noticed that seeds 42, 43, and 44 produced wildly different results. We may use **Stochastic Weight Averaging (SWA)**—which is built into PyTorch Lightning—to average weights over the final epochs. Alternatively, we may try **Label Smoothing** in our `CrossEntropyLoss` to prevent the model from becoming over-confident on specific noise in the training set, which should stabilize results across different seeds.
