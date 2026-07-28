import yaml
from dataclasses import dataclass


@dataclass
class TatoebaConfig:

    """
    Configuration class for the Tatoeba dataset and model.
    Parameters are the same as in file "base_config.yaml".
    """

    ###########################################################################
    # 1. Dataset & Paths
    ###########################################################################

    data_path: str = "datasets/tatoeba"
    checkpoint_dir: str = "checkpoints/"
    tokenizer_path: str = "datasets/tokenizers"

    max_seq_length: int = 256          # Geron: 256

    # In Machine Translation we train a tokenizer from scratch
    max_vocab_size: int = 10_000       # Geron: 10_000

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

    batch_size: int = 32              # Geron: 32
    learning_rate: float = 0.001       # Geron: torch.optim.NAdam(lr=0.001)
    epochs: int = 15                   # Geron: 10
    weight_decay: float = 0.01         # Geron: NOT used

    ###########################################################################
    # 4. Training Flow
    ###########################################################################

    patience: int = 3                  #
    gradient_clip_val: float = 1.0
    monitor_metric: str = 'val_loss'
    # Options: "plateau", "cosine", "one_cycle", or "none"
    scheduler_type: str = 'plateau'  

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))