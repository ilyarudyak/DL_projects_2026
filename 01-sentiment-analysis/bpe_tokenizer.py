from collections import Counter
import re
from typing import Dict, List, Tuple

import logging
logger = logging.getLogger("imdb.bpe_tokenizer")


class SimpleBPE:

    def __init__(self, vocab_size: int):

        # Initialize the BPE tokenizer with a target vocabulary size.
        self.vocab_size = vocab_size
        self.vocab: List[str] = []

        # token to index mapping for the learned vocabulary
        self.token_to_index: Dict[str, int] = {}

        # index to token mapping for the learned vocabulary
        self.index_to_token: Dict[int, str] = {}

        # Maps a pair of tokens (tuple) to their merged string form
        self.merges: Dict[Tuple[str, str], str] = {}

    def _get_stats(self, corpus_words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Counts the frequency of all adjacent token pairs in the corpus."""
        pairs = Counter()
        for word in corpus_words:
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += 1
        return pairs

    def _merge_vocab(
        self, corpus_words: List[List[str]], pair: Tuple[str, str]
    ) -> List[List[str]]:
        """Replaces all occurrences of the target pair with the merged token."""
        logger.debug(f"===MERGING VOCAB===")
        new_corpus = []
        p0, p1 = pair
        logger.debug(f"Merging pair: {pair}")
        merged_token = p0 + p1
        logger.debug(f"New merged token: {merged_token}")

        for word in corpus_words:
            new_word = []
            logger.debug(f"---Processing word: {word}---")
            i = 0
            while i < len(word):

                # If we find the pair, merge it
                # i < len(word) - 1 is necessary to avoid index out of range error when checking word[i + 1]
                # By definition we need 2 consequtive characters to form a pair, so
                # we check word[i] and word[i + 1]
                if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                    logger.debug(f"Found pair {pair} in word {word} at index {i}. Merging into '{merged_token}'.")

                    # We append the merged token to the new word,
                    # not the individual characters of the pair
                    new_word.append(merged_token)
                    logger.debug(f"New word after merging: {new_word}")

                    # Skip the next character since it's part of the merged pair
                    i += 2

                else:
                    # There is no match for the pair
                    # OR
                    # We reached the last character. There is no next character 
                    # to check for a pair, 
                    # so we just append the last character.
                    new_word.append(word[i])
                    logger.debug(f"New word after appending character: {new_word}")
                    i += 1
            logger.debug(f"Final new word after processing: {new_word}")
            new_corpus.append(new_word)
        return new_corpus
    
    def _get_initial_corpus(self, text: str) -> List[List[str]]:
        """Prepares the initial corpus by splitting text into words and characters."""
        raw_words = re.findall(r"\w+", text.lower())
        logger.debug(f"Initial raw words from text: {raw_words}")
        corpus_words = [list(word) + ["</w>"] for word in raw_words]
        logger.debug(f"Initial corpus words split into characters: {corpus_words}")
        return corpus_words

    def fit(self, text: str):
        """Learns the BPE merges from a training corpus."""

        logger.debug(f"Fitting BPE tokenizer to text with target vocab size {self.vocab_size}...")

        # # Clean text slightly and split into words, appending an end-of-word token
        # raw_words = re.findall(r"\w+", text.lower())
        # logger.debug(f"Found {raw_words} words in the corpus.")

        # # Split each word into individual characters
        # corpus_words = [list(word) + ["</w>"] for word in raw_words]
        # logger.debug(f"Corpus words split into characters: {corpus_words}")

        # Use the helper function to get the initial corpus
        corpus_words = self._get_initial_corpus(text)

        # Initialize base vocabulary with unique individual characters
        unique_chars = set(char for word in corpus_words for char in word)
        logger.debug(f"Unique characters in the corpus: {unique_chars}")    
        self.vocab = list(unique_chars)
        logger.debug(f"Initial vocabulary (characters): {self.vocab}")

        # Determine how many merges we need to perform to reach the target vocabulary size
        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            logger.warning(
                f"Target vocab size {self.vocab_size} is smaller than or equal to base char count {len(self.vocab)}."
            )
            return

        logger.debug(f"Base vocabulary size: {len(self.vocab)}")
        logger.debug(f"Iterating to learn {num_merges} merges...")

        # Iteratively find and merge the most frequent pairs
        for i in range(num_merges):

            debugging_range = [0, 1, 2]  # Only log the first few iterations for brevity
            logger.debug(f"\n===Iteration {i + 1}/{num_merges}===")

            # Count the frequency of all adjacent token pairs in the current corpus
            pairs = self._get_stats(corpus_words)
            if not pairs:
                break
            if i in debugging_range: logger.debug(f"Pairs frequency: {pairs}")

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if i in debugging_range: logger.debug(f"Most frequent pair: {best_pair} with count {pairs[best_pair]}")

            # Record the merge rule
            merged_token = best_pair[0] + best_pair[1]
            if i in debugging_range: logger.debug(f"Merging pair {best_pair} into new token '{merged_token}'")

            # Update the merges dictionary and vocabulary
            self.merges[best_pair] = merged_token
            self.vocab.append(merged_token)

            # Apply the merge to the corpus
            corpus_words = self._merge_vocab(corpus_words, best_pair)
            if i in debugging_range: logger.debug(f"Corpus after merge: {corpus_words}")

        # Build token mappings after learning all merges
        self._build_token_mappings()

    def _build_token_mappings(self):
        """Builds token to index and index to token mappings for the learned vocabulary."""
        self.token_to_index = {token: idx for idx, token in enumerate(self.vocab)}
        self.index_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """Encodes new text into a list of token indices using the learned merges."""
        tokens = self.tokenize(text)
        return [self.token_to_index[token] for token in tokens if token in self.token_to_index]
    
    def decode(self, token_indices: List[int]) -> str:
        """Decodes a list of token indices back into a string."""
        tokens = [self.index_to_token[idx] for idx in token_indices if idx in self.index_to_token]
        # Join tokens and remove the end-of-word markers
        return "".join(token.replace("</w>", "") for token in tokens)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes new text using the learned merge rules."""
        raw_words = re.findall(r"\w+", text.lower())
        corpus_words = [list(word) + ["</w>"] for word in raw_words]

        # Apply all learned merges in the exact order they were learned
        for pair, merged_token in self.merges.items():
            corpus_words = self._merge_vocab(corpus_words, pair)

        # Flatten the list of tokens
        return [token for word in corpus_words for token in word]
    

class SimpleBPEV2:

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, vocab_size: int):

        # Initialize the BPE tokenizer with a target vocabulary size.
        self.vocab_size = vocab_size
        self.vocab: List[str] = [self.PAD_TOKEN, self.UNK_TOKEN]  # Start with special tokens in the vocabulary

        # token to index mapping for the learned vocabulary
        self.token_to_index: Dict[str, int] = {self.PAD_TOKEN: self.PAD_ID, self.UNK_TOKEN: self.UNK_ID}

        # index to token mapping for the learned vocabulary
        self.index_to_token: Dict[int, str] = {self.PAD_ID: self.PAD_TOKEN, self.UNK_ID: self.UNK_TOKEN}

        # Maps a pair of tokens (tuple) to their merged string form
        self.merges: Dict[Tuple[str, str], str] = {}

    def _get_stats(self, corpus_words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Counts the frequency of all adjacent token pairs in the corpus."""
        pairs = Counter()
        for word in corpus_words:
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += 1
        return pairs

    def _merge_vocab(
        self, corpus_words: List[List[str]], pair: Tuple[str, str]
    ) -> List[List[str]]:
        """Replaces all occurrences of the target pair with the merged token."""
        logger.debug(f"===MERGING VOCAB===")
        new_corpus = []
        p0, p1 = pair
        logger.debug(f"Merging pair: {pair}")
        merged_token = p0 + p1
        logger.debug(f"New merged token: {merged_token}")

        for word in corpus_words:
            new_word = []
            logger.debug(f"---Processing word: {word}---")
            i = 0
            while i < len(word):

                # If we find the pair, merge it
                # i < len(word) - 1 is necessary to avoid index out of range error when checking word[i + 1]
                # By definition we need 2 consequtive characters to form a pair, so
                # we check word[i] and word[i + 1]
                if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                    logger.debug(f"Found pair {pair} in word {word} at index {i}. Merging into '{merged_token}'.")

                    # We append the merged token to the new word,
                    # not the individual characters of the pair
                    new_word.append(merged_token)
                    logger.debug(f"New word after merging: {new_word}")

                    # Skip the next character since it's part of the merged pair
                    i += 2

                else:
                    # There is no match for the pair
                    # OR
                    # We reached the last character. There is no next character 
                    # to check for a pair, 
                    # so we just append the last character.
                    new_word.append(word[i])
                    logger.debug(f"New word after appending character: {new_word}")
                    i += 1
            logger.debug(f"Final new word after processing: {new_word}")
            new_corpus.append(new_word)
        return new_corpus
    
    def _get_initial_corpus(self, text: str) -> List[List[str]]:
        """Prepares the initial corpus by splitting text into words and characters."""
        # raw_words = re.findall(r"\w+", text.lower())
        # In version 2, we want to keep punctuation as separate tokens, 
        # so we use a regex that captures both words and punctuation
        raw_words = re.findall(r"\w+|[^\w\s]", text.lower())
        logger.debug(f"Initial raw words from text: {raw_words}")

        # In version 2 DO not add the end-of-word token "</w>" to the characters. 
        # corpus_words = [list(word) + ["</w>"] for word in raw_words]
        corpus_words = [list(word) for word in raw_words]
        logger.debug(f"Initial corpus words split into characters: {corpus_words}")


        return corpus_words

    def fit(self, text: str):
        """Learns the BPE merges from a training corpus."""

        logger.debug(f"Fitting BPE tokenizer to text with target vocab size {self.vocab_size}...")

        # Use the helper function to get the initial corpus
        corpus_words = self._get_initial_corpus(text)

        # Initialize base vocabulary with unique individual characters
        unique_chars = set(char for word in corpus_words for char in word)
        logger.debug(f"Unique characters in the corpus: {unique_chars}")    
        # self.vocab = list(unique_chars)

        # We have already initialized the vocabulary with special tokens, 
        # so we need to modify the way we create the vocabulary
        for char in sorted(unique_chars):
            if char not in self.vocab:
                self.vocab.append(char)
        logger.debug(f"Initial vocabulary (characters): {self.vocab}")

        # Determine how many merges we need to perform to reach the target vocabulary size
        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            logger.warning(
                f"Target vocab size {self.vocab_size} is smaller than or equal to base char count {len(self.vocab)}."
            )
            return

        logger.debug(f"Base vocabulary size: {len(self.vocab)}")
        logger.debug(f"Iterating to learn {num_merges} merges...")

        # Iteratively find and merge the most frequent pairs
        for i in range(num_merges):

            debugging_range = [0, 1, 2]  # Only log the first few iterations for brevity
            logger.debug(f"\n===Iteration {i + 1}/{num_merges}===")

            # Count the frequency of all adjacent token pairs in the current corpus
            pairs = self._get_stats(corpus_words)
            if not pairs:
                break
            if i in debugging_range: logger.debug(f"Pairs frequency: {pairs}")

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if i in debugging_range: logger.debug(f"Most frequent pair: {best_pair} with count {pairs[best_pair]}")

            # Record the merge rule
            merged_token = best_pair[0] + best_pair[1]
            if i in debugging_range: logger.debug(f"Merging pair {best_pair} into new token '{merged_token}'")

            # Update the merges dictionary and vocabulary
            self.merges[best_pair] = merged_token
            self.vocab.append(merged_token)

            # Apply the merge to the corpus
            corpus_words = self._merge_vocab(corpus_words, best_pair)
            if i in debugging_range: logger.debug(f"Corpus after merge: {corpus_words}")

        # Build token mappings after learning all merges
        self._build_token_mappings()

    def _build_token_mappings(self):
        """Builds token to index and index to token mappings for the learned vocabulary."""
        self.token_to_index = {token: idx for idx, token in enumerate(self.vocab)}
        self.index_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """Encodes new text into a list of token indices using the learned merges."""
        tokens = self.tokenize(text)
        # return [self.token_to_index[token] for token in tokens if token in self.token_to_index]
        # In version 2, we return the UNK_ID for tokens not in the vocabulary
        return [self.token_to_index.get(token, self.UNK_ID) for token in tokens]
    
    def decode(self, token_indices: List[int]) -> str:
        """Decodes a list of token indices back into a string."""
        tokens = [self.index_to_token[idx] for idx in token_indices if idx in self.index_to_token]

        # Join tokens and remove the end-of-word markers
        # return "".join(token.replace("</w>", "") for token in tokens)
    
        # In version 2 we join tokens and add space between them to form the final string
        return " ".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes new text using the learned merge rules."""
        raw_words = re.findall(r"\w+", text.lower())
        corpus_words = [list(word) for word in raw_words]

        # Apply all learned merges in the exact order they were learned
        for pair, merged_token in self.merges.items():
            corpus_words = self._merge_vocab(corpus_words, pair)

        # Flatten the list of tokens
        return [token for word in corpus_words for token in word]


# --- Quick Test ---
if __name__ == "__main__":
    # Sample toy corpus with repeating subword patterns
    sample_text = (
        "low lower newest widest"
    )

    # We want a final vocabulary size of 20
    bpe = SimpleBPE(vocab_size=20)
    bpe.fit(sample_text)

    logger.debug("\nLearned Merge Rules:")
    for pair, merged in bpe.merges.items():
        logger.debug(f"{pair} -> '{merged}'")

    # Test tokenization on unseen words
    test_phrase = "lowest newer"
    tokens = bpe.tokenize(test_phrase)
    logger.debug(f"\nTokenizing '{test_phrase}':\n{tokens}")