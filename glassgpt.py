"""
================================================================================
 GlassGPT  -  a complete GPT in a single file, explained.
================================================================================

WHAT AN LLM IS, IN ONE SENTENCE
    An LLM predicts the NEXT token given the preceding context (a token = a chunk
    of text: here a byte, or a merge of bytes, often a word fragment). It appends
    the predicted token to the context and predicts again. Repeating this loop
    produces text one token at a time. Every line of code below implements this
    next-token-prediction objective.

THE COMPLETE FLOW WE BUILD
    raw text
       -> tokenization      (BPE: merge the most frequent byte pairs into tokens)
       -> embedding         (each token becomes a vector = a list of numbers)
       -> N Transformer blocks   (self-attention + MLP)
       -> final projection  -> logits (one score per possible next token)
       -> loss              (how wrong we are at predicting the next token)
       -> backprop + optimizer   (the network learns)
       -> generation        (sample tokens one at a time)

    We use PyTorch for only two things: tensors (fast n-dimensional arrays) and
    autograd (automatic gradient computation via .backward()). Every layer --
    attention, MLP, layernorm - is built here by hand, line by line.

--------------------------------------------------------------------------------
 THE DIMENSIONS THAT SHOW UP EVERYWHERE
--------------------------------------------------------------------------------
    Throughout the comments you will see shapes like (B, T, C): the size of a
    tensor, i.e. the length of each of its dimensions.

        B   batch size   how many sequences we process TOGETHER, in parallel
        T   time / block how many context tokens the model attends to (context length)
        C   n_embd       how many numbers describe EACH token (the model width)
        V   vocab_size   how many distinct tokens exist (here 512 BPE tokens)
        H   n_head       how many attention heads run in parallel
        hs  head_size    size of one head = C / H

    So a tensor (32, 64, 128) means: "32 sentences, each 64 tokens long, each
    token described by 128 numbers".

    Note on T: block_size is the MAXIMUM context (here 64). Inside forward the
    same letter T denotes the CURRENT sequence length (<= block_size), which
    grows during generation.

--------------------------------------------------------------------------------
 MASTER DIAGRAM  -  from text to the next token
--------------------------------------------------------------------------------
  TEXT  "..."
    |  BPETokenizer.encode          <- byte-level BPE, OUTSIDE the network (0 params)
    v
  idx  (B, T)   integers
    |
  +--------------------------- EMBEDDINGS ---------------------------+
  |  token_embedding[idx]      (B, T, C)   = WHAT the token is       |
  |  position_embedding[0..T-1](T, C)      = WHERE it is (order)     |
  |  x = token_emb + pos_emb   (B, T, C)   = the RESIDUAL STREAM     |
  +------------------------------------------------------------------+
    |
  ==== n_layer x Block ============================================
  |  x = x + MultiHeadAttention(LayerNorm(x))   <- mix across positions
  |  x = x + FeedForward(LayerNorm(x))          <- per-position transform
  =================================================================
    |
  LayerNorm (ln_f)
  lm_head: Linear C -> V        (weight shared with token_embedding)
    v
  logits (B, T, V)   one score per vocabulary token, at every position
    |
    |-- (training)   F.cross_entropy(logits, targets) -> loss
    |-- (generation) logits[:, -1, :] -> temperature -> top-k -> softmax
    |                -> multinomial -> next token -> append -> repeat
    v
  BPETokenizer.decode  -> generated TEXT

--------------------------------------------------------------------------------
 NOTATION USED IN THE DIAGRAMS
--------------------------------------------------------------------------------
    ->            data flows in this direction (also read as "maps to")
    |   v         vertical flow, top to bottom
    (op)          a pure operation with NO learnable parameters
                  (softmax, ReLU, mask, dropout, weighted sum, sampling)
    [WEIGHTS]     a component with LEARNED parameters (counted in the total)
    (B, T, C)     a tensor shape: batch, current length, channels
    x + f(x)      a residual connection: add to the stream, do not replace it
    A . B         matrix multiply;  A^T is the transpose of A

--------------------------------------------------------------------------------
 READING GUIDE  -  concept -> class/method where to see the detail
--------------------------------------------------------------------------------
    hyperparameters / network shape ......... section 1 (constants at the top)
    BPE merges, encode, decode .............. BPETokenizer.train / .encode / .decode
    sliding window that builds x, y ......... get_batch
    one head (Q/K/V, mask, softmax) ......... Head.forward
    multi-head + concat + projection ........ MultiHeadAttention.forward
    per-token MLP (4x expansion, ReLU) ...... FeedForward.forward
    residual + pre-norm ..................... Block.forward
    embeddings, weight tying, loss .......... GPT.__init__ / GPT.forward
    weight initialization + residual scaling  GPT._init_weights
    sampling (temperature, top-k) ........... GPT.generate
    the 4-step training loop ................ train_model
    train vs val, overfitting ............... estimate_loss
    save / load a trained model ............. save_checkpoint / load_checkpoint
    orchestration of the two paths .......... main

--------------------------------------------------------------------------------
 THIS MODEL'S SPECS
--------------------------------------------------------------------------------
    n_embd (C)      = 128    width: numbers per token
    n_head (H)      = 4      attention heads in parallel
    head_size (hs)  = 32     = C / H
    n_layer         = 4      stacked Transformer blocks
    block_size (T)  = 64     maximum context (number of positions)
    vocab_size (V)  = 512    256 bytes + 256 learned BPE merges (no special tokens)
    MLP hidden      = 512    = 4 * C
    batch (B)       = 32     sequences in parallel (in generation B = 1)
    dropout         = 0.1    active ONLY during training
    tokenizer       = byte-level BPE built in this file; saved in the checkpoint
    activation      = ReLU
    normalization   = LayerNorm, pre-norm
    positions       = absolute, LEARNED (a lookup table)
    weight tying    = token_embedding.weight IS lm_head.weight (lm_head bias is separate)

PARAMETER COUNT (this configuration)
    Only [WEIGHTS] count. Pure operations - softmax, ReLU, mask, dropout,
    weighted average, sampling - have 0 parameters.

        token_embedding    V * C             = 512 * 128        =  65,536
        position_embedding block_size * C    =  64 * 128        =   8,192
        per block          12*C^2 + 10*C     = 12*16384 + 1280  = 197,888
        n_layer blocks     4 * 197,888                          = 791,552
        ln_f               2 * C                                =     256
        lm_head            bias only = V  (matrix is shared)    =     512
        --------------------------------------------------------------------
        TOTAL                                                   = 866,048

    General formula (for this file):
        total = C*(V + block_size)          (embeddings)
              + n_layer*(12*C^2 + 10*C)     (blocks)
              + 2*C                         (ln_f)
              + V                           (lm_head bias, not shared)

--------------------------------------------------------------------------------
 GLOSSARY  -  every technical term in this file, one line each
--------------------------------------------------------------------------------
  DIMENSIONS & TENSORS
    tensor        a multi-dimensional array of numbers; PyTorch's core data type
    shape         a tensor's measurements, e.g. (B, T, C)
    B             batch size: sequences processed together, in parallel
    T             block/context length: how many tokens the model attends to
    C             n_embd: how many numbers describe each token (the width)
    V             vocab_size: how many distinct tokens exist (here 512)
    H             n_head: how many attention heads run in parallel
    hs            head_size = C / H: the size of one head (here 32)
    broadcasting  rule by which a smaller tensor stretches to match a larger one

  TEXT <-> NUMBERS (tokenizer)
    token         a chunk of text (a byte or a merge of bytes) with a numeric id
    tokenize      turn text into a list of ids (and back)
    byte          a number 0..255; any UTF-8 text is a sequence of bytes
    BPE           Byte-Pair Encoding: start from bytes, merge frequent pairs
    merge         fusing two frequent adjacent tokens into one new token
    vocabulary    the set of all tokens: id -> the bytes that id represents
    encode/decode text -> list of ids  /  list of ids -> text

  REPRESENTING TOKENS
    embedding     the vector of C numbers that represents a token or a position
    token embed.  table mapping a token id to its vector (the "WHAT")
    position emb. vector encoding the position 0,1,2... in the context (the "WHERE")
    weight tying  reusing the SAME tensor for the embedding and the final projection

  ATTENTION
    attention     mechanism by which each token aggregates info from other tokens
    self-attention  tokens attend to their own sequence
    Query (Q)     projection of a token used to score other tokens' Keys
    Key (K)       projection of a token matched against other tokens' Queries
    Value (V)     projection of a token aggregated when the token is attended to
    affinity      score (Q . K^T) of how relevant one token is to another
    scaling       dividing by sqrt(hs): bounds the dot-product magnitude so the
                  softmax does not saturate
    causal mask   (tril) prevents a token from attending to FUTURE positions
    softmax       turns a list of scores into probabilities that sum to 1
    multi-head    several heads in parallel, each learning a different relation

  PROCESSING & STRUCTURE
    MLP / feed-forward  small net applied to EACH token alone, to process info
    ReLU          non-linearity that zeroes negatives: lets the net learn non-linear maps
    non-linearity what stops two linear layers from collapsing into one
    residual      the x = x + layer(x) pattern: adds the sublayer output to its
                  input, preserving signal and gradient through depth
    layernorm     normalizes each token (mean 0, uniform scale): stable training
    block         one Transformer block = attention (cross-position mix) + MLP (per-position transform)
    Transformer   the architecture of N blocks stacked on top of each other

  MODEL OUTPUT
    logit         raw (pre-softmax) score of how likely a token is as the next one
    lm_head       final projection C -> V: one logit per vocabulary token

  HOW IT LEARNS (training)
    forward       the pass that runs data through the net and produces the logits
    loss          the error measure (cross-entropy): lower = better predictions
    cross-entropy low when the model gives the right token a high probability
    backward      (backprop) computes the gradient of every weight from the loss
    gradient      how much, and which way, to move each weight to reduce the loss
    optimizer     (AdamW) uses the gradients to update the weights efficiently
    learning rate how big each weight-update step is
    iteration     (step) one full forward -> backward -> update cycle
    train / val   data to learn from  /  unseen data, to measure generalization
    overfitting   memorizing training data instead of generalizing (train loss
                  falls, val loss rises)
    dropout       randomly disables neurons in training: regularization
    parameter     (weight) one of the net's tunable numbers; together, its capacity
    initialization  the starting values of the weights (small, Gaussian)
    seed          fixes randomness: identical, reproducible runs

  HOW IT GENERATES (generation)
    generation    (inference) producing new text, one token at a time
    autoregressive  each generated token feeds back in to predict the next
    sampling      drawing the next token at random according to the probabilities
    temperature   divides the logits before softmax: <1 sharpens the
                  distribution, >1 flattens it
    top-k         keep only the K most probable tokens before sampling

  SAVING
    checkpoint    file with weights + tokenizer + config: reuse without retraining
    state_dict    the dictionary of all the weights the network learned

--------------------------------------------------------------------------------
 HOW TO RUN
--------------------------------------------------------------------------------
    python glassgpt.py

    Drop your own text file named `input.txt` next to this script to train on
    your data: a book, notes, song lyrics. The more (and more varied) the text,
    the better it learns. If `input.txt` is missing, a small built-in sample runs
    so the script runs with no setup.
================================================================================
"""

import math
import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

# Windows: force the console to UTF-8. The BPE works on bytes and generation may
# produce accented or special characters that the console's default encoding
# (cp1252) cannot print -> otherwise the program crashes on print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==============================================================================
# 1) CONFIGURATION (hyperparameters)
# ------------------------------------------------------------------------------
# Every number below is a design choice. Larger values increase model capacity
# but slow down training; the values here are small enough to train in minutes on
# a laptop CPU. Increasing N_EMBD / N_LAYER / MAX_ITERS increases capacity.
#
# How the hyperparameters shape the network:
#                       C = N_EMBD = 128  (width: numbers per token)
#                    <------------------->
#                    +-------------------+   |
#      one token ->  | . . . . . . . . . |   |
#                    +-------------------+   |
#      T = 64 tokens | . . . . . . . . . |   |  depth N_LAYER = 4 blocks
#      of context    | . . . . . . . . . |   |  stacked on top of each other
#      (context)     |        ...        |   |
#                    +-------------------+   |
#                    C splits into H=4 heads -> hs = C/H = 32 each
#                    [__hs__|__hs__|__hs__|__hs__]
# ==============================================================================

# --- How much context the model uses, and how many sequences it processes ---
BLOCK_SIZE = 64  # T: maximum context length. To predict the next token, the
#    model attends to at most the previous 64 tokens. With BPE a token is ~4
#    characters on average, so 64 tokens cover more than 64 characters. Larger =
#    more context, but attention cost grows with T squared.
BATCH_SIZE = 32  # B: how many independent sequences we process each step.
#    Processing 32 together is more efficient than one at a time. Larger = less
#    noisy gradient but more RAM/CPU.

# --- Network dimensions ---
N_EMBD = 128  # C: each token and each position becomes a vector of 128 numbers.
#    The network "width". Must be divisible by N_HEAD.
N_HEAD = 4  # H: number of attention heads. Each head works on 128/4 = 32
#    dimensions and learns a different kind of relation between tokens.
N_LAYER = 4  # how many Transformer blocks are stacked: the network "depth".
DROPOUT = 0.1  # fraction of neurons randomly disabled during training. This is
#    regularization: randomly dropping activations prevents the network from
#    relying on any single neuron, which reduces overfitting.
INIT_STD = 0.02  # standard deviation of the Gaussian weight initialization.
#    Small = bounded outputs at the start. Residual projections use a reduced std,
#    INIT_STD/sqrt(2*N_LAYER) (see GPT._init_weights).

# --- BPE tokenizer ---
BPE_VOCAB_SIZE = 512  # vocabulary size: 256 base bytes + (512-256) = 256 learned
#    merges. Bigger = longer tokens (shorter sequences) but a larger embedding
#    table.

# --- Optimization (how, and how long, it learns) ---
MAX_ITERS = 2000  # number of training steps (forward + backward + update).
EVAL_INTERVAL = 300  # how often we print train and validation loss.
EVAL_ITERS = 100  # how many batches we average the loss over when evaluating
#    (a single measurement would be too noisy).
LEARNING_RATE = 3e-4  # size of each weight-update step. 3e-4 = 0.0003: a typical
#    value for AdamW on Transformers. Too large = the updates overshoot and the
#    loss diverges; too small = the loss decreases very slowly.

# --- Reproducibility and hardware ---
SEED = 1337  # fix randomness: identical runs every time.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # here it will be "cpu".

torch.manual_seed(SEED)  # makes weight init and batches deterministic.

# Sanity check: each head must receive a whole number of dimensions.
assert N_EMBD % N_HEAD == 0, "N_EMBD must be divisible by N_HEAD"
HEAD_SIZE = N_EMBD // N_HEAD  # hs: size of a single head.

# --- Generation: how to sample the produced text (see GPT.generate) ---
PROMPT = ""  # the starting text the model continues from. Empty = start from a
#    minimal context. This model has no separate system/user roles: a prompt is
#    text prepended to the context. With byte-level BPE any character is
#    encodable: there are no out-of-vocabulary tokens.
MAX_NEW_TOKENS = 500  # how many tokens to generate after the prompt (~4 chars
#    each with BPE, so more than 500 characters of text).
TEMPERATURE = 0.8  # scales the logits before softmax: <1 sharpens the
#    distribution, 1 leaves it unchanged, >1 flattens it (see GPT.generate).
TOP_K = 50  # at each step keep only the 50 most probable tokens (None = all).

# --- Saving / loading the trained model ---
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model.pt"
)  # where we save/load the weights + the vocabulary.
FORCE_RETRAIN = False  # True = ignore the checkpoint and always retrain from scratch.


# ==============================================================================
# 2) DATA + BPE TOKENIZER (text <-> numbers)
# ------------------------------------------------------------------------------
# A neural network operates on numbers, not characters, so the first step is to
# map text to integer token ids. This is tokenization.
#
# We use a byte-level BPE (Byte-Pair Encoding) tokenizer: instead of assigning an
# id to every character, we start from bytes and LEARN to merge the most frequent
# pairs into longer tokens ("word pieces"). This gives no out-of-vocabulary tokens
# and shorter sequences, so more context fits in the same BLOCK_SIZE. All the
# details are in the BPETokenizer class.
#
#     "in the middle of the journey"
#            |  encode()               ^   decode()
#            v                         |
#     [ bytes 0..255 ] --learned merges--> [ token ids: 110, 288, 41, ... ]
#            |                                     |
#            |                                     v
#            |                            the network works ONLY on these numbers
#            +-------------------------------------+
#   The tokenizer maps text <-> ids at the input and output boundaries.
# ==============================================================================

# A VARIED text (not a few repeated lines) is essential for BPE. On trivially
# repetitive text the BPE would merge everything into very few tokens, leaving the
# network without data. Here we use the opening of Dante's Inferno (21 distinct
# lines) repeated a modest number of times: enough data, but with real internal
# variety.
FALLBACK_TEXT = (
    "Nel mezzo del cammin di nostra vita\n"
    "mi ritrovai per una selva oscura,\n"
    "ché la diritta via era smarrita.\n"
    "Ahi quanto a dir qual era è cosa dura\n"
    "esta selva selvaggia e aspra e forte\n"
    "che nel pensier rinova la paura!\n"
    "Tant' è amara che poco è più morte;\n"
    "ma per trattar del ben ch'i' vi trovai,\n"
    "dirò de l'altre cose ch'i' v'ho scorte.\n"
    "Io non so ben ridir com' i' v'intrai,\n"
    "tant' era pien di sonno a quel punto\n"
    "che la verace via abbandonai.\n"
    "Ma poi ch'i' fui al piè d'un colle giunto,\n"
    "là dove terminava quella valle\n"
    "che m'avea di paura il cor compunto,\n"
    "guardai in alto e vidi le sue spalle\n"
    "vestite già de' raggi del pianeta\n"
    "che mena dritto altrui per ogne calle.\n"
    "Allor fu la paura un poco queta,\n"
    "che nel lago del cor m'era durata\n"
    "la notte ch'i' passai con tanta pietà.\n"
) * 40  # repeated to have enough data (but the block is varied internally).


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.

    The idea, in two steps
        1) Start from BYTES. Any text, in any language, in UTF-8 is a sequence of
           bytes (0..255). So the base vocabulary has 256 tokens and there are no
           "unknown" characters: an emoji or a rare letter is simply more bytes.
           (With char-level, a character never seen during training broke the
           encoding.)
        2) Learn to MERGE the most frequent pairs. We look at which two tokens are
           adjacent most often (e.g. 'i'+'n') and fuse them into a new token
           ('in'). Repeat: now maybe 'in'+'g' -> 'ing'. Continue until the
           vocabulary reaches the desired size (BPE_VOCAB_SIZE).

    How a byte pair becomes a token (one BPE merge)
        text:   c a m m i n   c a m m i n a
        bytes: [99][97][109][109][105][110] ...   (0..255, the base vocab)

        step 1) count adjacent pairs:
                (109,109)="mm" x2   (97,109)="am" x2   (99,97)="ca" x2 ...
        step 2) merge the most frequent -> "mm" becomes the new id [256]
                c a [256] i n   c a [256] i n a
        step 3) repeat: now "ca" -> [257], then "[257]mm" -> [258] ...
                [258] i n   [258] i n a         (tokens grow longer)

        After BPE_VOCAB_SIZE-256 merges the tokens are recurring "word pieces":
        shorter sequences + more context for the same BLOCK_SIZE.

    Why it improves on char-level
        Tokens become recurring word pieces. The same sentence becomes a SHORTER
        sequence of tokens, so more context fits in the same
        BLOCK_SIZE and learns relations between useful pieces, not single letters.

    Note: this is a minimal implementation. Production tokenizers first split the
    text into words with a regex, to prevent merges from spanning whitespace; this
    version omits that step and can therefore merge tokens across spaces.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        # Return the tokenizer to its "blank" state (just the 256 bytes, no
        # merges). Used by the constructor and at the start of train(), so the
        # same instance can be re-trained without carrying old merges along.
        #
        # merges: (id_a, id_b) -> new_id. The learned merges, in the order we
        # learned them (that order is also the PRIORITY in which they apply).
        self.merges = {}
        # vocab: id -> the byte sequence that id represents. The first 256 are the
        # bytes themselves; from 256 on they are the merges.
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def _get_stats(ids):
        """Count the frequency of every pair of CONSECUTIVE tokens in `ids`.

        Returns a dict {(id_a, id_b): how_many}. This is the step that tells us
        which pair is worth merging first: the most frequent one. `zip(ids,
        ids[1:])` walks all adjacent pairs (i, i+1).

            ids:   [a, b, c, a, b]
                    |__|              (a,b)
                       |__|           (b,c)
                          |__|        (c,a)
                             |__|     (a,b)  -> (a,b) counts 2, the rest 1
        """
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(ids, pair, new_id):
        """Return a new list where every occurrence of `pair` (two adjacent
        tokens) is replaced by the single token `new_id`.

        Example: _merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99].

            before: 1  2  3  1  2
                    |__|     |__|      find (1,2)
            after: 99     3    99      two tokens become one

        Advance by 2 when the pair is found, by 1 otherwise; the `i < len(ids) - 1`
        guard avoids reading past the end of the list.
        """
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text, target_vocab_size, verbose=True):
        """Learn the merges: start from bytes and fuse the most frequent pair,
        repeated (target_vocab_size - 256) times.

        The BPE training loop
            ids = bytes of the text
            repeat (target_vocab_size - 256) times:
                _get_stats(ids)        -> count all pairs
                argmax frequency       -> pick the winning pair
                _merge(ids, pair, id)  -> compress ids, record the merge
            Each pass adds 1 token to the vocab and shortens ids.

        Size guarantee: the vocabulary always ends up with exactly
        target_vocab_size tokens, even if the text was too short to complete every
        merge (see the padding at the end). This keeps vocab_size deterministic and
        equal to what the docs assume (V = 512), so the parameter count always
        holds.
        """
        assert target_vocab_size >= 256, "the BPE vocabulary starts at 256 (the bytes)"
        num_merges = target_vocab_size - 256
        ids = list(text.encode("utf-8"))  # text -> list of bytes (0..255)
        self._reset()  # always restart from a "blank" vocabulary (256 bytes)
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break  # text too short: nothing left to merge
            pair = max(stats, key=stats.get)  # the most frequent pair
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 50 == 0:
                print(f"  [bpe] merge {i + 1}/{num_merges}")

        # PAD UP TO target_vocab_size: if the loop stopped early (short text: no
        # more pairs to merge), fill the missing slots with empty "placeholder"
        # tokens (b""). They are INERT: no merge produces them, so encode never
        # emits them, and in decode they contribute the empty string. Their only
        # purpose is to guarantee vocab_size == target_vocab_size in EVERY case, so
        # the network shape (embedding table V x C) is always the expected one.
        n_missing = target_vocab_size - len(self.vocab)
        if n_missing > 0 and verbose:
            print(
                f"  [bpe] short text: only {len(self.merges)} merges possible; "
                f"filling {n_missing} placeholder slots to reach {target_vocab_size}."
            )
        for new_id in range(len(self.vocab), target_vocab_size):
            self.vocab[new_id] = b""

    def encode(self, text):
        """Text -> list of token ids, applying the learned merges in priority
        order (oldest first, exactly as during training).

            text -> bytes -> [apply merges in learned order] -> token ids
                             (oldest merges first, i.e. the lowest ids)

        Two limitations of this implementation:
        1) COST: we recompute _get_stats over the whole sequence at EVERY merge,
           so the cost is ~O(n * number_of_merges). On a large input.txt this is
           slow; a production implementation uses incremental data structures.
        2) MERGING ACROSS WORDS: we do not split the text into words first, so BPE
           may merge tokens ACROSS spaces (e.g. "of_the" as one token). A regex that
           separates words/punctuation first would prevent this; it is omitted
           here.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # among the pairs present, pick the highest-priority one (lowest merge
            # id = learned earliest). inf = a pair that cannot be merged.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no remaining pair can be merged
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        # Token ids -> text: concatenate the bytes of each id and decode UTF-8.
        # errors="replace": if generation produces an invalid byte sequence (can
        # happen mid multi-byte character), emit a placeholder character instead
        # of raising.
        # Note: any padding "placeholder" tokens (vocab = b"", see train) add the
        # empty string, so they are transparent here too.
        blob = b"".join(self.vocab[i] for i in ids)
        return blob.decode("utf-8", errors="replace")


# Variables prepared by prepare_data() (training path) or by load_checkpoint()
# (loading path). They stay None until needed.
tokenizer = None  # BPETokenizer instance
encode = None  # function: text -> list of ids
decode = None  # function: list of ids -> text
VOCAB_SIZE = None  # vocabulary size (number of tokens)
train_data = None  # tensor of ids for training
val_data = None  # tensor of ids for validation


def prepare_data():
    """Load the text, TRAIN the BPE tokenizer, and prepare the train/val tensors.

    Sets the global variables used by the model and by get_batch. Called only when
    we train (on the loading path the tokenizer comes from the checkpoint).

        input.txt (or FALLBACK) -> train BPE -> encode -> tensor of ids
                                                            |
                                              90% train ---+--- 10% validation
    """
    global tokenizer, encode, decode, VOCAB_SIZE, train_data, val_data

    # 1) read the text (from your input.txt, or the built-in sample)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[data] Loaded input.txt: {len(text):,} characters")
    else:
        text = FALLBACK_TEXT
        print("[data] input.txt not found: using the built-in sample text.")
        print("       (drop your own 'input.txt' in this folder and rerun)")

    # 2) train the BPE up to BPE_VOCAB_SIZE tokens
    print(f"[bpe] Training the BPE tokenizer (target {BPE_VOCAB_SIZE} tokens)...")
    tokenizer = BPETokenizer()
    tokenizer.train(text, BPE_VOCAB_SIZE)
    encode = tokenizer.encode
    decode = tokenizer.decode
    VOCAB_SIZE = tokenizer.vocab_size
    print(f"[bpe] Final vocabulary: {VOCAB_SIZE} tokens")

    # 3) encode all the text into ids, then split 90% train / 10% validation.
    # We evaluate on held-out validation data: measuring on training data would
    # overstate performance.
    ids = torch.tensor(encode(text), dtype=torch.long)
    print(
        f"[data] Encoded text: {len(ids):,} tokens "
        f"(compression {len(text) / max(len(ids), 1):.2f}x vs characters)"
    )
    n_split = int(0.9 * len(ids))
    train_data = ids[:n_split]
    val_data = ids[n_split:]

    # Guard: after BPE we need more than BLOCK_SIZE tokens for both train and val,
    # otherwise get_batch cannot cut out sequences. If this fires, the text is too
    # short or too compressed.
    assert len(train_data) > BLOCK_SIZE and len(val_data) > BLOCK_SIZE, (
        f"Text too small after tokenization ({len(ids)} tokens). Use a longer "
        f"input.txt, or lower BPE_VOCAB_SIZE or BLOCK_SIZE."
    )


def get_batch(split):
    """Draw a random batch of sequences (x) and their targets (y).

    The language-modeling trick: y is x "shifted by 1". The target at every
    position is simply the NEXT token. So the network learns, for each position,
    to predict what comes next:

        x = token ids  [10, 22, 5, 8]
        y = next ones  [22, 5, 8, 41]   (y[i] = x[i+1])

    How x and y are built (sliding window over the text)
        data:  ... 10  22   5   8  41  17 ...      (a long stream of ids)
                   [----- x, T tokens ----]
         x  =       10  22   5   8              context in
         y  =           22   5   8  41          target = x shifted by +1
                   ^    ^    ^    ^
                   |    |    |    +-- after [10,22,5,8]  the target is 41
                   |    |    +------- after [10,22,5]    the target is 8
                   |    +------------ after [10,22]      the target is 5
                   +----------------- after [10]         the target is 22

        Repeated B=32 times from B random start points -> x, y shape (B, T).
        In one pass the network makes T next-token predictions per sequence.

    Returns two tensors of shape (B, T).
    """
    d = train_data if split == "train" else val_data
    # Pick B random start positions. -BLOCK_SIZE so we do not run off the end.
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))  # shape (B,)
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])  # shape (B, T)
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])  # shape (B, T)
    return x.to(DEVICE), y.to(DEVICE)


# ==============================================================================
# 3) THE TRANSFORMER BUILDING BLOCKS
# ==============================================================================


class Head(nn.Module):
    """A single head of causal self-attention.

    Purpose
        To predict the next token, some positions in the context are more relevant
        than others. In "...the runnin_", the ending "g" depends on "runnin", not
        on "the". The head learns, per position, which earlier positions to weight.

    Query, Key, Value
        Each token vector is linearly projected into three vectors of size hs:
            Query (Q) - scored against other tokens' Keys
            Key   (K) - scored against other tokens' Queries
            Value (V) - aggregated in proportion to the resulting weights
        Token i's output is a weighted sum of the Values of tokens j, with weights
        from softmax(Q_i . K_j / sqrt(hs)).

    Self & causal
        Self   - the Q, K, V all come from the same sequence.
        Causal - a token attends only to positions <= its own; future positions
                 are masked, since they hold the tokens still to be predicted.

    One head, step by step (hs = C / H = 32)
        x (B,T,C)
          |--(x . Wq)--> Q (B,T,hs)   query
          |--(x . Wk)--> K (B,T,hs)   key
          |--(x . Wv)--> V (B,T,hs)   value

        scores = Q . K^T / sqrt(hs)      -> (B,T,T)   pairwise affinity
        causal mask (tril==0 -> -inf)    -> (B,T,T)   mask future positions
        softmax over the last dim        -> (B,T,T)   each row sums to 1
        out    = scores . V              -> (B,T,hs)  attention-weighted values

    The causal mask, visually
                who is attended to ->
                    t0    t1    t2    t3
          t0 [ ok  |-inf |-inf |-inf ]
          t1 [ ok  | ok  |-inf |-inf ]   -inf -> 0 after softmax
          t2 [ ok  | ok  | ok  |-inf ]
          t3 [ ok  | ok  | ok  | ok  ]

    The 1/sqrt(hs) scaling stops the dot products from growing with hs, which
    would saturate the softmax and vanish the gradients.
    """

    def __init__(self):
        super().__init__()
        # Q, K, V projections (weight matrices) mapping each token vector (C) into
        # the three views (hs each). bias=False: these projections use no bias
        # term. They are three independent linear projections of the same input.
        self.key = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)  # C -> hs
        self.query = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)  # C -> hs
        self.value = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)  # C -> hs
        self.dropout = nn.Dropout(DROPOUT)
        # The causal mask (tril) is NOT defined here. It is identical for all heads,
        # so we keep it once in MultiHeadAttention and pass it to forward, avoiding
        # duplicating the same buffer H*N_LAYER times.

    def forward(self, x, tril):
        # x:    (B, T, C). B sequences, T tokens, C numbers per token.
        # tril: (T, T) lower-triangular of ones = the causal mask, already sliced
        #       to the current length T by MultiHeadAttention.
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs) - the KEY projection
        q = self.query(x)  # (B, T, hs) - the QUERY projection

        # Attention scores: dot product between every Query and every Key.
        # (B, T, hs) @ (B, hs, T) = (B, T, T): for each sequence a T x T matrix of
        # affinities between token i and token j. Divided by sqrt(hs) (the
        # scaling): without it, large hs makes the dot products large, the softmax
        # saturates (outputs near 0 or 1), and the gradients vanish.
        wei = q @ k.transpose(-2, -1) * (HEAD_SIZE**-0.5)  # (B, T, T)

        # Causal mask: where tril==0 (future tokens) put -infinity, so after the
        # softmax those weights become 0. Token i never attends to j>i.
        wei = wei.masked_fill(tril == 0, float("-inf"))  # (B, T, T)

        # Softmax over the last dimension: turn scores into weights that sum to 1
        # on each row. wei[b, i, :] is token i's attention distribution over tokens
        # j <= i (the masked positions contribute 0).
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Aggregation: Value vectors averaged with the attention weights. Each
        # token's output is the weighted sum of the Values it attended to.
        v = self.value(x)  # (B, T, hs) - the VALUE projection
        out = wei @ v  # (B,T,T) @ (B,T,hs) = (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads in PARALLEL.

    Each head has its own Q, K, V projections and can learn a different relation
    between positions. The heads run in parallel; their outputs are concatenated
    along the channel dimension and passed through a final linear projection.

        x (B,T,128) --+--> Head 1 --> (B,T,32) --+
                      +--> Head 2 --> (B,T,32) --+  concat   proj (128->128)
                      +--> Head 3 --> (B,T,32) --+--> (B,T,128) --> (B,T,128)
                      +--> Head 4 --> (B,T,32) --+
                            |                        |            |
                      independent Q/K/V      4x32 = 128       mix across
                      per head               (back to C)      the heads

    Note: 4 heads of 32 cost the same as 1 head of 128 (same total C), split into
    H independent subspaces.
    """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])
        # Final projection applied after concatenating the heads. H*hs = C, so the
        # concatenation has dimension C and this Linear maps C -> C.
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        # This Linear adds to the residual stream (see Block): we mark it so that
        # _init_weights gives it the SCALED init (reduced std). See GPT._init_weights.
        self.proj.IS_RESIDUAL_PROJ = True
        self.dropout = nn.Dropout(DROPOUT)
        # THE single copy of the causal mask, shared by all heads in this layer.
        # "tril" = lower-triangular of ones. register_buffer: part of the model
        # (moves with .to(device), saved in the checkpoint) but NOT a trainable
        # parameter.
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        # Slice the mask to the current length T (in generation T grows up to
        # BLOCK_SIZE) and pass it to each head: one mask for all.
        T = x.shape[1]
        tril = self.tril[:T, :T]  # (T, T)
        # Run the H heads and concatenate along the last dim:
        # H times (B, T, hs) -> (B, T, H*hs) = (B, T, C)
        out = torch.cat([h(x, tril) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))  # (B, T, C)
        return out


class FeedForward(nn.Module):
    """A small per-token MLP: expand, filter, re-compress.

    Attention mixes information across positions; this MLP transforms each
    position independently, applying the same weights to every token.

        (B,T,128) --Linear--> (B,T,512) --ReLU--> (B,T,512) --Linear--> (B,T,128)
                     |            |          |                   |
                  expand 4x    hidden 4C  drop negatives     re-compress to C
                               (wider dim) (non-linearity)

    The 4x expansion (C -> 4C -> C) widens the hidden layer, increasing the
    layer's capacity. ReLU (zero out negatives) introduces the non-linearity:
    without it two Linears in a row would collapse into a single linear map.

    This layer applies to each position independently (attention already mixed
    across positions). In large models it holds the majority of the parameters.
    """

    def __init__(self):
        super().__init__()
        # The second Linear (4C -> C) is the projection that adds to the residual
        # stream: we define it separately to mark it and give it the scaled init.
        # The two Linears (C -> 4C and 4C -> C) with a ReLU between them form the
        # MLP.
        proj = nn.Linear(4 * N_EMBD, N_EMBD)  # 4C -> C  (re-compression)
        proj.IS_RESIDUAL_PROJ = True
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),  # C -> 4C  (expansion)
            nn.ReLU(),  # non-linearity
            proj,
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)  # (B, T, C) -> (B, T, C)


class Block(nn.Module):
    """One Transformer block = cross-position mixing (attention) + per-position transform (MLP).

    We stack N_LAYER of them. Deep stacks rely on two mechanisms present in every
    Transformer:

    1) RESIDUAL (x = x + sublayer(x)): the sublayer output is ADDED to its input
       rather than replacing it. This gives an identity path along which the
       activations and the gradient flow undamped through depth. Without it, deep
       networks are hard to optimize.

    2) LAYERNORM (pre-norm: normalize BEFORE the sublayer): normalizes each token's
       activations to zero mean and unit variance, then applies a learned scale and
       shift. This bounds the activation magnitude and stabilizes training.

        x -----------------------------(+)------------------------------(+)----> x
        |                               ^                                ^
        |  the input is added           |  identity path again           |
        |  unchanged                    |                                |
        +--> LayerNorm --> Multi-Head --+   +--> LayerNorm --> FeedFwd --+
                           (mix positions)     (per-position)

            x = x + MultiHeadAttention(LayerNorm(x))    # mix across positions
            x = x + FeedForward(LayerNorm(x))           # per-position transform

    Without the residual sum, at several layers the gradient vanishes and the
    network fails to train. Large models stack dozens of blocks.
    """

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()  # self-attention: mix across positions
        self.ffwd = FeedForward()  # MLP: per-position transform
        self.ln1 = nn.LayerNorm(N_EMBD)  # normalize before attention
        self.ln2 = nn.LayerNorm(N_EMBD)  # normalize before the MLP

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # residual around attention
        x = x + self.ffwd(self.ln2(x))  # residual around the MLP
        return x


# ==============================================================================
# 4) THE COMPLETE GPT MODEL
# ==============================================================================


class GPT(nn.Module):
    """Assembles the full model:
      - token embedding    : WHAT this token is (its content)
      - position embedding : WHERE it is in the context (its order)
      - N Transformer blocks : the stacked attention + MLP layers
      - layernorm + final linear head -> logits over the whole vocabulary

    Why embeddings? A token id is an integer that carries no information about the
    token. The embedding maps each id to a learned vector of C real numbers. The
    vectors start random; training adjusts them so that tokens used in similar
    contexts get similar vectors.

    Why also positions? Self-attention is permutation-invariant: without position
    information "abc" and "cba" would produce the same representation. The position
    embedding adds, to each token, a learned vector encoding its index 0, 1, 2...
    The two embeddings are summed.

        idx (B,T)  integers
           |  token_embedding[idx]        pos_embedding[0..T-1]
           |     (B,T,C) = WHAT      +       (T,C) = WHERE
           v
        x (B,T,C)  = meaning + position
           |  Block 1 -> Block 2 -> Block 3 -> Block 4     (N_LAYER times)
           v
        x (B,T,C)
           |  final LayerNorm ; lm_head: Linear C -> V
           v
        logits (B,T,V)   one score per vocab token, at EVERY position
           |  (training) compare with targets -> loss
           |  (use)      take the last position -> sample the next token
    """

    def __init__(self):
        super().__init__()
        # Table V x C: one row (vector of C numbers) per vocab token. A lookup
        # table: given a token id, returns its vector.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)  # (V, C)
        # Table T x C: one vector per POSITION 0..T-1.
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)  # (T, C)
        # The stacked Transformer blocks.
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)  # final layernorm
        # "Language model head": projects from C to the vocabulary -> one score
        # (logit) for each of the V possible tokens as the "next token".
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)  # C -> V
        # This lm_head keeps its bias, which is a separate parameter. The weight
        # tying below shares only the MATRIX, not the bias. (Parameter count +
        # formula: see the header.)

        # WEIGHT TYING: the embedding table (V, C) and the final head matrix (V, C)
        # have the SAME shape and represent the same token<->vector relation. We tie
        # them: the same tensor is reused for both. Benefits: ~V*C fewer parameters
        # (~66k here) and often better generalization. After this line, updating one
        # updates the other.
        #
        #   token_embedding.weight (V,C)  <== SAME TENSOR ==>  lm_head.weight (V,C)
        #        "id -> vector"                              "vector -> id score"
        #
        # ORDER MATTERS: initialize ALL weights first, then tie. This way the
        # shared tensor holds a deterministic value (the embedding's) independent
        # of the order self.apply() visits modules. Tying first would let apply()
        # initialize the same tensor twice (as Embedding and as Linear): harmless
        # today (same std) but a silent bug the moment the two stds diverge.
        self.apply(self._init_weights)  # 1) initialize everything
        self.lm_head.weight = self.token_embedding.weight  # 2) then tie

    def _init_weights(self, module):
        """Initialize the weights. Called by self.apply() on EVERY submodule of the
        network.

        - Weights ~ Normal(0, 0.02): the small standard deviation of 0.02 keeps the
          initial activations bounded. Biases start at 0.
        - SCALED residual init: the projections that write to the residual stream
          (marked IS_RESIDUAL_PROJ in MultiHeadAttention and FeedForward) use a
          smaller std, 0.02/sqrt(2*N_LAYER). Reason: at each layer we add TWO
          contributions (attention + MLP) to the residual stream; without damping,
          the signal variance grows layer after layer and destabilizes training.
          The 1/sqrt(2*N_LAYER) factor exactly compensates that cumulative sum.

          residual stream:  x --(+)--(+)--(+)--(+)--(+)--(+)--(+)--(+)--> ...
                                 ^    ^    ^    ^    ^    ^    ^    ^
                                 each adds variance: the reduced std keeps it in check
        """
        if isinstance(module, nn.Linear):
            std = INIT_STD
            if getattr(module, "IS_RESIDUAL_PROJ", False):
                std = INIT_STD / math.sqrt(2 * N_LAYER)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)

    def forward(self, idx, targets=None):
        # idx: (B, T) integers = the input tokens.
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)  # (B, T, C)
        # device taken from idx, NOT from the global DEVICE: so the model runs on the same
        # device as its inputs and does not depend on an external constant.
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, C)
        # Sum content + position. pos_emb (T,C) broadcasts over all B sequences:
        # every sequence uses the same positions 0..T-1.
        x = tok_emb + pos_emb  # (B, T, C)

        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            # Generation mode: no targets, no loss to compute.
            return logits, None

        # LOSS = cross-entropy between logits and targets. The logits are the
        # model's raw scores: a higher logit means a higher predicted probability
        # for that token. Cross-entropy is LOW when the model assigned high
        # probability to the RIGHT token (the one in targets), HIGH when it was
        # wrong; it is the quantity minimized during training. F.cross_entropy
        # expects logits (N, V) and targets (N,), so we flatten the B and T
        # dimensions together: N = B*T.
        #
        #    logits[i] = [ 0.1, 3.2, -1.0, ... ] -> softmax -> probabilities
        #    target[i] =         ^ (the right token is id 1)
        #    loss is high if the prob on the right token is low; low if it is high.
        Bc, Tc, Vc = logits.shape
        logits = logits.view(Bc * Tc, Vc)  # (B*T, V)
        targets = targets.view(Bc * Tc)  # (B*T,)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()  # generation needs no gradients -> faster/lighter
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text token by token. The loop is autoregressive: each sampled
        token is appended to the context and fed back in. idx: (B, T) initial
        context.

        Two parameters control the sampling distribution:

        - temperature: the logits are divided by this value before softmax:
            < 1  -> sharper distribution: concentrates probability on the most
                    likely tokens. More repetitive output.
            = 1  -> no effect (the distribution is unchanged).
            > 1  -> flatter distribution: more varied output, and more low-
                    probability tokens sampled.
        - top_k: before sampling, keep only the K most probable tokens and set the
          rest to probability 0. This removes the low-probability tail. None = no
          filter.

        The autoregressive loop
              context idx (B, t)
                 |  crop to the last BLOCK_SIZE tokens
                 v
              forward -> logits, take ONLY the last position (B, V)
                 |  / temperature       (scale the logits)
                 |  top-k                (keep the K most probable, rest -> -inf)
                 |  softmax              (-> probabilities)
                 |  multinomial          (sample 1 token: NOT always the max)
                 v
              idx_next (B,1)  --->  append: idx = [idx, idx_next]  (B, t+1)
                 |                                     |
                 +-------------------------------------+
                 repeat max_new_tokens times -> the text grows one token at a time
        """
        temperature = max(temperature, 1e-6)  # avoid division by zero
        for _ in range(max_new_tokens):
            # The model attends to at most BLOCK_SIZE tokens: crop the context.
            idx_cond = idx[:, -BLOCK_SIZE:]  # (B, <=T)
            logits, _ = self(idx_cond)  # (B, T, V)
            # We only care about the LAST position (the next-token prediction), and
            # we divide it by the temperature (see docstring).
            logits = logits[:, -1, :] / temperature  # (B, V)
            # top-k: keep only the K highest logits; the rest -> -infinity, so
            # after the softmax they get probability 0.
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[
                    :, [-1]
                ]  # (B, 1): the K-th value
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probs = F.softmax(logits, dim=-1)  # (B, V) probabilities
            # Sample the next token from the probability distribution instead of
            # taking the argmax: this introduces variability in the output.
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ==============================================================================
# 5) TRAINING
# ------------------------------------------------------------------------------
# The network is trained by three steps repeated MAX_ITERS times:
#   1) LOSS     : the prediction error (see forward, cross-entropy).
#   2) BACKWARD : loss.backward() computes, for EACH of the ~866,000 weights
#                 (this configuration, weight tying included), the gradient of the
#                 loss with respect to that weight (its magnitude and sign).
#   3) UPDATE   : optimizer.step() moves each weight along the negative gradient,
#                 reducing the loss. Repeating this decreases the error over time.
#
#      get_batch("train")          fetch x, y (B,T)
#            |
#            v
#      FORWARD:  model(x, y) --------> loss   (the prediction error)
#            |                          |
#            |                          v
#            |                    zero_grad()   (clear the old gradients!)
#            |                          |
#            |                          v
#            |                    BACKWARD: loss.backward()  d(loss)/d(weight)
#            |                          |
#            |                          v
#            |                    UPDATE: optimizer.step()  update the weights
#            |                          |
#            +--------------------------+   repeat MAX_ITERS times
#
#   Expected starting point: loss ~ ln(V) = ln(512) ~ 6.24 (uniform prediction
#   over V tokens). If the loss starts near there and decreases, the
#   initialization is correct and the network is learning.
# ==============================================================================


@torch.no_grad()
def estimate_loss(model):
    """Measure the average loss on train and validation.

    We average over EVAL_ITERS batches because a single measurement would be too
    noisy. model.eval()/train() switch dropout off/on (we want it off when
    evaluating).

    Why measure TWO losses:
      - train loss = error on data seen during training;
      - val loss   = error on data NEVER seen.

        train v
              |\\
              | \\____ they fall TOGETHER  -> really learning (generalizes)
        val   |  ----
              +------------------> iterations

        train v
              |\\
              | \\___                     train falls...
        val   |   __/                    ...but val RISES -> OVERFITTING
              +------------------> iterations   (memorizing, not generalizing)

    If both fall together the network generalizes. If train falls but val rises,
    the network is OVERFITTING: memorizing the training data instead of
    generalizing. (With the built-in sample text - 21 distinct lines repeated 40
    times - the network can largely memorize the block: val loss drops a lot but
    not to zero. With genuinely large, varied text, val loss settles higher, which
    indicates generalization rather than memorization.)
    """
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(model):
    """Run the training loop on the model. Modifies the weights in-place."""
    n_params = sum(p.numel() for p in model.parameters())
    # Total number of trainable parameters: the model's capacity. Large models use
    # the same architecture with far more parameters.
    print(f"[model] Trainable parameters: {n_params:,}")

    # AdamW: a variant of Adam, a common optimizer for Transformers. Adam adapts
    # the step size for EACH parameter using running averages of the gradient and
    # its squared magnitude.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("[train] Starting training...")
    # The loss starts around ln(VOCAB_SIZE). With a BPE vocabulary of 512 tokens
    # that is ln(512) = 6.24 (a char-level vocabulary of 28 characters would give
    # ln(28) = 3.33). This is the reference value: at the start the weights are
    # random and the model predicts a uniform distribution over the VOCAB_SIZE
    # tokens, and the cross-entropy of a uniform prediction over N options is
    # exactly ln(N). A loss starting near this value indicates a correct
    # initialization.
    for it in range(MAX_ITERS):
        # Periodic evaluation (does not update the weights, only measures).
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model)
            print(
                f"  step {it:5d} | train loss {losses['train']:.4f} "
                f"| val loss {losses['val']:.4f}"
            )

        # --- The 4 training steps ---
        xb, yb = get_batch("train")  # 1) fetch a batch
        _, loss = model(xb, yb)  # 2) FORWARD: compute the loss
        optimizer.zero_grad(set_to_none=True)  # clear the old gradients.
        #    PyTorch ACCUMULATES gradients; not
        #    clearing them would add the previous
        #    step's gradients to this step's.
        loss.backward()  # 3) BACKWARD: d(loss)/d(each weight)
        optimizer.step()  # 4) UPDATE: update the weights


def save_checkpoint(model, path):
    """Save to disk everything needed to reuse the model without retraining:
      - the weights (state_dict): the values learned during training;
      - the BPE tokenizer (merges + vocab): required to map ids <-> text. The
        learned merges are part of the model;
      - the architecture hyperparameters: needed to rebuild the same network shape.

        model.pt = [ weights ] + [ merges + vocab ] + [ config ]
                    the values    text <-> ids map    the shape
        All three are required; without any one the saved model is unusable.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": {"merges": tokenizer.merges, "vocab": tokenizer.vocab},
            "config": {
                "block_size": BLOCK_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "vocab_size": VOCAB_SIZE,
            },
        },
        path,
    )
    print(f"[model] Checkpoint saved to {path}")


def load_checkpoint(path):
    """Reload a saved model. Returns the ready-to-use model, or None if loading
    fails (e.g. you changed the hyperparameters after saving: the network "shape"
    no longer matches the saved weights -> better to retrain).

    IMPORTANT: we also restore the EXACT BPE tokenizer the model was trained with
    (the learned merges), so generation works even if the original text is gone.
    If loading fails, we put everything back as it was, so a later retrain stays
    consistent.
    """
    global tokenizer, VOCAB_SIZE, encode, decode
    global BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, HEAD_SIZE
    backup = (
        tokenizer, VOCAB_SIZE, encode, decode,
        BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, HEAD_SIZE,
    )
    try:
        # weights_only=False because we also save Python objects (merges, vocab, config).
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        # restore the saved model's BPE tokenizer
        tokenizer = BPETokenizer()
        tokenizer.merges = ckpt["tokenizer"]["merges"]
        tokenizer.vocab = ckpt["tokenizer"]["vocab"]
        encode = tokenizer.encode
        decode = tokenizer.decode
        VOCAB_SIZE = tokenizer.vocab_size
        # Apply the saved ARCHITECTURE to the globals: the checkpoint is
        # self-describing, so it loads correctly even if you changed the
        # hyperparameters at the top of the file in the meantime (otherwise the
        # network shape would not match the saved weights).
        cfg = ckpt["config"]
        # ANTI-SURPRISE WARNING: if you changed the hyperparameters in the file but
        # a model.pt already exists, the (self-describing) checkpoint WINS and your
        # changes are ignored. We flag it explicitly before overwriting the globals
        # with the saved values.
        current = {
            "block_size": BLOCK_SIZE, "n_embd": N_EMBD,
            "n_head": N_HEAD, "n_layer": N_LAYER,
        }
        diff = {k: (current[k], cfg[k]) for k in current if current[k] != cfg[k]}
        if diff:
            print("[model] WARNING: the hyperparameters in the file do NOT match")
            print("        the checkpoint. Using the SAVED ones (the checkpoint")
            print("        wins). To apply your changes set FORCE_RETRAIN = True")
            print("        and rerun (retrains from scratch).")
            for k, (in_file, in_ckpt) in diff.items():
                print(f"            - {k}: in file={in_file}  in checkpoint={in_ckpt}")

        BLOCK_SIZE = cfg["block_size"]
        N_EMBD = cfg["n_embd"]
        N_HEAD = cfg["n_head"]
        N_LAYER = cfg["n_layer"]
        HEAD_SIZE = N_EMBD // N_HEAD
        # rebuild the network (using the just-restored globals) and load the weights
        model = GPT().to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[model] Checkpoint loaded from {path} (no training needed).")
        return model
    except Exception as e:
        # restore the original state, so a possible retrain stays consistent.
        (
            tokenizer, VOCAB_SIZE, encode, decode,
            BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, HEAD_SIZE,
        ) = backup
        print(
            f"[model] Could not load the checkpoint ({e}). Retraining from scratch."
        )
        return None


def generate_text(model):
    """Generate text starting from PROMPT, using the TEMPERATURE and TOP_K
    sampling parameters."""
    model.eval()
    # With byte-level BPE any character is encodable: nothing to discard. If PROMPT
    # is empty we start from a newline, as a minimal context.
    prompt_ids = encode(PROMPT) if PROMPT else encode("\n")

    context = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)  # (1, len)

    print("\n[gen] Generated text (prompt + continuation):\n")
    out = model.generate(
        context,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )[0].tolist()
    print(decode(out))


def main():
    """Entry point. Orchestrates the two possible paths:

      - LOADING: if a model.pt exists and FORCE_RETRAIN is False, reload weights
        and tokenizer from the checkpoint and skip training.
      - TRAINING: otherwise prepare the data (training the BPE), build the network,
        train, and save the checkpoint.

    In both cases, at the end it generates text with PROMPT and the sampling
    parameters (TEMPERATURE, TOP_K).

        model.pt exists and not FORCE_RETRAIN ?
            yes -> load_checkpoint --------------------+
            no  -> prepare_data -> GPT() -> train ->    |
                    save_checkpoint ----------------->  +--> generate_text
    """
    # 1) If a checkpoint exists and we are not forcing a retrain, load it and skip
    #    training. Otherwise prepare the data (training the BPE tokenizer), build
    #    the network, train, and save.
    model = None
    if os.path.exists(CHECKPOINT_PATH) and not FORCE_RETRAIN:
        model = load_checkpoint(CHECKPOINT_PATH)
    if model is None:
        prepare_data()  # train the BPE and prepare train/val
        model = GPT().to(DEVICE)
        train_model(model)
        save_checkpoint(model, CHECKPOINT_PATH)

    # 2) Generation with prompt + sampling parameters (temperature, top-k).
    generate_text(model)


if __name__ == "__main__":
    main()
