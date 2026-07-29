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

    PyTorch is used for only two things: tensors (fast n-dimensional arrays) and
    autograd (automatic gradient computation via .backward()). Every layer -
    attention, MLP, layernorm - is built here by hand, line by line. The master
    diagram below is the whole of it, end to end.

--------------------------------------------------------------------------------
 THE DIMENSIONS THAT SHOW UP EVERYWHERE
--------------------------------------------------------------------------------
    Shapes like (B, T, C) appear throughout the comments. A shape lists the
    length of each of a tensor's dimensions.

        B   batch size   how many sequences are processed TOGETHER, in parallel
        T   time / block how many context tokens the model attends to
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
    |                -> loss.backward() -> AdamW step  (the network learns)
    |-- (generation) logits[:, -1, :] -> temperature -> top-k -> softmax
    |                -> multinomial -> next token -> append -> repeat
    |                one full pass through the whole diagram PER TOKEN
    |                (real serving splits this in two: see INFERENCE below)
    v
  BPETokenizer.decode  -> generated TEXT

--------------------------------------------------------------------------------
 READING GUIDE  -  concept -> class/method where to see the detail
--------------------------------------------------------------------------------
    hyperparameters / network shape ......... section 1 (constants at the top)
    BPE merges, encode, decode .............. BPETokenizer.train / .encode / .decode
    loading text, train/val split ........... prepare_data
    sliding window that builds x, y ......... get_batch
    Query/Key/Value, in plain terms ......... Head (docstring)
    one head (Q/K/V, mask, softmax) ......... Head.forward
    why the Value map is split in two ....... MultiHeadAttention (docstring)
    multi-head + concat + projection ........ MultiHeadAttention.forward
    the MLP as a store of facts ............. FeedForward (docstring)
    per-token MLP (4x expansion, ReLU) ...... FeedForward.forward
    residual + pre-norm ..................... Block.forward
    embeddings, weight tying, loss .......... GPT.__init__ / GPT.forward
    weight init + residual scaling .......... GPT._init_weights
    sampling (temperature, top-k) ........... GPT.generate
    the 4-step training loop ................ train_model
    train vs val, overfitting ............... estimate_loss
    save / load a trained model ............. save_checkpoint / load_checkpoint
    prompt -> generated text ................ generate_text
    orchestration of the two paths .......... main

    Any unfamiliar term is defined in the GLOSSARY at the end of this header.

    Explained in this header, with no code of their own:
    every term in the file, one line each ... "GLOSSARY"
    per-component weight table, vs GPT-3 .... "WHERE THE 12*C^2 COMES FROM"
    what a vector's numbers mean ............ "WHAT THE NUMBERS IN A VECTOR MEAN"
    softmax, and what temperature does ...... "SOFTMAX, IN BOTH PLACES IT APPEARS"
    why depth needs non-linearity ........... "WHERE THE NON-LINEARITY ENTERS"
    prefill vs decode, KV cache, TTFT ....... "INFERENCE"
    RMSNorm, RoPE, SwiGLU, GQA .............. "MODERN MODELS"

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
    weight tying    = token_embedding.weight IS lm_head.weight (bias separate)

PARAMETER COUNT (this configuration)
    Only learned weights count. Pure operations - softmax, ReLU, mask, dropout,
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
 WHERE THE 12*C^2 COMES FROM  -  the same table used for GPT-3
--------------------------------------------------------------------------------
    "12*C^2 per block" above is a compressed total. Broken out per component it
    becomes the standard weight table - the SAME rows used to account for
    GPT-3's 175 billion parameters. Only the numbers differ: the architecture
    behind both columns is the one in this file.

    Symbols: C = width, V = vocab, H = heads, hs = head size, L = layers,
             n_hidden = MLP hidden width = 4*C. Note H*hs = C in both columns.

                             GlassGPT              GPT-3
        C  (d_embed)              128             12,288
        V  (n_vocab)              512             50,257
        hs (d_query/d_value)       32                128
        H  (n_heads)                4                 96
        L  (n_layers)               4                 96
        n_hidden (n_neurons)      512             49,152

      component        formula (matrices)  GlassGPT            GPT-3
      --------------------------------------------------------------
      Embedding        C * V                 65,536      617,558,016
      Key              hs * C * H * L        65,536   14,495,514,624
      Query            hs * C * H * L        65,536   14,495,514,624
      Value            hs * C * H * L        65,536   14,495,514,624
      Output           C * hs * H * L        65,536   14,495,514,624
      Up-projection    n_hidden * C * L     262,144   57,982,058,496
      Down-projection  C * n_hidden * L     262,144   57,982,058,496
      Unembedding      V * C               0 (tied)      617,558,016
      --------------------------------------------------------------
      matrices only                         851,968  175,181,291,520

    Divide the four attention rows by L and by C^2: since hs*H = C, each is
    exactly C^2 per layer. The two MLP rows are 4*C^2 each per layer.

        K + Q + V + Output   =  4 * C^2   per layer
        Up + Down            =  8 * C^2   per layer
        ------------------------------------------
                                12 * C^2   <- the header number, explained

    The 10*C remainder is everything the table above leaves out - biases and
    LayerNorms, which are negligible at scale but real here:

        Output bias C + Up bias 4C + Down bias C + ln1 2C + ln2 2C = 10*C

    GlassGPT total = 851,968 (matrices) + 8,192 (position embedding) + 5,888
    (all biases + all LayerNorms, i.e. L*10*C + 2*C + V) = 866,048.

    Both columns leave out the same small terms: biases, normalization
    parameters, and the position embedding table. GPT-3 has one too - learned
    and absolute, like this file - at 2048*12288 = 25.2M, which rounds away
    against 175B. The same table is 1% of GlassGPT.

    Reading down the GPT-3 column:
      - The MLP holds ~66% of the weights (115.9B of 175.2B). Most of a
        Transformer's capacity sits in its MLPs, while attention does the
        mixing across positions.
      - Embedding and unembedding together are ~0.7% of the 175B figure, which
        counts them as two separate matrices. This file ties them - hence the 0
        in the Unembedding row (see GPT.__init__) - and even shared they are 7.6%
        of its total; untied the pair would be 14%. The vocabulary dominates a
        small model's budget and rounds away in a large one, so weight tying
        earns its keep here and barely registers at scale.

--------------------------------------------------------------------------------
 WHAT THE NUMBERS IN A VECTOR MEAN
--------------------------------------------------------------------------------
    Every token becomes C numbers. Nothing in the code says what those numbers
    are for - they start random and training decides. But the arrangement
    training settles on explains why the rest of the architecture is shaped the
    way it is.

    DIRECTIONS CARRY MEANING
        Training pushes the embeddings into an arrangement where DIRECTIONS in
        the space correspond to properties. The classic demonstration is that
        the difference between two related embeddings, added to a third, lands
        near a fourth:

            embedding("king") - embedding("man") + embedding("woman")
                                                    ~=  embedding("queen")

        Read it as: the step from "man" to "woman" is the same step as the one
        from "king" to "queen", so one direction in the space has come to encode
        something like grammatical gender. The same trick finds plurals -
        embedding("cats") - embedding("cat") is a "more than one" direction, and
        its dot product with other nouns comes out reliably higher for the
        plural ones.

        Real embeddings only approximate this - "queen" is used for more than
        "female king", so it does not land exactly where the arithmetic points.
        The useful part is the principle: meaning lives in directions, and a
        space with many dimensions has room for many independent ones.

    THE DOT PRODUCT MEASURES ALIGNMENT
        This is the single operation the whole architecture is built from. The
        dot product of two vectors is

            positive   when they point in a similar direction
            zero       when they are perpendicular (unrelated)
            negative   when they point in opposite directions

        Every matrix multiply in this file is a pile of dot products, and in
        each case the question being asked is "how well do these two align?":

            Q . K^T       does this token's Key answer that token's Query?
            up-projection does this vector point along the direction stored in
                          this row of the weight matrix?
            lm_head       does this vector align with the embedding of token v?
                          (that dot product IS the logit for token v)

    THE VECTOR ACCUMULATES CONTEXT
        The token embedding is a lookup table: it has no idea what surrounds
        it. The vector for "mole" is identical in "American shrew mole", "one
        mole of carbon dioxide" and "a biopsy of the mole" - one entry in a
        table, one vector, three unrelated meanings.

        Everything downstream exists to fix that. Each block ADDS to the vector
        it receives (the residual stream, see Block), so meaning accumulates: a
        vector that entered as the generic word "tower" can be nudged toward
        "Eiffel Tower" by a preceding "Eiffel", and then away from "tall thing"
        again by a preceding "miniature". By the last layer it encodes far more
        than the word it started as.

        This matters most at the final position, because that is the only one
        generation reads. Given a whole detective novel ending in "therefore
        the murderer was", the vector that started life as the plain word "was"
        has to have absorbed everything relevant in the context, or the
        prediction cannot be right.

--------------------------------------------------------------------------------
 SOFTMAX, IN BOTH PLACES IT APPEARS
--------------------------------------------------------------------------------
    softmax turns any list of numbers into a probability distribution: every
    entry between 0 and 1, all of them summing to 1. It exponentiates each
    entry and divides by the total:

        softmax(z)_i = e^(z_i) / sum_j e^(z_j)

    Exponentiating makes every value positive and magnifies the gaps between
    them, so the largest input takes most of the mass - but not all of it,
    which is what the "soft" refers to. A hard max would put 1 on the winner
    and 0 everywhere else.

    The same arithmetic appears twice in this file, for two different purposes:

      Head.forward    over each ROW of the T x T score matrix. Turns "how
                      relevant is token j to token i" into weights summing to
                      1, which makes the aggregation that follows a weighted
                      AVERAGE - it cannot inflate the residual stream.
      GPT.generate    over the V logits at the last position. Turns raw scores
                      into the probability of each token coming next.

    TEMPERATURE divides the logits before exponentiating: softmax(z / tau).
    (The letter T is already taken here by the context length, so tau is used
    for the temperature.) Larger tau flattens the distribution and gives
    unlikely tokens more of a chance; smaller tau sharpens it; tau approaching
    0 puts all the mass on the single largest logit, which is just argmax. Only
    the second softmax takes a temperature: it is a knob on the output, while
    the attention softmax is part of the computation itself.

--------------------------------------------------------------------------------
 WHERE THE NON-LINEARITY ENTERS
--------------------------------------------------------------------------------
    A stack of matrix multiplies is still one matrix multiply: W1.(W2.(W3.x))
    collapses into (W1.W2.W3).x. Without something non-linear between them, all
    n_layer blocks would algebraically fold into a single linear map, and depth
    would buy nothing. Five places break linearity in this file:

      1. ReLU in the MLP (FeedForward)  - the main source. In modern models
         this is SwiGLU or GeLU instead, but the role is identical.
      2. softmax in the attention (Head.forward) - normalizes each row.
      3. Q.K^T is QUADRATIC in the input: x appears twice (once through Wq, once
         through Wk), so the scores are already non-linear before the softmax.
      4. LayerNorm - divides by the vector's own standard deviation.
      5. softmax + multinomial sampling at the output (GPT.generate).

    Point 3 is the one usually missed. Attention is often described as "linear
    given the weights", and that is true: once A = softmax(...) is fixed, A.V is
    a linear operation. But A itself is computed FROM x, so every input builds
    its own mixing matrix on the way through.

--------------------------------------------------------------------------------
 INFERENCE  -  the same forward pass, two very different shapes
--------------------------------------------------------------------------------
    Training runs one forward per batch. Generation runs one forward PER TOKEN,
    and those forwards come in two shapes. The first reads the whole prompt at
    once; every later one reads a single token, the one just produced. Serving
    systems name the two phases PREFILL and DECODE: same code, opposite
    bottlenecks. This file implements the simple version ("WHAT THIS FILE DOES
    INSTEAD", below); the split explains almost everything about how real LLM
    serving behaves.

    Take a 1000-token prompt and 200 generated tokens. That is the SAME compute
    graph, executed 201 times:

      PREFILL   1 forward, 1000 tokens at once, one single time
                |###################################|          -> TTFT
                (time to first token: the wait before any text appears)

      DECODE    200 forwards, 1 token each, strictly one after another
                |#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#|#| ...      -> TPOT
                token n is the input of token n+1, so they CANNOT be parallelized
                (time per output token: the speed at which text scrolls)

    PREFILL - matrix x matrix (GEMM), compute-bound
        X          [1000 x C]      all positions enter together
        A          [1000 x 1000]   O(n^2), half of it masked away
        KV cache   1000 rows written; from here on they are never recomputed
        logits     only the LAST row is needed

        The other 999 rows of logits are thrown away: that forward existed to
        FILL THE CACHE, not to predict. The GPU runs near peak FLOPs, so adding
        more requests to the batch buys nothing - it is already saturated.

    DECODE - vector x matrix (GEMV), memory-bound
        x          [1 x C]         a single token
        A          [1 x 1000]      O(n), one row only
        KV cache   read in full, one row appended; it grows with every token
        logits     all of it used, for exactly 1 token

        The arithmetic is tiny, but producing that one token still requires
        reading EVERY weight of the model out of memory (140 GB for a large
        model). The GPU sits mostly idle, waiting on memory. Batching pays
        enormously here: the same weight read serves 32 requests at once.

    SPECULATIVE DECODING - spending the idle compute
        Decode wastes almost all of the hardware's arithmetic capacity: the
        weights must be read in full to produce a single token, and while they
        are being read the compute units have nothing to do. That asymmetry is
        the whole idea: verifying 4 tokens costs nearly the same as generating
        1, because either way the weights are loaded exactly once.

        So a second, much smaller model - the DRAFT - guesses ahead, and the
        real model - the TARGET - checks the guesses in a single pass:

            draft   (small, fast)      proposes 4 tokens:
                                       " and" / " then" / " he" / " ran"
                                          |
            target  (the real model)   ONE forward over all 4 at once
                                       (prefill shape: GEMM, not GEMV)
                                          |
                                       compares each proposed token with what
                                       it would have produced itself
                                          |
            accepted: " and then he" <----+  first three match, the fourth
                                             does not: it is discarded and
                                             replaced by the target's own token
                                          |
            result: 4 tokens emitted for the cost of roughly 1 target forward

        The critical property is that this is NOT an approximation. With the
        right acceptance rule (a rejection-sampling step that corrects for the
        difference between the two distributions), the text produced is drawn
        from EXACTLY the distribution the target model would have produced on
        its own. All the draft model changes is how fast the tokens arrive: a bad
        one simply gets rejected more often and the speedup shrinks toward 1x.

        Every step emits at least one token - the one the target itself
        produces - so progress is guaranteed even when every guess is wrong.
        Typical speedups are 2-3x, governed by the acceptance rate: how often
        a cheap model happens to agree with an expensive one. It agrees often,
        because much of any text is easy - whitespace, function keywords,
        the second half of a word already begun, common phrasing.

        Variants differ mainly in where the draft comes from: a separate small
        model of the same family, n-gram or prompt lookup (no model at all -
        propose text copied from the context, which works well on summarization
        and code editing), or extra prediction heads grafted onto the target
        itself (Medusa, EAGLE), which avoids serving two models.

    WHAT THIS FILE DOES INSTEAD
        GPT.generate has no KV cache: every step re-runs the full forward over
        the whole context (see the crop at `idx_cond = idx[:, -BLOCK_SIZE:]`).
        Every step is therefore a small prefill, and the K and V of the earlier
        tokens are recomputed from scratch each time. Correct, and far simpler
        to read - but O(T) times the work.

        Adding a cache here would be more than a speed change. This model uses
        ABSOLUTE LEARNED position embeddings, added once at the start
        (GPT.forward). As soon as the context passes BLOCK_SIZE the window
        slides, every token's position index shifts, and the cached K/V - which
        already have the old positions baked in - become wrong, so the cache
        would have to be rebuilt. RoPE avoids this, because it applies position
        by rotating Q and K inside each attention, based on relative distance.
        That is a large part of why modern models moved to it.

--------------------------------------------------------------------------------
 MODERN MODELS  -  what production systems swap out, and what they keep
--------------------------------------------------------------------------------
    The skeleton in this file is the one still in use: embed, N blocks of
    (attention + MLP) with residuals and pre-norm, project to the vocabulary,
    sample. The differences are component swaps inside that same design.

      this file            modern models        why
      --------------------------------------------------------------------
      LayerNorm            RMSNorm              same job, no mean subtraction
                                                and no bias: cheaper, as stable
      learned absolute     RoPE                 encodes RELATIVE distance, by
      position embedding                        rotating Q and K inside each
                                                attention instead of adding a
                                                vector once at the input
      ReLU                 SwiGLU               gated activation, better loss
                                                at equal parameter count
      H independent        GQA / MQA            K and V shared across groups of
      K/V per head                              heads: shrinks the KV cache,
                                                which is what limits batch size
      no cache             KV cache             see INFERENCE, above
      -                    Flash Attention      never materializes the T x T
                                                matrix in memory; same math
      pre-norm residuals   pre-norm residuals   unchanged
      softmax attention    softmax attention    unchanged
      next-token           next-token           unchanged, at every scale:
      cross-entropy        cross-entropy        the objective is the same

    Note on scale: the bottleneck story above (compute-bound vs memory-bound,
    weight bandwidth, batching) is about GPUs moving tens of gigabytes of weights.
    GlassGPT has 866k parameters and runs on a CPU, so none of it is observable
    here. It is documented because the architecture is the same one: GPT.generate
    runs a small prefill at every step, which is the shape real serving pays for
    only once.

--------------------------------------------------------------------------------
 GLOSSARY  -  every technical term in this file, one line each
--------------------------------------------------------------------------------
  DIMENSIONS & TENSORS   (B, T, C, V, H, hs: see "THE DIMENSIONS THAT SHOW UP
                          EVERYWHERE" at the top)
    tensor        a multi-dimensional array of numbers; PyTorch's core data type
    shape         a tensor's measurements, e.g. (B, T, C)
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
    dot product   sum of the products of matching components: positive when two
                  vectors point the same way, zero when perpendicular, negative
                  when opposed. The alignment test underlying every matmul here
    direction     a line through the embedding space that training has given a
                  meaning; features are stored along directions of this kind
    feature       any property a model represents internally (a topic, a name,
                  a tone). Rarely one neuron: see superposition

  ATTENTION
    attention     mechanism by which each token gathers information from others
    self-attention  tokens attend to their own sequence
    Query (Q)     projection of a token: what it is looking for in the context
    Key (K)       projection of a token: what it offers in reply to Queries
    Value (V)     projection of a token: the edit to add where it is attended to
    affinity      score (Q . K^T) of how relevant one token is to another
    attention     the T x T grid of weights after the softmax: row i says how
      pattern     token i distributed its attention over the tokens before it
    output matrix (W_O) the block's final C -> C projection; it holds the
                  second half of every head's Value map (see MultiHeadAttention)
    cross-attn.   variant where Queries and Keys come from DIFFERENT sequences
                  (translation, image-text). Not used here: this file is
                  self-attention only
    scaling       dividing by sqrt(hs): bounds the dot-product magnitude so the
                  softmax does not saturate
    causal mask   (tril) prevents a token from attending to FUTURE positions
    softmax       turns a list of scores into probabilities that sum to 1
    multi-head    several heads in parallel, each learning a different relation

  PROCESSING & STRUCTURE
    MLP / feed-forward  small net applied to EACH token alone: the
                  per-position transform
    ReLU          zeroes every negative value: the non-linearity used here
    non-linearity what stops two linear layers from collapsing into one
    neuron        one of the 4C values leaving the ReLU inside the MLP; ACTIVE
                  when positive, inactive when zero
    superposition storing more features than there are dimensions by placing
                  them along nearly (not exactly) perpendicular directions, at
                  the cost of slight interference. Why single neurons are not
                  individually readable (see FeedForward)
    residual      the x = x + layer(x) pattern: adds the sublayer output to its
                  input, preserving signal and gradient through depth
    layernorm     normalizes each token (mean 0, uniform scale): stable training
    block         one Transformer block = attention (cross-position mix) +
                  MLP (per-position transform)
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
    dropout       randomly zeroes activations in training: regularization
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
    top-p         (nucleus) keep the smallest set of tokens whose probabilities
                  sum to p; not implemented here, top-k is used instead
    greedy        always take the argmax instead of sampling; then the final
                  softmax is unnecessary, since it does not change the ranking

  SERVING  -  terms from INFERENCE, above (not implemented here)
    prefill       the one forward that processes the whole prompt at once
    decode        the per-token forwards that follow, one token each, in series
    TTFT          time to first token: how long prefill takes
    TPOT          time per output token: how fast decode produces text
    KV cache      the stored K and V of every past token, so they are computed
                  once instead of being recomputed at every generated token
    GEMM / GEMV   matrix x matrix / vector x matrix: the shape prefill and
                  decode respectively reduce to
    compute-bound the GPU's arithmetic units are the limit (prefill)
    memory-bound  reading the weights from memory is the limit (decode)
    batching      serving many requests in one pass; a huge win in decode (one
                  weight read serves all of them) and no help in prefill
    speculative   a small draft model proposes several tokens; the real model
      decoding    verifies them in one pass and keeps the ones it agrees with.
                  Exploits decode's idle compute; provably preserves the output
                  distribution and changes only the speed
    draft model   the small, fast model that proposes tokens in speculative
                  decoding
    acceptance    the fraction of proposed tokens the target model keeps: what
      rate        determines the speedup

  MODERN VARIANTS  -  what this file does NOT use (see MODERN MODELS, above)
    RMSNorm       LayerNorm without mean subtraction or bias
    RoPE          rotary position embedding: position applied by rotating Q and
                  K inside attention, encoding RELATIVE distance
    SwiGLU        gated activation used in the MLP instead of ReLU
    GQA / MQA     several heads sharing one K/V pair, to shrink the KV cache
    Flash Attn.   attention computed without ever storing the T x T matrix

  SAVING
    checkpoint    file with weights + tokenizer + config: reuse without retraining
    state_dict    the dictionary of all the weights the network learned

--------------------------------------------------------------------------------
 HOW TO RUN
--------------------------------------------------------------------------------
    python glassgpt.py

    Drop a UTF-8 text file named `input.txt` next to this script to train on
    custom data: a book, notes, song lyrics. The more (and more varied) the text,
    the better it learns. If `input.txt` is missing, a small built-in sample is
    used instead, so the script runs with no setup.
================================================================================
"""

import math
import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

# Windows: force the console to UTF-8. The BPE works on bytes, so generation can
# produce accented or special characters that the console's default encoding
# (cp1252) cannot print, and the program would die on the first such print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==============================================================================
# 1) CONFIGURATION (hyperparameters)
# ------------------------------------------------------------------------------
# Every number below is a design choice. Raising N_EMBD / N_LAYER / MAX_ITERS
# buys capacity and costs training time; the values here are small enough to
# train in minutes on a laptop CPU.
#
# How the hyperparameters shape the network:
#                       C = N_EMBD = 128  (width: numbers per token)
#                    <------------------->
#                    +-------------------+   |
#      one token ->  | . . . . . . . . . |   |
#                    +-------------------+   |
#      T = 64 tokens | . . . . . . . . . |   |  depth N_LAYER = 4 blocks
#      of context    | . . . . . . . . . |   |  stacked on top of each other
#                    |        ...        |   |
#                    +-------------------+   |
#                    C splits into H=4 heads -> hs = C/H = 32 each
#                    [_hs_|_hs_|_hs_|_hs_]
# ==============================================================================

# --- How much context the model uses, and how many sequences it processes ---
BLOCK_SIZE = 64  # T: maximum context length. To predict the next token, the
#    model attends to at most the previous 64 tokens. With this 512-token BPE a
#    token covers roughly 2 to 4 characters depending on how repetitive the text
#    is (the run prints the ratio it measured), so 64 tokens reach well past 64
#    characters. Larger = more context, but attention cost grows with T squared.
BATCH_SIZE = 32  # B: how many independent sequences are processed each step.
#    Processing 32 together is more efficient than one at a time. Larger = less
#    noisy gradient but more RAM/CPU.

# --- Network dimensions ---
N_EMBD = 128  # C: each token and each position becomes a vector of 128 numbers.
#    The network "width". Must be divisible by N_HEAD.
N_HEAD = 4  # H: number of attention heads. Each head works on 128/4 = 32
#    dimensions and learns a different kind of relation between tokens.
N_LAYER = 4  # how many Transformer blocks are stacked: the network "depth".
DROPOUT = 0.1  # fraction of activations randomly zeroed during training (inside
#    the attention head it is the attention weights that get dropped). This is
#    regularization: it prevents the network from relying on any single unit,
#    which reduces overfitting.
INIT_STD = 0.02  # standard deviation of the Gaussian weight initialization.
#    Small = bounded outputs at the start. Residual projections use a reduced std,
#    INIT_STD/sqrt(2*N_LAYER) (see GPT._init_weights).

# --- BPE tokenizer ---
BPE_VOCAB_SIZE = 512  # vocabulary size: 256 base bytes + (512-256) = 256 learned
#    merges. Bigger = longer tokens (shorter sequences) but a larger embedding
#    table.

# --- Optimization (how, and how long, it learns) ---
MAX_ITERS = 2000  # number of training steps (forward + backward + update).
EVAL_INTERVAL = 300  # how often train and validation loss are printed.
EVAL_ITERS = 100  # how many batches the loss is averaged over when evaluating
#    (a single measurement would be too noisy).
LEARNING_RATE = 3e-4  # size of each weight-update step. 3e-4 = 0.0003: a typical
#    value for AdamW on Transformers. Too large = the updates overshoot and the
#    loss diverges; too small = the loss decreases very slowly.

# --- Reproducibility and hardware ---
SEED = 1337  # fix randomness: identical runs every time.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # "cpu" unless a CUDA
#    GPU is present; the defaults here are sized for the CPU case.

torch.manual_seed(SEED)  # makes weight init and batches deterministic.

# Sanity check: each head must receive a whole number of dimensions.
assert N_EMBD % N_HEAD == 0, "N_EMBD must be divisible by N_HEAD"
HEAD_SIZE = N_EMBD // N_HEAD  # hs: size of a single head.

# --- Generation: how to sample the produced text (see GPT.generate) ---
PROMPT = ""  # the starting text the model continues from. Empty = start from a
#    minimal context. This model has no separate system/user roles: a prompt is
#    text prepended to the context. With byte-level BPE any character is
#    encodable: there are no out-of-vocabulary tokens.
MAX_NEW_TOKENS = 500  # how many tokens to generate after the prompt (a few
#    characters each with BPE, so well over 500 characters of text).
TEMPERATURE = 0.8  # sampling sharpness; see GPT.generate.
TOP_K = 50  # at each step keep only the 50 most probable tokens (None = all).

# --- Saving / loading the trained model ---
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model.pt"
)  # where the weights + the vocabulary are saved to and loaded from.
FORCE_RETRAIN = False  # True = ignore the checkpoint and always retrain from scratch.


# ==============================================================================
# 2) DATA + BPE TOKENIZER (text <-> numbers)
# ------------------------------------------------------------------------------
# A neural network operates on numbers, so the first step is to map text to
# integer token ids. This is tokenization, and it happens entirely outside the
# network - 0 parameters. The BPETokenizer class below covers how the mapping
# is learned and why it starts from bytes.
#
#     "in the middle of the journey"
#            |  encode()               ^   decode()
#            v                         |
#     [ bytes 0..255 ] --learned merges--> [ token ids: 110, 288, 41, ... ]
#                                                  |
#                                                  v
#                                     the network works ONLY on these numbers
# ==============================================================================

# A VARIED text (not a few repeated lines) is essential for BPE. On trivially
# repetitive text the BPE would merge everything into very few tokens, leaving the
# network without data. The text below is the opening of Dante's Inferno (21 distinct
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
) * 40  # repeated to have enough data; the block itself stays varied.


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.

    The idea, in two steps
        1) Start from BYTES. Any text, in any language, in UTF-8 is a sequence of
           bytes (0..255). So the base vocabulary has 256 tokens and there are no
           "unknown" characters: an emoji or a rare letter is simply more bytes.
           (A char-level vocabulary breaks on any character it never saw during
           training.)
        2) Learn to MERGE the most frequent pairs. The two tokens that are
           adjacent most often (e.g. 'i'+'n') are fused into a new token
           ('in'). Repeat: now maybe 'in'+'g' -> 'ing'. Continue until the
           vocabulary reaches the desired size (BPE_VOCAB_SIZE).

    How a byte pair becomes a token (one BPE merge)
        text:   c a m m i n   c a m m i n a
        bytes: [99][97][109][109][105][110] ...   (0..255, the base vocab)

        step 1) count adjacent pairs:
                (109,109)="mm" x2   (97,109)="am" x2   (99,97)="ca" x2 ...
                (on a fragment this short several pairs tie at 2, and max()
                 breaks the tie by insertion order; the trace below follows
                 "mm" because it is the easiest to read)
        step 2) merge the most frequent -> "mm" becomes the new id [256]
                c a [256] i n   c a [256] i n a
        step 3) repeat: now "ca" -> [257], then "[257]mm" -> [258] ...
                [258] i n   [258] i n a         (tokens grow longer)

    Why it improves on char-level
        Tokens become recurring word pieces, so the same sentence turns into a
        SHORTER sequence of ids. More text fits inside the same BLOCK_SIZE, and
        each position the network attends to holds a whole word piece.

    Note: this is a minimal implementation - see the two limitations spelled out
    in encode().
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        # Return the tokenizer to its "blank" state (just the 256 bytes, no
        # merges). Used by the constructor and at the start of train(), so the
        # same instance can be re-trained without carrying old merges along.
        #
        # merges: (id_a, id_b) -> new_id. The learned merges, in the order they
        # were learned (that order is also the PRIORITY in which they apply).
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

        Returns a dict {(id_a, id_b): how_many}. This is the step that identifies
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
        merge (see the padding at the end). This keeps vocab_size deterministic
        and equal to what the header assumes (V = 512), so the parameter count
        always holds.
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
        1) COST: _get_stats is recomputed over the whole sequence at EVERY merge,
           so the cost is ~O(n * number_of_merges). On a large input.txt this is
           slow; a production implementation uses incremental data structures.
        2) MERGING ACROSS WORDS: the text is not split into words first, so BPE
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

    Sets the global variables used by the model and by get_batch. Called only on
    the training path (on the loading path the tokenizer comes from the checkpoint).

        input.txt (or FALLBACK) -> train BPE -> encode -> tensor of ids
                                                            |
                                              90% train ---+--- 10% validation
    """
    global tokenizer, encode, decode, VOCAB_SIZE, train_data, val_data

    # 1) read the text (from input.txt, or the built-in sample)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[data] Loaded input.txt: {len(text):,} characters")
    else:
        text = FALLBACK_TEXT
        print("[data] input.txt not found: using the built-in sample text.")
        print("       (drop an 'input.txt' in this folder and rerun)")

    # 2) train the BPE up to BPE_VOCAB_SIZE tokens
    print(f"[bpe] Training the BPE tokenizer (target {BPE_VOCAB_SIZE} tokens)...")
    tokenizer = BPETokenizer()
    tokenizer.train(text, BPE_VOCAB_SIZE)
    encode = tokenizer.encode
    decode = tokenizer.decode
    VOCAB_SIZE = tokenizer.vocab_size
    print(f"[bpe] Final vocabulary: {VOCAB_SIZE} tokens")

    # 3) encode all the text into ids, then split 90% train / 10% validation.
    # Evaluation uses held-out validation data: measuring on training data would
    # overstate performance.
    ids = torch.tensor(encode(text), dtype=torch.long)
    print(
        f"[data] Encoded text: {len(ids):,} tokens "
        f"(compression {len(text) / max(len(ids), 1):.2f}x vs characters)"
    )
    n_split = int(0.9 * len(ids))
    train_data = ids[:n_split]
    val_data = ids[n_split:]

    # Guard: after BPE both train and val need more than BLOCK_SIZE tokens,
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
                   [-x, T tokens-]
         x  =      10  22   5   8               context in
         y  =          22   5   8  41           target = x shifted by +1
                   ^   ^    ^   ^
                   |   |    |   +-- after [10,22,5,8]  the target is 41
                   |   |    +------ after [10,22,5]    the target is 8
                   |   +----------- after [10,22]      the target is 5
                   +--------------- after [10]         the target is 22

        Repeated B=32 times from B random start points -> x, y shape (B, T).
        In one pass the network makes T next-token predictions per sequence.

    Returns two tensors of shape (B, T).
    """
    d = train_data if split == "train" else val_data
    # Pick B random start positions. -BLOCK_SIZE so the window stays in bounds.
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
        A token embedding is a lookup with no knowledge of its surroundings (see
        "WHAT THE NUMBERS IN A VECTOR MEAN" in the header). This class is the
        machinery that fixes that: it decides, for each position, which earlier
        positions are relevant, and what to add to that position as a result.

    Query, Key, Value - a question, an answer, and an edit
        Every token vector is multiplied by three separate weight matrices,
        producing three vectors that play three different roles. Take the
        phrase "a fluffy blue creature":

            Query (Q)  what this token is looking for. The noun "creature"
                       emits, in effect, "any adjectives to the left?"
            Key   (K)  what this token offers in reply. "fluffy" and "blue"
                       emit keys that line up with that question.
            Value (V)  what to add if the answer matched - the edit that turns
                       a generic "creature" into "fluffy blue creature".

        The matching happens BETWEEN tokens: the Query of "creature" is
        compared against the Key of every token before it, one dot product per
        pair. A large dot product means that Key answers that Query, so those
        products are the first thing the head computes. The scores then become
        weights, and the weights are applied to the Values; "ONE HEAD" below
        maps that sequence line by line onto the code.

        V is a CHANGE to be added to the residual stream (see Block): whatever
        the head produces is summed into what the token already carried.

        This adjective-noun example is a fiction, chosen because it is easy to
        picture. A real head's matrices are tuned to whatever reduces the loss,
        and one head typically does several unrelated things at once.

    Self & causal
        Self   - the Q, K, V all come from the same sequence.
        Causal - a token attends only to positions <= its own; future positions
                 are masked, since they hold the tokens still to be predicted.

    ONE HEAD: THE FORMULA, THE CODE, THE SHAPES
        In the literature the whole head is a single line:

            Attention(Q, K, V) = softmax( Q . K^T / sqrt(d_k) ) . V

        Each piece of it is one line of forward() below (hs = C / H = 32, and
        d_k IS hs - the two names mean the same thing):

          Q, K, V        self.query(x) / .key(x) / .value(x)   (B,T,hs) each
          Q . K^T        q @ k.transpose(-2, -1)               (B,T,T)
          / sqrt(d_k)    * (HEAD_SIZE**-0.5)                   (B,T,T)
          (causal mask)  .masked_fill(tril == 0, -inf)         (B,T,T)
          softmax(...)   F.softmax(wei, dim=-1)                (B,T,T)
          ... . V        wei @ v                               (B,T,hs)

        The published formula leaves out the mask. It is the one addition
        that makes attention CAUSAL, and therefore usable to predict the next
        token. The 1/sqrt(hs) scaling stops the dot products from growing with
        hs, which would saturate the softmax and vanish the gradients.

    WHAT THE ATTENTION MATRIX LOOKS LIKE
        `wei` after the softmax IS the attention matrix A from the formula.
        Here it is for the sentence "a fluffy blue creature roamed the verdant
        forest", in one head. Rows = the token doing the attending (its Query).
        Columns = the token being attended to (its Key).

                       a  fluf  blue  crea  roam   the  verd  fore
        a        [  1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00 ]
        fluffy   [  0.42  0.58  0.00  0.00  0.00  0.00  0.00  0.00 ]
        blue     [  0.19  0.23  0.58  0.00  0.00  0.00  0.00  0.00 ]
        creature [  0.04  0.31  0.44  0.21  0.00  0.00  0.00  0.00 ]
        roamed   [  0.06  0.12  0.09  0.55  0.18  0.00  0.00  0.00 ]
        the      [  0.08  0.05  0.04  0.21  0.34  0.28  0.00  0.00 ]
        verdant  [  0.02  0.05  0.11  0.08  0.06  0.14  0.54  0.00 ]
        forest   [  0.01  0.03  0.22  0.06  0.02  0.05  0.54  0.07 ]

        What the grid shows:

          - EVERY ROW SUMS TO 1. That is the softmax: each token spends exactly
            one unit of attention, distributed over the tokens it can see.
          - THE UPPER TRIANGLE IS ALL ZERO. That is the causal mask: the -inf
            placed there before the softmax comes out the other side as exactly
            0, so "blue" gets no access to "forest".
          - ROW 1 IS 1.00 ON ITSELF, necessarily. The first token has nothing
            else to look at, so its softmax runs over a single unmasked value.
            Every model does this, so that 1.00 says nothing about what this
            one learned.

        A row is the record of what one token decided to pay attention to.
        "forest" drew 54% of its Value from "verdant" and 22% from "blue" - the
        two adjectives that describe it - and almost nothing from "the".

        Note the shape: T x T. It grows with the SQUARE of the context, which
        is what makes long contexts expensive: the weights do not grow with the
        prompt, this matrix does. Flash Attention produces the same result
        without ever storing it.

    A NOTE ON TRANSPOSED DIAGRAMS
        Many well-known figures (and some papers) write the formula as
        softmax(K^T Q / sqrt(d_k)) V, drawing Queries along the COLUMNS and Keys
        along the ROWS - the transpose of the layout above and of the code. Both
        are correct and describe the same operation; only the axes are swapped.
        When comparing a diagram with this code, check first which axis is the
        Query, otherwise the causal mask appears to be on the wrong side.
    """

    def __init__(self):
        super().__init__()
        # Q, K, V projections (weight matrices) mapping each token vector (C) into
        # the three views (hs each). bias=False: no bias term on any of them.
        # They are three independent linear projections of the same input.
        self.key = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)  # C -> hs
        self.query = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)  # C -> hs
        self.value = nn.Linear(N_EMBD, HEAD_SIZE, bias=False)  # C -> hs
        self.dropout = nn.Dropout(DROPOUT)
        # The causal mask (tril) is NOT defined here. It is identical for all heads,
        # so it is kept once in MultiHeadAttention and passed to forward, avoiding
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
        # affinities between token i and token j. On the sqrt(hs) division, see
        # the docstring above.
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
    between positions. Their outputs are concatenated along the channel
    dimension and passed through a final linear projection.

        x (B,T,128) --+--> Head 1 --> (B,T,32) --+
                      +--> Head 2 --> (B,T,32) --+  concat   proj (128->128)
                      +--> Head 3 --> (B,T,32) --+--> (B,T,128) --> (B,T,128)
                      +--> Head 4 --> (B,T,32) --+
                            |                        |            |
                      independent Q/K/V      4x32 = 128       mix across
                      per head               (back to C)      the heads

    Note: 4 heads of 32 cost the same as 1 head of 128 (same total C), split into
    H independent subspaces. With the parameter count identical, the gain comes
    from the softmaxes: each head runs its own, so a position can attend to the
    adjective on its left and to the subject ten tokens back at the same time.
    One wide head has a single attention distribution to spend, and splitting it
    between two targets means giving up on both.

    WHY THE VALUE MAP IS SPLIT IN TWO
        A head's edit has to land in the residual stream, which is C-dimensional,
        so conceptually the Value map goes C -> C. Written that way it would cost
        C*C = 16,384 parameters per head, against 4,096 each for Query and Key -
        one map dominating the other two, and the imbalance gets worse with
        every head added.

        So the map is factored through a narrow bottleneck. Head.value handles
        the way down, C -> hs. The way back up, hs -> C, lives in a single
        matrix shared by the whole block: self.proj below, where the H up-maps
        sit side by side.

            head 1 -> concat slots   0..31   |  proj takes all 128 slots at
            head 2 -> concat slots  32..63   |  once, but the 32 slots of a
            head 3 -> concat slots  64..95   |  given head are only ever
            head 4 -> concat slots  96..127  |  touched by their own 128 x 32
                                             |  slice of its matrix

        The arithmetic is exact: proj's matrix holds 128 * 128 = 16,384 weights,
        which is 4 heads * (32 * 128), i.e. 4,096 per head. A head's complete
        Value map therefore costs 4,096 down + 4,096 up - the same as its Query
        and its Key. The concat-then-project code below runs all H up-maps as a
        single matmul. (proj's bias belongs to the block as a whole.)

        The split also constrains each head's Value map to rank hs: every edit
        it can produce has to squeeze through 32 dimensions on the way. And it
        explains a naming confusion in the papers: "the value matrix" usually
        means only the down part (C -> hs), while the up part is described
        separately as the block's output projection, W_O.
    """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])
        # Final projection applied after concatenating the heads. H*hs = C, so the
        # concatenation has dimension C and this Linear maps C -> C.
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        # This Linear adds to the residual stream (see Block): it is marked so that
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

    The 4x expansion (C -> 4C -> C) is where the layer's capacity comes from:
    4C hidden units against a stream only C wide. ReLU (zero out negatives)
    introduces the non-linearity: without it two Linears in a row would collapse
    into a single linear map.

    In large models this layer holds the majority of the parameters: 8*C^2 per
    block against attention's 4*C^2, so two thirds of every block goes to the
    MLP - 115.9B of GPT-3's 175.2B (see the weight table in the header). Modern
    models replace the ReLU with a gated activation (SwiGLU), which changes the
    shape of this layer but not its role.

    WHAT THE TWO MATRICES DO
        Attention moves information between positions. This block leaves every
        position alone - so what is two thirds of the network for? This is
        where FACTS are stored, and the two matrices play opposite roles: the
        first asks questions, the second writes answers.

        (nn.Linear stores its weight as (out_features, in_features), which is why
        the two matrices below are read along opposite axes.)

        Read the UP-PROJECTION BY ROWS. Each of its 4C rows is a vector in the
        same space as x, so each row's dot product with x asks "how much does
        this token point along that row's direction?". Suppose one row held
        the direction (first name Michael) + (last name Jordan). Each half that
        matches contributes about 1 to the dot product, so that row's output is:

            2  if x encodes the full name "Michael Jordan"
            1  if x encodes only one of the two ("Michael Phelps")
            0  or negative if neither

        The bias adds a threshold. With -1 in that position the row's output
        becomes 1 for the full name, 0 for a single half, negative for neither.

        The ReLU then clips everything negative to zero, and what comes out is
        a clean yes/no: nonzero for "Michael Jordan", flat zero for "Michael
        Phelps" or "Alexis Jordan". Without it the layer would leak partial
        credit for a half-match. The pair (threshold, then clip) behaves like an
        AND gate.

        The values leaving the ReLU are what people mean by the NEURONS of a
        transformer: 4C of them per token, each said to be ACTIVE when positive
        and inactive when zero.

        Read the DOWN-PROJECTION BY COLUMNS. Each of its 4C columns is also a
        vector in the space of x, and the output is the sum of those columns,
        each scaled by its neuron. So a column is what gets ADDED to the token
        when its neuron fires. If the column paired with the "Michael Jordan"
        neuron holds the direction for "basketball", then the block has stored a
        fact: a vector encoding that name flows in, and flows out with
        "basketball" added to it.

            up-projection, by rows      down-projection, by columns
            "is this X?"                "then add Y"
            row . x -> neuron           neuron * column -> added to the stream

        This whole example is a clean fiction, useful for holding the mechanism
        in mind. Real networks are considerably messier, for the reason below.

    SUPERPOSITION - why the neurons are not individually readable
        Looking inside a trained model rarely reveals a "Michael Jordan neuron".
        Features are spread across many neurons at once, and there is a good
        reason why.

        Features stored along EXACTLY perpendicular directions never disturb
        one another, and a C-dimensional space holds exactly C such directions
        - so that arrangement fits C features and no more. Loosen the
        requirement to NEARLY perpendicular, say 85 degrees instead of 90, and
        the number of directions available grows exponentially with C (a
        consequence of the Johnson-Lindenstrauss lemma). At GPT-3's 12,288
        dimensions that is room for something on the order of tens of billions
        of nearly independent directions. The exact figure depends on how much
        interference is tolerated; the point is how fast it climbs with C.

        So a model has strong incentive to pack far more features than it has
        dimensions, accepting a little interference between them. The cost is
        paid in legibility: a feature ends up as a pattern ACROSS many neurons
        - a superposition. This also suggests why capacity scales so well with
        width: doubling C does much more than double the number of ideas that
        fit.

        At this file's C = 128 there is very little room for any of this. The
        mechanism still runs, on a space far too small for it to pay off.
    """

    def __init__(self):
        super().__init__()
        # The second Linear (4C -> C) is the projection that adds to the residual
        # stream: it is defined separately to mark it and give it the scaled
        # init.
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
    """One Transformer block = attention (the cross-position mix) followed by
    the MLP (the per-position transform).

    N_LAYER of them are stacked. Deep stacks rely on two mechanisms present in
    every Transformer:

    1) RESIDUAL (x = x + sublayer(x)): the sublayer output is ADDED to its input
       rather than replacing it. This gives an identity path along which the
       activations and the gradient flow undamped through depth. Without it the
       gradient vanishes after a few layers and the network fails to train at all.

    2) LAYERNORM (pre-norm: normalize BEFORE the sublayer): normalizes each token's
       activations to zero mean and unit variance, then applies a learned scale and
       shift. This bounds the activation magnitude and stabilizes training.

        x -----------------------------(+)------------------------------(+)----> x
        |                               ^                                ^
        |  the input is added           |  identity path again           |
        |  unchanged                    |                                |
        +--> LayerNorm --> Multi-Head --+   +--> LayerNorm --> FeedFwd --+
                           (mix positions)                     (per-position)

    forward() below is exactly those two lines. Large models stack dozens of
    these blocks; this file stacks N_LAYER = 4.
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

    Why embeddings? A token id is just a label: nothing about the number 41 says
    what token 41 means. The embedding maps each id to a learned vector of C real
    numbers. The vectors start random; training adjusts them so that tokens used
    in similar contexts get similar vectors.

    Why also positions? Self-attention is permutation-invariant: without position
    information "abc" and "cba" would produce the same representation. The position
    embedding adds, to each token, a learned vector encoding its index 0, 1, 2...
    The two embeddings are summed, so x carries meaning AND position.

    For the shapes at every step, see the master diagram in the header; forward()
    below is that diagram in code.
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
        # have the SAME shape and represent the same token<->vector relation. They
        # are tied: the same tensor is reused for both. Benefits: ~V*C fewer parameters
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
          smaller std, 0.02/sqrt(2*N_LAYER). Reason: at each layer TWO
          contributions (attention + MLP) are added to the residual stream;
          without damping, the signal variance grows layer after layer and
          destabilizes training. The 1/sqrt(2*N_LAYER) factor exactly
          compensates that cumulative sum.

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
        # device taken from idx, NOT from the global DEVICE: the model then runs
        # on the same device as its inputs, with no external constant involved.
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
        # expects logits (N, V) and targets (N,), so the B and T dimensions are
        # flattened together: N = B*T.
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

        - temperature: divides the logits before the softmax. Below 1 sharpens
          the distribution and the output turns repetitive; above 1 flattens it
          and unlikely tokens start appearing; 1 changes nothing. The mechanics
          are in the header, under "SOFTMAX, IN BOTH PLACES IT APPEARS".
        - top_k: before sampling, keep only the K most probable tokens and set
          the rest to probability 0, cutting off the low-probability tail.
          None = no filter.

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

        WHY THE LOGITS ARE THROWN AWAY
            The forward returns (B, T, V): a prediction at EVERY position. Only
            [:, -1, :] is kept; the other T-1 rows are discarded. Training uses
            all of them - that is how the model learns T predictions per
            sequence in one pass (see get_batch) - while generation only ever
            needs the prediction sitting at the end of the context.

        THIS LOOP HAS NO KV CACHE
            Real inference stores K and V instead of recomputing them, which
            splits generation into a prefill phase and a decode phase with very
            different bottlenecks - see the INFERENCE section in the header.
        """
        temperature = max(temperature, 1e-6)  # avoid division by zero
        for _ in range(max_new_tokens):
            # The model attends to at most BLOCK_SIZE tokens: crop the context.
            idx_cond = idx[:, -BLOCK_SIZE:]  # (B, <=T)
            logits, _ = self(idx_cond)  # (B, T, V)
            # Only the LAST position matters (the next-token prediction); it is
            # divided by the temperature (see docstring).
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
# The network is trained by four steps repeated MAX_ITERS times:
#   1) BATCH    : get_batch cuts B random windows out of the text (x, and y = x
#                 shifted by one).
#   2) FORWARD  : model(x, y) runs the network and returns the prediction error
#                 (the loss - see GPT.forward, cross-entropy).
#   3) BACKWARD : loss.backward() computes, for EACH of the ~866,000 weights
#                 (this configuration, weight tying included), the gradient of the
#                 loss with respect to that weight (its magnitude and sign).
#   4) UPDATE   : optimizer.step() moves each weight along the negative gradient,
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
# ==============================================================================


@torch.no_grad()
def estimate_loss(model):
    """Measure the average loss on train and validation.

    The loss is averaged over EVAL_ITERS batches because a single measurement
    would be too noisy. model.eval()/train() switch dropout off/on (it must be
    off when evaluating).

    Why measure TWO losses:
      - train loss = error on data seen during training;
      - val loss   = error on data NEVER seen.

        loss  ^
              |\\
              | \\___  val                  they fall TOGETHER
              |  \\__  train                -> really learning (generalizes)
              +------------------> iterations

        loss  ^
              |\\
              | \\_/   val                 val turns back UP -> OVERFITTING
              |  \\__  train               while train keeps falling
              +------------------> iterations   (memorizing, not generalizing)

    With the built-in sample text - 21 distinct lines repeated 40 times - the
    network can largely memorize the block, so val loss drops a lot, though not
    to zero. On genuinely large and varied text it settles higher, and that
    higher number is what real generalization looks like.
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
    # ln(28) = 3.33). This is the reference value. At the start the weights are
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


# ==============================================================================
# 6) PERSISTENCE, GENERATION, ENTRY POINT
# ==============================================================================


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
    """Reload a saved model. Returns the ready-to-use model, or None if the file
    cannot be read (missing, corrupt, or written by an incompatible version).

    Editing the hyperparameters at the top of the file is NOT such a failure: the
    checkpoint is self-describing, so its saved values win and the network is
    rebuilt to match them. See the warning printed below.

    IMPORTANT: the EXACT BPE tokenizer the model was trained with (the learned
    merges) is restored too, so generation works even if the original text is
    gone. If loading fails, everything is put back as it was, so a later retrain
    stays consistent.
    """
    global tokenizer, VOCAB_SIZE, encode, decode
    global BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, HEAD_SIZE
    backup = (
        tokenizer, VOCAB_SIZE, encode, decode,
        BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, HEAD_SIZE,
    )
    try:
        # weights_only=False: the checkpoint also holds Python objects
        # (merges, vocab, config).
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        # restore the saved model's BPE tokenizer
        tokenizer = BPETokenizer()
        tokenizer.merges = ckpt["tokenizer"]["merges"]
        tokenizer.vocab = ckpt["tokenizer"]["vocab"]
        encode = tokenizer.encode
        decode = tokenizer.decode
        VOCAB_SIZE = tokenizer.vocab_size
        # Apply the saved ARCHITECTURE to the globals: the checkpoint is
        # self-describing, so it loads correctly even if the hyperparameters at
        # the top of the file changed in the meantime (otherwise the network
        # shape would not match the saved weights).
        cfg = ckpt["config"]
        # ANTI-SURPRISE WARNING: when the hyperparameters in the file differ from
        # an existing model.pt, the (self-describing) checkpoint WINS and the
        # edits are ignored. The mismatch is reported explicitly before the
        # globals are overwritten with the saved values.
        current = {
            "block_size": BLOCK_SIZE, "n_embd": N_EMBD,
            "n_head": N_HEAD, "n_layer": N_LAYER,
        }
        diff = {k: (current[k], cfg[k]) for k in current if current[k] != cfg[k]}
        if diff:
            print("[model] WARNING: the hyperparameters in the file do NOT match")
            print("        the checkpoint. Using the SAVED ones (the checkpoint")
            print("        wins). To apply the new values set FORCE_RETRAIN = True")
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
    # is empty, generation starts from a newline, as a minimal context.
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
    # 1) Reuse a checkpoint if there is one, otherwise train from scratch.
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
