# Arrow of Time in LLMs with POS-Tagged Data

## Project Description

This project investigates the **"Arrow of Time" phenomenon in Large Language Models (LLMs)** by examining how sequence directionality (forward vs. backward) affects language modeling performance when only syntactic information is available. We **isolate syntactic structure** from semantic content by converting text data into sequences of Part-of-Speech (POS) tags. Two novel tokenization schemes are applied to POS-tagged corpora, and Transformer language models (based on a modified nanoGPT implementation) are trained on these sequences. By comparing models trained on normal (forward) text order versus reversed (backward) text, we explore whether an inherent "arrow of time" exists in language modeling – i.e., whether the natural forward direction of language provides an advantage in learning syntactic patterns.

## Objectives and Research Questions

- **Is there an “arrow of time” in language models?** – Do LLMs trained on forward-ordered text outperform those trained on backward-ordered text when semantics are removed, indicating a directional bias in learning syntax?
- **Syntactic Structure Learning:** How well can a language model learn and predict sequences of POS tags (syntactic structure) in the absence of lexical meaning?
- **Impact of Tokenization Scheme:** What is the effect of different POS token encoding schemes on model performance? We compare:

  - _Unique POS ID mapping_ – each POS tag mapped to a single unique token ID.
  - _Random ID within tag-specific ranges_ – each POS tag assigned a range of token IDs, with each token occurrence randomly assigned an ID from that range.

- **Forward vs. Backward Performance:** How does model performance (e.g. cross-entropy loss or perplexity on a validation set) differ between a model trained on forward sequences and one trained on backward (reversed) sequences, under each tokenization scheme?
- **Theoretical vs. Observed Results:** Can we predict the performance of the more complex random-ID model based on the simpler unique-ID model (e.g., by accounting for the added uncertainty of random token assignments), and do the empirical results align with these predictions?

By addressing these questions, the project aims to deepen understanding of how **temporal order** and **syntactic information alone** influence learning in LLMs.

## Methodology

### Datasets and POS Tagging

We use two major English text corpora for experimentation: **OpenWebText** and **CC100 (English subset)**. These provide a diverse range of natural text for analysis:

- **OpenWebText (OWT):** An open reproduction of the WebText dataset (\~10GB of internet text). We sample a subset (on the order of 1–2 GB) of this corpus for training. The data is loaded via Hugging Face _datasets_ in streaming mode for efficiency.
- **CC100 (English):** A large Common Crawl-based dataset with multilingual text. We use the English portion (CC100-En). A local `.xz` compressed text file is streamed and decompressed on the fly to obtain \~1–2 GB of raw English text.

**POS Tagging:** All raw text is converted into sequences of `(word, POS tag)` using **spaCy** (English model `en_core_web_sm`). Only coarse grammatical tags are retained (e.g., `NN` for noun, `VBZ` for verb in 3rd person singular, etc.), and punctuation and special tokens are also tagged. This process strips away specific word meanings and retains only syntactic roles in sequence. The output of this stage is a long sequence of POS tags corresponding to the input text.

### Tokenization Schemes

After tagging, we apply one of two tokenization schemes to map the POS tags into integer token IDs suitable for model training:

1. **Unique POS ID Mapping (Tag-Level Vocabulary):** Each distinct POS tag is mapped to a single unique ID (a one-to-one mapping). For example, `NN` might always map to token ID 5, `VBZ` to 17, `JJ` to 9, etc. This yields a **small, fixed vocabulary** approximately equal to the number of unique POS tags (on the order of tens of tokens, e.g. 40–50 for English). In this scheme, the model sees an **abstracted sequence of POS categories** (one token per tag) with no variation in token IDs for the same tag. This represents the pure syntactic structure of the text.

2. **Random ID within Tag-Specific Ranges:** Each POS tag is assigned a **range of possible token IDs**, and every occurrence of that tag in the sequence is randomly assigned one of the IDs from that range. For example, determiners (tag `DT`) might be assigned the ID range 1000–1100, nouns (`NN`) 1100–1300, verbs (`VB`) 1300–1400, etc. If a noun appears in the text, it could be encoded as any ID between 1100 and 1299 (chosen uniformly at random _per occurrence_). The exact range sizes are proportional to the number of unique words observed for each tag in the corpus (with an upper cap for proper nouns to limit vocabulary size). All tag-specific ranges together form a large combined vocabulary (capped at 65,535 tokens to fit in 16-bit IDs). In this scheme, **lexical identity is completely obfuscated** – the model cannot tell if two tokens are the same word, only that they are of the same POS category (due to the ID’s range). The random assignment forces the model to learn patterns based solely on the _distribution of categories_ rather than memorizing specific token identities.

**Rationale:** The first scheme (unique ID per tag) provides an idealized scenario where the model sees a deterministic POS sequence. The second scheme injects maximal uncertainty at the token level (the model knows the category of each token by its range, but not a stable identity), thus further isolating syntactic structure by preventing the model from associating any meaning or even a consistent symbol with a particular word or lemma. By comparing these, we can assess how token variability affects the learning of syntax and whether the model essentially learns an abstract grammar.

### Model and Training

We train **Transformer language models** on the POS-tokenized data using a _modified [nanoGPT](https://github.com/karpathy/nanoGPT) implementation_. Key aspects of the model and training setup include:

- **Model Architecture:** A GPT-style decoder-only transformer. We experiment with several model sizes:

  - **“Nano” model:** Extremely small (e.g. 3 layers, 48-dimensional embeddings, 3 attention heads) – useful for quick tests.
  - **“Mini” and “Small” models:** Intermediate sizes (e.g. 6-layer or 8-layer models with a few hundred embedding dimensions).
  - **GPT-1 scale model:** \~12 layers, 768-dimensional embeddings, 12 attention heads (roughly 110M parameters, comparable to the original GPT-1 from OpenAI). This is the largest model we train to ensure sufficient capacity to learn the POS sequences.

  All models use **ReLU-based feedforward networks** (actually, in this implementation the feedforward dimension is 4× the embedding size, and activation is typically GELU by default in GPT; dropout is set to 0 for these pretraining runs).

- **Sequence Length:** We use a relatively short context window (`block_size`, e.g. 64 tokens) for training. Even though the datasets are large, using a smaller context focuses the model on learning local syntactic structures and saves memory. A sliding window (with stride, e.g. 32) is used to cover the corpus.

- **Forward vs. Backward Training:** For each experiment, we train two versions of the model:

  - **Forward Model:** Sequences are fed in natural reading order. We prepend a begin-of-sequence token (BOS) at the start of each training chunk. The model is trained to predict the next POS token in forward direction.
  - **Backward Model:** Sequences are reversed. We append a BOS token at the _end_ of each chunk (effectively acting as an end-of-sequence marker before reversal), then reverse the order. The model is thus trained to predict previous tokens (when viewed in original order). In practice, this means the backward model processes text from the end to the beginning, learning to predict the token that came _before_. We use the same model architecture and hyperparameters; only the data order and the `backwards` flag differ. This allows a direct performance comparison to identify any directional discrepancies attributable to the arrow of time.

- **Learning Schedule (Constant + Cooldown):** We employ a learning rate schedule with three phases: **linear warmup**, **constant rate**, and **cooldown decay**. Initially, the learning rate is linearly ramped up from 0 to the target rate over a small number of iterations (e.g. a few hundred warmup steps) to avoid instability. Then it is held constant for the majority of training. In the final portion of training (e.g. last 10% of steps), a **cooldown** phase linearly decays the learning rate to a minimum (often down to 0). This **constant-plus-cooldown** schedule helps ensure the model converges smoothly at the end of training without sudden shocks in learning rate. The fraction of training used for cooldown and the decay function (linear or 1−√t) are configurable (we use linear cooldown by default). We found this schedule to be effective for stable training, especially since we typically train for only 1 epoch over the data (a limited number of iterations given the dataset size) and want to maximize progress early on while still finishing with a fine convergence.

- **Optimization and Regularization:** We use the AdamW optimizer (β₁=0.9, β₂≈0.99) with weight decay on non-bias parameters. Gradient clipping is applied to prevent exploding gradients (e.g. clip at global norm 1.0). In these pretraining runs we did not use dropout (dropout=0.0) to focus on fitting the data; since the data is synthetically limited in entropy (especially in the unique-ID case), overfitting was not a major concern for our analysis.

- **Logging and Evaluation:** The training loop evaluates the model on a held-out validation set periodically (e.g. every 2000 iterations) and logs training/validation loss. We integrate with Weights & Biases (wandb) for experiment tracking (optional; can be disabled) to record metrics. Checkpoints are saved to the `out/` directory for each run. All runs use the same train/val split of the tokenized data (e.g. 95% train, 5% validation). The primary evaluation metric is **cross-entropy loss** on the validation POS sequence (and corresponding **perplexity**), which reflects how well the model has learned to predict the syntactic sequence.

By training models under these conditions and comparing forward vs. backward and different tokenizations, we isolate the effects of temporal order on syntactic learning in LLMs.

## Technologies and Frameworks Used

This project leverages a range of technologies in the machine learning and NLP ecosystem:

- **Python 3.x** – Primary programming language for all experiments.
- **PyTorch** – Deep learning framework used for implementing the Transformer model and training loop (the code is built upon the lightweight nanoGPT implementation).
- **spaCy** – NLP library used for tokenizing text and extracting part-of-speech tags (`en_core_web_sm` model for English POS tagging).
- **Hugging Face Datasets** – Used to stream the OpenWebText dataset (`Skylion007/openwebtext`) conveniently. This allows us to load large corpora without storing everything in memory at once.
- **lzma (Python)** – Used to stream-decompress the CC100 dataset from an `.xz` file on the fly.
- **tqdm** – For progress bars during data processing (e.g., reading and tagging the corpus).
- **NumPy** – For efficient array manipulations, especially when preparing the dataset and converting to numpy arrays or memmaps.
- **Pickle** – Used for serializing metadata (such as vocabulary dictionaries and configuration) for later reuse.
- **Weights & Biases (wandb)** – Integrated for experiment logging and visualization of training/validation curves (optional, can be enabled in config).
- **nanoGPT codebase** – The repository is structured as an extension of nanoGPT (by A. Karpathy), modified to handle custom data and new training schedule. This provides utilities for model definition, training loop, and text generation, adapted here for POS tag sequences.

All experiments were run on an environment with these libraries installed, and training was accelerated using GPU hardware via PyTorch (CUDA). The code is platform-independent aside from the need for a Python environment with the above dependencies.

## Repository Structure

The repository is organized to separate data preparation, configuration, and core code. Key files and directories include:

- **`data/`** – Contains subdirectories for datasets and data preparation scripts:

  - `openwebtext/` – Scripts for processing the OpenWebText corpus. For example, `data/openwebtext/tokenizer.py` downloads a subset of OpenWebText and converts it to POS-tag token sequences (using the **random ID ranges** scheme by default). Running this script produces `train.bin`, `val.bin` (binary files of token IDs), and `meta.pkl` (metadata including the tokenizer dictionary and vocabulary size) in the `data/openwebtext` folder.
  - `openwebtext_id/` (if used) – Intended for the **unique ID mapping** variant of OpenWebText. (This may use a similar procedure to generate data where each POS tag corresponds to one ID. In practice, one can adapt the CC100 ID script to OpenWebText to create this dataset.)
  - `cc100/` – Scripts for processing the CC100 English corpus with the **random ID** scheme. For example, `data/cc100/tokenizer.py` streams a local CC100 `.xz` file, tags it with POS, assigns random IDs per tag range, and outputs `train.bin`, `val.bin`, and `meta.pkl` in `data/cc100`.
  - `cc100_id/` – Scripts for the **unique POS ID** scheme on CC100. Notably, `data/cc100_id/tokenizer_id.py` will tag the text and assign one unique ID per POS tag (creating a very small vocabulary). It outputs data files in `data/cc100_id`.
  - Other subdirectories like `cc100_finnish/`, `cc100_german/`, etc., may be present for experimental extension to other languages (e.g., testing the approach on Finnish or German text), though the core project analysis focuses on English.
  - Each tokenizer script is self-contained and can be run independently to prepare that dataset. They also print statistics (like vocabulary size and tag distributions) for insight.

- **`config/`** – Contains configuration files for training runs. These are Python scripts that set hyperparameters and options, which the main training script can load. The configs are organized into subfolders:

  - `id_model/` – Configs for the **unique POS ID mapping** experiments.

    - e.g. `config/id_model/openwebtext/train_id_gpt1.py` – config for training a GPT-1 sized model on the OpenWebText dataset with unique-ID scheme.
    - `config/id_model/cc100/train_id_small.py` – config for a smaller model on CC100 with unique-ID scheme.
    - These configs set parameters like `dataset` name (matching a folder in `data/`), model size (`n_layer, n_head, n_embd`), training length (`max_iters or num_epochs`), learning rate, `backwards = False/True`, etc.

  - `range_model/` – Configs for the **random ID range** scheme experiments.

    - e.g. `config/range_model/openwebtext/train_range_gpt1.py` for OpenWebText with random IDs, GPT-1 size.
    - `config/range_model/cc100/train_range_nano.py` for a nano model on CC100 with random IDs.

  - Each config file is named to indicate dataset and model scale (nano, mini, small, gpt1). They can be used to easily reproduce specific experiments by passing the file to the training script.

- **`model.py`** – Defines the Transformer (GPT) model and configuration class. This is adapted from nanoGPT, modified to accept our `vocab_size` (which corresponds to the number of POS token IDs plus the BOS token) and other configurations. It includes the GPT architecture (multi-layer self-attention, feedforward, etc.) implementation in PyTorch.

- **`dataloader.py`** – Defines the `POSDataset` class, a PyTorch `Dataset` for our tokenized POS data. It uses memory-mapped `.bin` files for efficiency. This class handles slicing the data into sequences of length `block_size` with a given `stride`. Crucially, it implements the logic for forward vs. backward sequence output:

  - In forward mode, it prepends the BOS token at the start of each sequence and shifts the sequence to produce input-target pairs.
  - In backward mode, it appends BOS, reverses the sequence, then likewise produces input-target pairs (so the model learns to predict the previous token in original order).
    The dataset ensures that the BOS token ID is `vocab_size` (i.e., one above the highest actual token id from the data, as stored in `meta.pkl`). This dataset is used by the PyTorch DataLoader to feed training batches.

- **`train.py`** – The main training script. It reads a configuration (from a config file or command-line overrides), loads the dataset, initializes the model, and runs the training loop with evaluation. It supports running in distributed mode (PyTorch DDP) for multi-GPU training if needed. Key features:

  - Uses the config to determine which dataset to load (it looks for `data/<dataset>/train.bin` etc.), model hyperparameters, and training settings.
  - Incorporates the learning rate scheduler (warmup, constant, cooldown) as described.
  - Logs progress to the console and optionally to wandb.
  - Saves model checkpoints to the specified `out_dir`.

- **`generate.py`** – An inference script to generate sequences from a trained model. You can use this to sample new POS tag sequences from the model, either forward or backward. It loads a saved checkpoint, then generates a sequence of length `max_new_tokens` given a prompt (which can be just the BOS token to start fresh, or a specific tag sequence as context). This script also includes logic to **map generated POS tags back to actual words** using a provided dictionary (`pos_dictionary.json`). The dictionary contains a list of example words for each POS tag (for example, for `NN` it might list \["time","person","year",…,"car","idea",…], and for `VBZ` \["is","has","goes",…], etc.). Using this, the script can construct a pseudo-sentence from the sampled POS sequence by replacing each tag with a random example word of that category. This is useful for interpreting the model’s output in readable form (though the words are generic placeholders, not actual predicted words). The `generate.py` script handles punctuation spacing and capitalization rules when reconstructing text from POS to ensure the output reads properly.
  _Usage example:_ After training a model, you might run:

  ```bash
  python generate.py --out_dir=out/openwebtext_gpt1 \
      --start "<BOS>" --num_samples=3 --max_new_tokens=50
  ```

  This would load the GPT-1 model trained on OpenWebText (forward, random-ID scheme by default) and generate 3 sequences of 50 new tokens (POS tags), then print them along with a possible English realization.

- **`predict_loss.py`** – A specialized analysis script to compare the **theoretical expected loss** between the unique-ID model and the random-ID model. This script demonstrates how to use the small-vocabulary tag-only model to predict a lower bound on the larger model’s loss. It loads the tokenizer metadata of the “ID model” (unique tag IDs) and the “Bigger model” (random IDs), and computes an estimate of what loss the bigger model should achieve if it perfectly learns the POS structure but still has to random-guess the token within each tag’s range. This is done by calculating the entropy contribution of each tag’s ID range. Essentially, if the tag-only model has a cross-entropy of _L_id_ (in nats) on the POS sequence, and each tag spans N possible IDs in the random scheme, the best possible cross-entropy for the random-ID model would be _L_id + E\[log N]_ (the additional uncertainty from having to predict a random ID for the correct tag). The script uses the `meta.pkl` info from both models to estimate this and compare to the observed validation loss. This helps verify that the random-ID model’s performance aligns with expectations (and thus it is indeed learning the POS patterns and only the random assignment is adding extra loss).
  _(This is an optional analysis tool; it’s not needed for basic training or generation.)_

- **`pos_dictionary.json`** – A JSON file containing example words for each POS tag (used by `generate.py` as described above). This is a static resource to help interpret model outputs.

Other files include standard Python utilities and initialization (e.g., `configurator.py` which helps parse config files in `train.py`, and possibly a `README.md` if provided). The above are the most important components for understanding and using the project.

## Setup and Installation

Follow these steps to set up the project environment and prepare the data:

1. **Clone the Repository:** If you haven’t already, clone this GitHub repository to your local machine.

2. **Create Environment:** It’s recommended to create a Python virtual environment for the project (optional but best practice):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Linux/Mac
   venv\Scripts\activate      # on Windows
   ```

3. **Install Dependencies:** Install the required Python packages. You can use the provided `requirements.txt` if available, or install manually:

   ```bash
   pip install torch tqdm spacy datasets wandb numpy
   ```

   _(Ensure you have a PyTorch version compatible with your CUDA if using a GPU. You may install PyTorch via pip or follow instructions from the [official site](https://pytorch.org) for your CUDA version.)_
   Additionally, install the spaCy English model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

   This downloads the language model needed for POS tagging.

4. **Prepare Datasets:** Next, obtain and preprocess the data into the required format. Depending on which experiments you want to run:

   - **OpenWebText:** The tokenizer script can download the data automatically. Run:

     ```bash
     python data/openwebtext/tokenizer.py
     ```

     This will stream \~2GB of OpenWebText text, perform POS tagging, and output the files `data/openwebtext/train.bin`, `val.bin`, and `meta.pkl`. (Ensure you have an internet connection for HuggingFace Datasets to fetch OpenWebText. This might take some time due to data size and POS tagging on CPU. The script uses multiple processes to speed up tagging.)

   - **CC100 (English):** Download an English CC100 dataset file. You can obtain an English text dump from the [CC100 dataset on Hugging Face](https://huggingface.co/datasets/cc100) or other sources. For example, the file might be named like `en.txt.xz` (an XZ-compressed text file of several GB). Update the path in `data/cc100/tokenizer.py` (and `tokenizer_id.py` if using) to point to the location of your CC100 English `.xz` file:

     ```python
     data = load_cc100_subset("/path/to/en.txt.xz", target_size_gb=2)
     ```

     By default, the script is set to `/home/larosa/en.txt.xz` as a placeholder – replace this with your path. Then run:

     ```bash
     python data/cc100/tokenizer.py
     ```

     This will read the CC100 file, stop after \~2GB of uncompressed text, tag it and produce `data/cc100/train.bin`, `val.bin`, `meta.pkl`. For the unique-ID version on CC100, run:

     ```bash
     python data/cc100_id/tokenizer_id.py
     ```

     which similarly processes the text but assigns one ID per tag. It will output to `data/cc100_id/`. Make sure to update the file path inside this script as well.

   - **Other Data (Optional):** If exploring other languages or datasets (e.g., the Shakespeare example or other CC100 languages), follow similar steps: ensure you have the raw text, then run the corresponding tokenizer script in `data/<dataset>/`.

   Each tokenizer script will print out the vocabulary size and some examples. The resulting `.bin` files are memory-mapped 16-bit token sequences, and `meta.pkl` contains the mapping (POS tag ranges or IDs) and `vocab_size`. **Note:** The BOS token is not included in `vocab_size` in `meta.pkl`; the code will add it dynamically (effectively, the BOS token ID = `vocab_size` from meta).

5. **Verify Data Preparation:** After running the scripts, you should have directories like `data/openwebtext` (and/or `data/cc100`, etc.) each containing `train.bin`, `val.bin`, and `meta.pkl`. The sizes of these files will correspond to the number of tokens (for example, if \~2e9 bytes of text were processed, you might have on the order of 300 million POS tokens, since POS tagging roughly yields one token per word/punctuation).

## How to Run Training and Evaluation

With data prepared and environment set up, you can train the models and evaluate their performance as follows:

- **Launching Training:** Use `train.py` with a configuration file to start training. For example, to train a forward-direction model on OpenWebText with the random-ID (range) scheme at GPT-1 size:

  ```bash
  python train.py config/range_model/openwebtext/train_range_gpt1.py
  ```

  This will load the config, initialize the model and data loader, and begin training. Logs will be printed to stdout showing training loss and validation loss at intervals. The model checkpoints and logs will be saved under the `out/` directory specified in the config (e.g., `out/openwebtext_gpt1`).

  If you want to run a **backward (reversed) model** training, you have two options:

  - Edit the config file: open the config and set `backwards = True`, save it, then run as above. (You might also change the `wandb_run_name` or output directory to distinguish it as a backward model run.)
  - **OR** override via command line:

    ```bash
    python train.py config/range_model/openwebtext/train_range_gpt1.py --backwards=True --wandb_run_name=openwebtext_gpt1_backward
    ```

    The `configurator.py` will apply the `--backwards=True` override to the loaded config. This way you don’t have to create a separate file. _(Note: ensure the run name or output dir is different to avoid overwriting the forward model.)_

  Similarly, to run the **unique-ID tag model** on the same data:

  ```bash
  python train.py config/id_model/openwebtext/train_id_gpt1.py
  ```

  (Again, set `backwards=True` in the config or via CLI for the backward version.)

  You can experiment with smaller models for quicker results:

  ```bash
  python train.py config/range_model/openwebtext/train_range_nano.py
  ```

  This uses a tiny model (3 layers) that should train very fast, useful to sanity-check the pipeline. The validation loss will be higher for such a small model, but it will confirm everything is working.

- **During Training:** Monitor the console (or wandb dashboard if enabled) for logs. You will see lines printing training iteration, training loss, and every `eval_interval` steps a validation loss. For example:

  ```
  iter 1000: train loss 2.45, val loss 2.38
  iter 2000: train loss 2.30, val loss 2.28
  ...
  ```

  This indicates the cross-entropy (likely in nats by default) on the POS sequence. A lower loss means better predictive performance. You can stop training early if needed (the script will save a checkpoint at the last evaluation). The config as given might only run 1 epoch (for instance, 2000 iterations with a certain batch size/stride might cover the data once). Ensure that the `max_iters` or `num_epochs` in the config is set as you intend.

- **Evaluation/Results:** After training, the final **validation loss** gives an indication of model performance. You can convert loss to perplexity for interpretability: `perplexity = exp(loss_in_nats)` (if log base e was used) or `2^(loss_in_bits)` if log base 2. For example, a loss of 2.0 nats corresponds to perplexity \~7.39 (since e^2 ≈ 7.39). Compare the forward vs backward runs:

  - If the forward model yields lower perplexity than the backward model, it suggests the model benefits from the natural forward order – evidence of an "arrow of time" effect. We expect forward models to have a slight advantage since languages are not symmetric in time (forward context tends to be more predictive for syntactic structures like determiner-noun agreement, etc., whereas backward order might confuse some dependencies).
  - Check also the difference between tokenization schemes: The unique-ID (tag-only) model should achieve much lower loss (since the task is easier – fewer options to predict). The random-ID model will have higher loss because even a perfect syntactic predictor still faces uncertainty in guessing the random token ID. In fact, using the `predict_loss.py` script, you can calculate the expected gap. For instance, if the tag-only model’s perplexity is P_tag, and on average each tag has M possible IDs in the random scheme, the random-ID model’s perplexity cannot go below roughly P_tag \* M (this is a rough intuition; the script does a more precise calculation).

  All evaluation in this project is intrinsic (on the POS prediction task). We do not evaluate on generating real text since the model is not trained on actual word sequences. However, the generate script can be used for qualitative inspection of learned structures.

- **Generating Examples:** To further inspect what the model has learned, you can generate POS tag sequences and map them to example sentences:

  ```bash
  python generate.py --out_dir=out/openwebtext_gpt1 --start "<BOS>" --max_new_tokens=100
  ```

  Ensure that `pos_dictionary.json` is present in the working directory. The script will output something like:

  ```
  <BOS> DT NN VBZ DT JJ NN .
  e.g., "The dog eats the red ball."
  ```

  (The second line is the POS sequence mapped to an example sentence.) This can give an intuitive sense of whether the model’s generated POS sequences look fluent or grammatical. You could also prompt the model by providing a custom POS sequence as `--start "DT JJ"` etc., to see how it continues.

## Expected Outputs and Results

**Conclusion:**

## Usage Examples and Tips

- To experiment quickly, start with a small model (nano or mini config) on a smaller subset of data. You can reduce `target_size_gb` in the tokenizer script (e.g., 0.5 GB instead of 2 GB) and train a nano model to see results in minutes.
- If you want to test the **backward model**, remember to set the `backwards` flag. You can run forward and backward in parallel (with different output directories) to compare outcomes.
- Utilize Weights & Biases logging by setting `wandb_log = True` and providing your W\&B API key (if required). This will help visualize the training curves and compare runs easily.
- When generating from the model, note that the output is a sequence of POS tags. The provided `pos_dictionary.json` is just for illustration; the quality of generated _sentences_ from it is not the focus (we only care that the POS sequence is plausible).
- The **Shakespeare example** (if provided in `config/shakespeare/`) shows how to apply the same POS-sequence modeling to a different style of text (Shakespeare’s works). This can be a fun extension to see if the model learns old-style syntax or just general patterns.
- Training the GPT-1 sized model on the full data may require a GPU with \~8-12GB memory. If you run out of memory, lower the `batch_size` or use gradient accumulation (increase `gradient_accumulation_steps`) to effectively batch across iterations.
- All results in this project are based on **synthetic tasks** (POS prediction), so they don’t directly translate to typical language modeling of words. However, the approach and findings can spark ideas for research into separating syntax/semantics, or investigating why unidirectional models (like GPT) might inherently be better at language generation than if we tried to generate backwards.
