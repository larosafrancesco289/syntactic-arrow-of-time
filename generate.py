import os
import pickle
from contextlib import nullcontext
import torch
import random
import json
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration & File Paths
# -----------------------------------------------------------------------------
init_from = "resume"
out_dir = "out"
start = "<BOS>"
num_samples = 5
max_new_tokens = 100
temperature = 0.8
top_k = 40
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16" if torch.cuda.is_available() else "float32"
)
compile = False

# --- Location of the POS dictionary JSON file ---
# Assumes it's in the same directory as this script. Adjust if needed.
script_dir = os.path.dirname(__file__)
dictionary_file = os.path.join(script_dir, "pos_dictionary.json")
# -----------------------------------------------------------------------------

exec(open("configurator.py").read())  # Overrides from command line or config file

# --- Load the Hardcoded POS Dictionary from JSON ---
try:
    with open(dictionary_file, "r", encoding="utf-8") as f:
        hardcoded_pos_to_words = json.load(f)
    print(f"Successfully loaded POS dictionary from {dictionary_file}")
except FileNotFoundError:
    print(f"Error: POS dictionary file not found at {dictionary_file}")
    print("Please ensure 'pos_dictionary.json' exists.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Could not parse POS dictionary file {dictionary_file}")
    print(f"JSON Error: {e}")
    exit(1)
# -----------------------------------------------------------------------------

# --- Define Tags for Specific Handling (Derived from dictionary keys/structure) ---
symbol_tags = {
    ".",
    ",",
    ":",
    ";",
    "?",
    "!",
    "``",
    "''",
    "-LRB-",
    "-RRB-",
    "HYPH",
    "#",
    "$",
    "SYM",
    "POS",
    "NFP",
}
no_space_before_tags = {".", ",", ":", ";", "?", "!", "''", "-RRB-", "POS"}
no_space_after_tags = {"``", "-LRB-", "$", "#"}
proper_noun_tags = {"NNP", "NNPS"}
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# --- Model Loading ---
ckpt_path = os.path.join(out_dir, "ckpt.pt")
if not os.path.exists(ckpt_path):
    # Try finding checkpoint relative to script dir if not in default out_dir
    alt_ckpt_path = os.path.join(script_dir, out_dir, "ckpt.pt")
    if not os.path.exists(alt_ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at {ckpt_path} or {alt_ckpt_path}"
        )
    ckpt_path = alt_ckpt_path

print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
required_ckpt_keys = ["model_args", "model", "config"]
if not all(key in checkpoint for key in required_ckpt_keys):
    missing = [key for key in required_ckpt_keys if key not in checkpoint]
    raise ValueError(f"Checkpoint is missing required keys: {missing}")

gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model)
print(f"Model loaded successfully.")

# --- Metadata Loading (Tokenizer ONLY) ---
config = checkpoint["config"]
dataset_name = config.get("dataset")
if not dataset_name:
    raise ValueError("Could not find 'dataset' key in checkpoint config.")

# Construct potential paths for meta.pkl
meta_path_options = [
    os.path.join("data", dataset_name, "meta.pkl"),  # Standard path
    os.path.join(script_dir, "data", dataset_name, "meta.pkl"),  # Relative to script
]
meta_path = None
for path in meta_path_options:
    if os.path.exists(path):
        meta_path = path
        break

if meta_path is None:
    raise FileNotFoundError(
        f"Cannot find meta.pkl for dataset '{dataset_name}' in expected locations: {meta_path_options}"
    )

print(f"Loading metadata (tokenizer) from {meta_path}")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

if "tokenizer_dict" not in meta or "vocab_size" not in meta:
    raise ValueError(
        f"meta.pkl at {meta_path} is missing 'tokenizer_dict' or 'vocab_size'."
    )

tokenizer_dict = meta["tokenizer_dict"]
meta_vocab_size = meta["vocab_size"]
bos_token_id = meta_vocab_size

itos = {i: s for s, i in tokenizer_dict.items()}
print(f"POS Tag Vocabulary size (from meta.pkl): {meta_vocab_size}")
print(f"BOS token ID: {bos_token_id}")

# Verify hardcoded dict coverage (using the loaded dictionary)
missing_keys = []
for i, tag in itos.items():
    # Check against the keys of the dictionary loaded from JSON
    if tag not in hardcoded_pos_to_words:
        missing_keys.append(tag)
if missing_keys:
    print(
        f"\nWarning: The following POS tags from meta.pkl are missing in the dictionary loaded from {dictionary_file}:"
    )
    print(f"  {missing_keys}")
    print(f"  These tags will be output as [UNKNOWN_TAG:...] if generated.")

# --- Generation ---
# (The rest of the generation logic remains exactly the same as before)
# ... (encoding start sequence, generation loop, decoding, substitution, spacing, capitalization) ...

if start == "<BOS>":
    start_ids = [bos_token_id]
    print("Starting generation with BOS token.")
else:
    try:
        start_tags = start.split()
        start_ids = []
        unknown_start_tags = []
        for tag in start_tags:
            if tag in tokenizer_dict:
                start_ids.append(tokenizer_dict[tag])
            else:
                unknown_start_tags.append(tag)
        if unknown_start_tags:
            print(
                f"Warning: Could not encode the following start tags (not in tokenizer): {unknown_start_tags}"
            )
        if not start_ids:
            print(
                "No valid start tags provided or encoded. Falling back to BOS token start."
            )
            start_ids = [bos_token_id]
        else:
            encoded_tags = [itos.get(i, "?") for i in start_ids]
            print(
                f"Starting generation with sequence: {' '.join(encoded_tags)} -> {start_ids}"
            )

    except Exception as e:
        print(f"Error encoding start string '{start}': {e}. Falling back to BOS token.")
        start_ids = [bos_token_id]

start_tensor = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print(
    f"\nGenerating {num_samples} samples, each with max {max_new_tokens} new tokens..."
)
print("-" * 80)
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(f"--- Sample {k+1}/{num_samples} ---")
            y = model.generate(
                start_tensor, max_new_tokens, temperature=temperature, top_k=top_k
            )

            generated_ids = y[0].tolist()
            generated_pos_tags = [
                itos.get(idx, f"[UNK_ID:{idx}]")
                for idx in generated_ids[len(start_ids) :]
            ]

            print(f"Generated POS Tags:\n{' '.join(generated_pos_tags)}\n")

            # Substitute tags with improved spacing and capitalization
            output_pieces = []
            previous_tag = None  # Keep track of the previous tag for spacing logic
            first_word_done = False

            for i, tag in enumerate(generated_pos_tags):
                word = f"[{tag}_NO_DICT]"  # Default if tag not in hardcoded dict

                if tag.startswith("[UNK_ID"):
                    word = tag  # Keep the unknown ID marker
                # Use the dictionary loaded from JSON
                elif tag in hardcoded_pos_to_words:
                    word_options = hardcoded_pos_to_words[tag]
                    if word_options:
                        # Use direct mapping for symbols/punctuation
                        if tag in symbol_tags:
                            word = word_options[0]
                        else:
                            # Randomly choose a word for other tags
                            word = random.choice(word_options)
                    else:
                        word = f"[{tag}_EMPTY_LIST]"  # Placeholder if list is empty

                # --- Capitalization ---
                if tag in proper_noun_tags and word:
                    word = word.capitalize()
                elif not first_word_done and word and not tag.startswith("["):
                    word = word.capitalize()

                if word and not tag.startswith("["):
                    first_word_done = True

                # --- Spacing ---
                needs_space_before = True
                if i == 0:
                    needs_space_before = False
                elif tag in no_space_before_tags:
                    needs_space_before = False
                elif previous_tag in no_space_after_tags:
                    needs_space_before = False

                if needs_space_before:
                    output_pieces.append(" ")

                output_pieces.append(word)
                previous_tag = tag

            final_sentence = "".join(output_pieces)

            print(f"Substituted Sentence:\n{final_sentence}\n")
            print("-" * 80)
