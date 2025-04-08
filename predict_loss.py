import os
import pickle
import numpy as np
import math
import torch  # Needed for POSDataset if used
from tqdm import tqdm
from dataloader import POSDataset

# --- Configuration ---

# --- Paths related to the ID Model ---
ID_MODEL_DATA_DIR = "data/openwebtext_id"  # Directory where ID model's data (val.bin, meta.pkl) is stored
ID_MODEL_META_PATH = os.path.join(ID_MODEL_DATA_DIR, "meta.pkl")
ID_MODEL_VAL_BIN_PATH = os.path.join(ID_MODEL_DATA_DIR, "val.bin")

# --- Parameters used for the ID Model's POSDataset ---
# You MUST use the same parameters as when the ID model was trained/validated
ID_MODEL_BLOCK_SIZE = 64
ID_MODEL_STRIDE = 32
ID_MODEL_BACKWARDS = False

# --- Validation Loss of the *trained* ID Model ---
# Replace this with the actual final validation loss you recorded
L_id_observed = 2.035  # <<<--- IMPORTANT: SET THIS VALUE

# --- Paths related to the Bigger Model ---
BIGGER_MODEL_DATA_DIR = (
    "data/openwebtext"  # Directory where Bigger model's data (meta.pkl) is stored
)
BIGGER_MODEL_META_PATH = os.path.join(BIGGER_MODEL_DATA_DIR, "meta.pkl")

# --- Main Calculation ---


def calculate_estimated_loss():
    """
    Calculates the estimated cross-entropy for the Bigger Model.
    """
    print("--- Starting Estimation Calculation ---")

    # 1. Load ID Model Metadata
    print(f"Loading ID Model metadata from: {ID_MODEL_META_PATH}")
    if not os.path.exists(ID_MODEL_META_PATH):
        raise FileNotFoundError(
            f"ID Model metadata file not found: {ID_MODEL_META_PATH}"
        )
    with open(ID_MODEL_META_PATH, "rb") as f:
        meta_id = pickle.load(f)
    tokenizer_dict_id = meta_id.get("tokenizer_dict")
    id_model_vocab_size = meta_id.get("vocab_size")
    if tokenizer_dict_id is None or id_model_vocab_size is None:
        raise ValueError(
            f"Metadata file {ID_MODEL_META_PATH} must contain 'tokenizer_dict' and 'vocab_size'"
        )
    print(f"ID Model vocab size (excluding BOS): {id_model_vocab_size}")
    print(f"Found {len(tokenizer_dict_id)} POS tags in ID tokenizer.")

    # 2. Load Bigger Model Metadata
    print(f"Loading Bigger Model metadata from: {BIGGER_MODEL_META_PATH}")
    if not os.path.exists(BIGGER_MODEL_META_PATH):
        raise FileNotFoundError(
            f"Bigger Model metadata file not found: {BIGGER_MODEL_META_PATH}"
        )
    with open(BIGGER_MODEL_META_PATH, "rb") as f:
        meta_big = pickle.load(f)
    tokenizer_dict_big = meta_big.get("tokenizer_dict")
    if tokenizer_dict_big is None:
        raise ValueError(
            f"Metadata file {BIGGER_MODEL_META_PATH} must contain 'tokenizer_dict'"
        )
    print(f"Found {len(tokenizer_dict_big)} POS tags in Bigger tokenizer.")

    # 3. Invert ID Tokenizer
    print("Inverting ID tokenizer mapping (ID -> POS tag string)...")
    try:
        id_to_pos_tag = {v: k for k, v in tokenizer_dict_id.items()}
    except Exception as e:
        print(f"Error inverting ID tokenizer: {e}")
        print("Ensure tokenizer_dict_id contains unique integer values.")
        return

    # 4. Calculate Range Sizes for Bigger Model
    print("Calculating range sizes for Bigger Model POS tags...")
    pos_tag_range_sizes = {}
    for pos_tag, range_tuple in tokenizer_dict_big.items():
        try:
            start, end = range_tuple
            size = end - start
            if size < 0:
                print(
                    f"Warning: Invalid range for tag '{pos_tag}': {range_tuple}. Size is negative. Skipping."
                )
                continue
            pos_tag_range_sizes[pos_tag] = size
        except Exception as e:
            print(
                f"Warning: Could not process range for tag '{pos_tag}': {range_tuple}. Error: {e}. Skipping."
            )
    print(f"Calculated sizes for {len(pos_tag_range_sizes)} POS tags.")
    if not pos_tag_range_sizes:
        print(
            "Error: Could not determine any valid range sizes from the Bigger Model tokenizer."
        )
        return

    # 5. Get ID Model Validation Target Sequence
    print(f"Loading ID Model validation target sequence using POSDataset...")
    print(f"  Data file: {ID_MODEL_VAL_BIN_PATH}")
    print(f"  Block size: {ID_MODEL_BLOCK_SIZE}")
    print(f"  Vocab size (for BOS): {id_model_vocab_size}")
    print(f"  Backwards: {ID_MODEL_BACKWARDS}")
    print(f"  Stride: {ID_MODEL_STRIDE}")

    if not os.path.exists(ID_MODEL_VAL_BIN_PATH):
        raise FileNotFoundError(
            f"ID Model validation data file not found: {ID_MODEL_VAL_BIN_PATH}"
        )

    try:
        # Instantiate the dataset specific to the ID model's validation phase
        id_val_dataset = POSDataset(
            data_file=ID_MODEL_VAL_BIN_PATH,
            block_size=ID_MODEL_BLOCK_SIZE,
            vocab_size=id_model_vocab_size,  # Pass vocab size used by ID model
            backwards=ID_MODEL_BACKWARDS,
            stride=ID_MODEL_STRIDE,
        )
        # Extract all target tokens (the 'y' part of each item)
        # This might consume memory if the validation set is huge.
        # Alternative: process in batches if memory is an issue.
        print(
            f"Extracting target tokens from {len(id_val_dataset):,} validation chunks..."
        )
        id_validation_target_sequence = []
        # Make sure to handle potential tensor outputs correctly
        for i in tqdm(range(len(id_val_dataset)), desc="Extracting Targets"):
            _, y_tensor = id_val_dataset[i]
            id_validation_target_sequence.extend(
                y_tensor.tolist()
            )  # Add all targets from the chunk

        num_target_tokens = len(id_validation_target_sequence)
        print(f"Extracted {num_target_tokens:,} target tokens.")
        if num_target_tokens == 0:
            print("Error: No target tokens extracted. Check POSDataset logic and data.")
            return

    except Exception as e:
        print(f"Error creating POSDataset or extracting targets: {e}")
        # Print traceback for more details
        import traceback

        traceback.print_exc()
        return

    # 6. Calculate Average Log Range Size
    print(
        "Calculating average log range size over the ID validation target sequence..."
    )
    log_range_sizes = []
    unknown_id_count = 0
    unknown_pos_in_big_count = 0
    zero_size_count = 0

    for target_id in tqdm(id_validation_target_sequence, desc="Calculating Log Sizes"):
        pos_tag = id_to_pos_tag.get(target_id)

        if pos_tag is None:
            # This ID from the validation data doesn't map back to a known POS tag
            # Might happen if target_id is BOS or something unexpected.
            unknown_id_count += 1
            continue  # Skip this token

        size = pos_tag_range_sizes.get(pos_tag)

        if size is None:
            # The POS tag exists in ID model data, but not in Bigger model ranges
            unknown_pos_in_big_count += 1
            continue  # Skip this token

        if size > 0:
            log_range_sizes.append(math.log(size))
        elif size == 0:
            # Range size is zero, log is undefined. This shouldn't happen with end > start.
            zero_size_count += 1
            # Option: append log(1)=0, or skip. Let's skip and warn.
            continue
        # Negative size already handled during range size calculation

    print("Finished calculating log sizes.")
    if unknown_id_count > 0:
        print(
            f"Warning: Skipped {unknown_id_count} target tokens due to unknown ID -> POS mapping."
        )
    if unknown_pos_in_big_count > 0:
        print(
            f"Warning: Skipped {unknown_pos_in_big_count} target tokens because their POS tag ('{pos_tag}' was last example) wasn't found in the Bigger Model's ranges."
        )
    if zero_size_count > 0:
        print(
            f"Warning: Skipped {zero_size_count} target tokens because their POS tag had a range size of 0."
        )

    if not log_range_sizes:
        print(
            "Error: Could not calculate log range size for any target token. Check mappings and data."
        )
        return

    AvgLogRangeSize = np.mean(log_range_sizes)
    print(f"Average Log Range Size (AvgLogRangeSize): {AvgLogRangeSize:.6f}")

    # 7. Estimate Bigger Model Loss
    L_big_estimated = L_id_observed + AvgLogRangeSize

    print("\n--- Estimation Results ---")
    print(f"Observed ID Model Validation Loss (L_id): {L_id_observed:.6f}")
    print(f"Calculated Average Log Range Size:          {AvgLogRangeSize:.6f}")
    print(f"--------------------------------------------------")
    print(f"Estimated Bigger Model Validation Loss:     {L_big_estimated:.6f}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    calculate_estimated_loss()
