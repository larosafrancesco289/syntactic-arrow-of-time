# train_small.py
out_dir = "out-shakespeare_pos_range_small"
eval_interval = 250
eval_iters = 20
log_interval = 1

always_save_checkpoint = False

# Logging
wandb_log = False
wandb_project = "semester-project-gpu"
wandb_run_name = "shakespeare_pos_range_small"

# Dataset
dataset = "shakespeare_pos_range"
# Matching GPT1’s effective batch size:
batch_size = 64
gradient_accumulation_steps = 2
block_size = 64

# Model hyperparameters (small):
n_layer = 10
n_head = 10
n_embd = 380
dropout = 0.0
# feedforward dimension = 4 * n_embd (inside model code)

# Optimizer hyperparameters
learning_rate = 1e-4  # same across models
max_iters = 2000
lr_decay_iters = 2000
min_lr = 0.0
beta2 = 0.99
warmup_iters = 100

# Learning rate scheduler
decay_lr = True
cooldown_fraction = 0.1
cooldown_type = "linear"

backwards = False

# Device settings
device = "cuda"
compile = False
