# train_small.py
out_dir = "out/cc100_small"
eval_interval = 2000
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

# Logging
wandb_log = True
wandb_project = "semester-project-all"
wandb_run_name = "cc100_small"

# Dataset
dataset = "cc100"
# We are forcing the same effective batch size as GPT1:
batch_size = 128
gradient_accumulation_steps = 1
block_size = 64
stride = 32

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
warmup_iters = 300

# Use epochs
train_on_epochs = True  # whether to train for a number of epochs instead of max_iters
num_epochs = 1  # number of epochs to train for, if train_on_epochs is True

# Learning rate scheduler
decay_lr = True
cooldown_fraction = 0.1
cooldown_type = "linear"

backwards = False

# Device settings
device = "cuda"
compile = False
