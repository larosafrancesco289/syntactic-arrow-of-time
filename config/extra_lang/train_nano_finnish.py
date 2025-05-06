# train_gpt1.py
out_dir = "out/cc100_nano/finnish"
eval_interval = 2000
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

# Logging
wandb_log = True
wandb_project = "semester-project-all"
wandb_run_name = "cc100_nano_finnish"

# Dataset
dataset = "cc100"
# GPT1 is our limiting model. Suppose we fit batch_size=64, then do accumulation:
batch_size = 128
gradient_accumulation_steps = 1
block_size = 64
stride = 32

# Model hyperparameters (nano):
n_layer = 3
n_head = 3
n_embd = 48
dropout = 0.0
# feedforward dimension = 4 * n_embd

# Optimizer hyperparameters
learning_rate = 1e-4
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
