# train_mini.py
out_dir = "out-shakespeare_pos_range_mini"
eval_interval = 250
eval_iters = 20
log_interval = 1

# We'll still only save when val improves
always_save_checkpoint = False

# Logging
wandb_log = False
wandb_project = "semester-project-gpu"
wandb_run_name = "shakespeare_pos_range_mini"

# Dataset
dataset = "shakespeare_pos_range"
# We are forcing the same effective batch size as GPT1:
batch_size = 64
gradient_accumulation_steps = 2
block_size = 64

# Model hyperparameters (mini):
n_layer = 6  # layers
n_head = 6  # attention heads
n_embd = 192  # embedding dimension
dropout = 0.0
# feedforward dimension implicitly 4 * n_embd inside the model code

# Optimizer hyperparameters
learning_rate = 1e-4  # same as all models
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
