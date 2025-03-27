# train_mini.py
out_dir = "out-train_id_mini"
eval_interval = 250
eval_iters = 20
log_interval = 1

# We'll still only save when val improves
always_save_checkpoint = False

# Logging
wandb_log = False
wandb_project = "semester-project-gpu"
wandb_run_name = "openwebtext_id_mini"

# Dataset
dataset = "openwebtext_id"
# We are forcing the same effective batch size as GPT1:
batch_size = 64
gradient_accumulation_steps = 2
block_size = 64
stride = 32

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
warmup_iters = 300  # Since it's a larger run, we can afford to warm up a bit more

# Use epochs
train_on_epochs = True  # whether to train for a number of epochs instead of max_iters
num_epochs = 2  # number of epochs to train for, if train_on_epochs is True

# Learning rate scheduler
decay_lr = True
cooldown_fraction = 0.1
cooldown_type = "linear"

backwards = False

# Device settings
device = "cuda"
compile = False
