# train a medium-sized model for MacBook Pro with M1 Pro processor on Shakespeare dataset

from torch import device

out_dir = "out-shakespeare-medium"
eval_interval = 500  # evaluate less frequently to speed up training
eval_iters = 40
log_interval = 10  # don't print too often to avoid cluttering the console

# save checkpoints when validation improves or at significant milestones
always_save_checkpoint = True

wandb_log = False  # set to True if you want to use Weights & Biases logging
wandb_project = "shakespeare-medium"
wandb_run_name = "medium-gpt"

dataset = "shakespeare"
gradient_accumulation_steps = 4  # accumulate gradients to simulate larger batch
batch_size = 8  # smaller batch size to fit in memory
block_size = 256  # increased context window from small model

# Medium GPT model - larger than small but still M1 Pro friendly
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1  # add some regularization for larger model

learning_rate = 5e-4  # slightly lower learning rate for larger model
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 5e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 200  # more warmup for larger model

# M1 Pro specific settings
device = "mps"  # Metal Performance Shaders for Macs
compile = False  # torch.compile() is currently not fully supported on MPS

# Memory optimization
# Enable gradient checkpointing if needed to save memory at the cost of recomputation
# gradient_checkpointing = True

# For command line use:
# python train.py config/train_shakespeare_medium.py --device=mps --compile=False --eval_iters=40
