# train a very small model for testing purposes on the shakespeare dataset

from torch import device


out_dir = "out-shakespear_pos_id"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "semester-project"
wandb_run_name = "shakespeare_pos_id_16bits"

dataset = "shakespeare_pos_id"
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64  # context of up to 64 previous characters

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 2000
lr_decay_iters = 2000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# Macbook settings
device = "mps"  # run on mps (macs only)
compile = False

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# use python train.py config/train_shakespeare.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --vocab_size=287758
