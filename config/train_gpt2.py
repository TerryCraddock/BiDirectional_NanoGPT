# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
wandb_log = True
wandb_project = 'Expermintal-BiDir-GPBT'
wandb_run_name='gpbt-151_M'
init_from = 'scratch'
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 #5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# model
n_layer = 12
n_head = 12
n_embd = 768

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10
dropout_adjust_interval = 200
dropout = 0.0001 #if we initilize at 0.0 the code will not work for hte dynamic dropout as anything * 0 is 0

# weight decay
weight_decay = 1e-1