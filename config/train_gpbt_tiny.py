# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'Expermintal-BiDir-GPBT'
wandb_run_name='gpbt(tiny)-7_M'
init_from = 'resume'
out_dir = 'out_tiny'
#compile = False

# 2 batch size * 256 block size 512 tokens per itteration
# if gradient_accumulation = 2 then 1024 etc usefull for GPU with smaller memory size
# each gradient_accumulation step doubles itteration time.
batch_size = 1
block_size = 768
gradient_accumulation_steps = 6

max_iters = 600000
lr_decay_iters = 600000

# model
n_layer = 4
n_head = 4
n_embd = 128

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10
dropout_adjust_interval = 200
dropout = 0.0001 #if we initilize at 0.0 the code will not work for hte dynamic dropout as anything * 0 is 0

# weight decay
weight_decay = 1e-1
learning_rate = 1e-3 #3e-5 --for larger models -- 1e-3 -- for small models 