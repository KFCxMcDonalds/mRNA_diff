# Models
## 5'UTR
1. Diffusion model for 5'UTR generation. Using huggingface UNet1DModel as the base model.(built, but not performes well.)
2. VAE+Diffusion model for 5'UTR generation. (under construction)


## 3'UTR

under construction...


## CDS

under construction...

# Usage

for now, this is just for my convenience.

## Train

## Sweep
1. initialize a sweep
```terminal
wandb sweep --project <project-name> <path-to-config file>
# return <sweep-ID>
```
2. start the sweep (or restart)
```terminal
wandb agent <sweep-ID>
```
3. parallelize agents, multi-GPU
```terminal
# terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep-ID>

# terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep-ID>
```
## Inference

under construction


# Utils
## data_preprocess.py

<>_full.fasta -(`remove_redundancy`)-> <>_redundancy.fasta -(`length_filter`)-> <>_redundancy_lbtoUb.fasta -(`read_and_pad`)-> <>_redundancy_lbtoUb.fasta -(`onehot_encoder`)-> ohe_<>_redundancy_lbtoUb.pt


the final .pt file will contain tensor of one-hoe-encoded sequences.


