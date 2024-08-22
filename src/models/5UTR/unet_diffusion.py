import os
import datetime

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.optim as optim
import torch.nn as nn

from diffusers import UNet1DModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

import wandb
from tqdm import tqdm

# configs

class TrainingConfig:
    # dataloader
    data_path = "5utr_95.pt"
    batch = 64
    train_prop = 0.8
    valid_prop = 0.2
    shuffle = True

    # model
    input_length = 512
    in_channels = 5
    out_channels = 5
    layers_per_block = 5
    block_out_channels = [256, 256, 512, 512]  # 4 blocks each side
    down_block_types = ["DownBlock1D", "DownBlock1D", "AttnDownBlock1D", "DownBlock1D"]
    up_block_types = ["UpBlock1D", "AttnUpBlock1D", "UpBlock1D", "UpBlock1D"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train
    scheduler = "DDPM"  # ['DDPM', 'DDIM']
    num_train_timesteps = 1000
    optimizer = "AdamW"  # ['AdamW', ...]
    lr_warmup_steps = 500
    epoch = 1000
    lr = 1e-5

    # log
    save_model_epochs = 10
    save_path = "./save_models"
    # mixed_precision = 'fp16'
    
    seed = 2024
    
config = TrainingConfig()

# data

loaded_sequences = torch.load(config.data_path, weights_only=True)

dataset = TensorDataset(loaded_sequences)

train_size = int(config.train_prop * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=config.batch, shuffle=config.shuffle)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch)

# model

class UNet1DWithSoftmax(nn.Module):
    def __init__(self):
        super(UNet1DWithSoftmax, self).__init__()
        self.unet = UNet1DModel(
            sample_size = config.input_length,  # the input length of data
            in_channels = config.in_channels,  # the one-hot encoded data
            out_channels = config.out_channels,  # reconstructed channel of data (also 5, cuz we need gain a sequence)
            layers_per_block = config.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels = config.block_out_channels,  # block output channels on each side
            down_block_types = config.down_block_types,
            up_block_types = config.up_block_types
        )
        self.softmax = nn.Softmax(dim=1)  # apply to channels (=>5, 512)

    def forward(self, x, timesteps, return_dict=False):
        x = self.unet(x, timesteps, return_dict=return_dict)[0]
        x = self.softmax(x)
        return x

model = UNet1DWithSoftmax().to(config.device)

# scheduler
if config.scheduler == "DDPM":
    scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)  # clip_sample: True?
elif config.scheduler == "DDIM":
    scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps)

# optimizer

optimizer = optim.AdamW(model.parameters(), lr=config.lr)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer = optimizer, 
    num_warmup_steps = config.lr_warmup_steps,
    num_training_steps = (len(train_dataloader) * config.epoch),
)
criterion = torch.nn.MSELoss()

# wandb
wandb.require("core")
wandb.login()

wandb.init(
    project = "5utr-diffusion",
    config = config
)

wandb.watch(model, log='all', log_freq=1000, log_graph=True)  # log weights of model every 1000 batches
wandb.config.system = {
    "monitor": True,
}
wandb.define_metric("global_step")  # every batch
wandb.define_metric("epoch")
wandb.define_metric("train_loss/batch", step_metric="global_step")
wandb.define_metric("lr/batch", step_metric="global_step")
wandb.define_metric("train_loss/epoch", step_metric="epoch")
wandb.define_metric("test_loss/epoch", step_metric="epoch")

# !! train it

global_step = 0  # for wandb log
best_val_loss = float('inf')

for epoch in tqdm(range(config.epoch), desc="Epochs"):
    model.train()  # switch to train mode
    train_loss_list = []
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config.epoch}", leave=False):
        batch = batch[0]
        clean_data = batch.to(config.device)

        # sample noise to add to the sequences
        noise = torch.randn_like(batch).to(config.device)
        
        # sample a random timestep for each sequence
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps, (batch.size(0),), device=config.device
        ).long()

        # add noise to the clean sequences
        noisy_seq = scheduler.add_noise(clean_data, noise, timesteps)

        # predict the noise added by scheduler
        noise_pred = model(noisy_seq, timesteps, return_dict=False)
        loss = criterion(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # logs
        train_loss_list.append(loss.mean().item())
        global_step += 1
        wandb.log({"train_loss/batch": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "global_step": global_step})

    # end of one epoch (all data has been used to train model once)
    ## evalueation
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        for val_batch in val_dataloader:
            val_batch = val_batch[0]
            clean_data = val_batch.to(config.device)
            val_noise = torch.randn_like(val_batch).to(config.device)
            val_timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (val_batch.size(0),), device=config.device
            ).long()
            val_noisy_seq = scheduler.add_noise(clean_data, val_noise, val_timesteps)

            val_noise_pred = model(val_noisy_seq, val_timesteps, return_dict=False)
            val_loss = criterion(val_noise_pred, val_noise)

            val_loss_list.append(val_loss.mean().item())

        # log epoch results
        train_loss = sum(train_loss_list) / len(train_loss_list)
        val_loss = sum(val_loss_list) / len(val_loss_list)
        wandb.log({"train_loss/epoch": train_loss, "test_loss/epoch": val_loss, "epoch": epoch})

        # save the best model for now
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            torch.save(model.state_dict(), os.path.join(config.save_path, TIME+f"_5utr-diffusion_best_unet_model.pt"))
    
    # model log
    if epoch % config.save_model_epochs == 0 and epoch != 0:
        TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        pt_file = os.path.join(config.save_path, TIME+f"_5utr-diffusion_unet_epoch_{epoch}.pt")
        torch.save(model.state_dict(), pt_file)
        
TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
torch.save(model.state_dict(), os.path.join(config.save_path, TIME+"_final_unet_model.pt"))
print(">>> Training finished.")
