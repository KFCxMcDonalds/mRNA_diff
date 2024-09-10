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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sweep
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'test_loss/epoch',
        'goal': 'minimize',
    },
    'parameters': {
        # dataloader
        "data_path": {
            'value': '5utr_95_sweep.pt' 
        },
        'batch': {
            'value': 64
        },
        'train_prop': {
            'value': 0.8    
        },
        'valid_prop': {
            'value': 0.2    
        },
        'shuffle': {
            'value': True
        },
        # model
        'input_length': {
            'value': 512
        },
        'in_channels': {
            'value': 5
        },
        'out_channels': {
            'value': 5
        },
        'layers_per_block': {
            'values': [2, 4, 6, 8]
        },
        'block_out_channels': {
            'values': [[128, 256, 512], [16, 64, 128], [16, 32, 64], [64, 128, 256], [64, 64, 64],[16, 16, 16], [256, 256, 256]]
        },
        'down_block_types': {
            'values': [["DownBlock1D", "AttnDownBlock1D", "DownBlock1D"], ["DownBlock1D", "DownBlock1D", "DownBlock1D"], ['AttnDownBlock1D', "DownBlock1D", "DownBlock1D"], ["DownBlock1D", "DownBlock1D", "AttnDownBlock1D"]]
        },
        'up_block_types': {
            'values': [["UpBlock1D", "AttnUpBlock1D", "UpBlock1D"], ["UpBlock1D", "UpBlock1D", "UpBlock1D"], ['AttnUpBlock1D', "UpBlock1D", "UpBlock1D"], ["AttnUpBlock1D", "UpBlock1D", "UpBlock1D"]]
        },
        # train
        'scheduler': {
            'values': ["DDPM", "DDIM"]
        },
        'num_train_timesteps': {
            'values': [1000, 100]
        },
        'optimizer' : {
            'value': "AdamW"
        },
        'lr': {
             'values': [1e-3, 1e-4, 1e-5, 1e-6]
         },
        'epoch' : {
            'value': 3
        },
        # log
        'save_model_epochs' : {
            'value': 100  # not save 
        },
        'save_path' : {
            'value': './save_models'
        },
        'seed' : {
            'value': 2024
        },
    },
    'run': 200
}

# data
def build_dataset(data_path, batch, train_prop, shuffle):

    loaded_sequences = torch.load(data_path, weights_only=True)

    dataset = TensorDataset(loaded_sequences)

    train_size = int(train_prop * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch)

    print(f'train data: {len(train_dataloader.dataset)}')
    print(f'test data: {len(val_dataloader.dataset)}')
    print(f"=== data building completed. ===")
    
    return train_dataloader, val_dataloader

# model

def build_model(device, sample_size, in_channels, out_channels, layers_per_block,  block_out_channels, down_block_types, up_block_types):

    class UNet1DWithSoftmax(nn.Module):
        def __init__(self):
            super(UNet1DWithSoftmax, self).__init__()
            self.unet = UNet1DModel(
                sample_size = sample_size,  # the input length of data
                in_channels = in_channels, # the one-hot encoded data
                out_channels = out_channels,  # reconstructed channel of data (also 5, cuz we need gain a sequence)
                layers_per_block = layers_per_block,  # how many ResNet layers to use per UNet block
                block_out_channels = block_out_channels,  # block output channels on each side
                down_block_types = down_block_types,
                up_block_types = up_block_types
            )
            self.softmax = nn.Softmax(dim=1)  # apply to channels (=>5, 512)

        def forward(self, x, timesteps, return_dict=False):
            x = self.unet(x, timesteps, return_dict=return_dict)[0]
            x = self.softmax(x)
            return x

    model = UNet1DWithSoftmax().to(device)

    print(f"=== model building completed. ===")
    return model

# optimizer

def build_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        print("unsuppor optimizer, using AdamW")

    print(f"=== optimizer building completed. ===")
    return optimizer

# scheduler

def build_scheduler(optimizer, scheduler, num_train_timesteps, train_dataloader, epoch):

    if scheduler == "DDPM":
        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)  # clip_sample: True?
    elif scheduler == "DDIM":
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    return scheduler


def train():
    with wandb.init():
        config = wandb.config

        train_dataloader, val_dataloader = build_dataset(config.data_path, config.batch, config.train_prop, config.shuffle)
        model = build_model(device, config.input_length, 
                            config.in_channels, config.out_channels, 
                            config.layers_per_block,  config.block_out_channels, µ
                            config.down_block_types, config.up_block_types)

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

        optimizer = build_optimizer(model, config.optimizer, config.lr)
        scheduler = build_scheduler(optimizer, config.scheduler, config.num_train_timesteps, train_dataloader, config.epoch)

        criterion = torch.nn.MSELoss()
        
        global_step = 0  # for wandb log
        best_val_loss = float('inf')
        for epoch in range(config.epoch):
            print(f"=== training epochs: {epoch}/{config.epoch} ===")
            model.train()  # switch to train mode
            train_loss_list = []
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config.epoch}", leave=False):
                batch = batch[0]
                clean_data = batch.to(device)

                # sample noise to add to the sequences
                noise = torch.randn_like(batch).to(device)
                
                # sample a random timestep for each sequence
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps, (batch.size(0),), device=device
                ).long()

                # add noise to the clean sequences
                noisy_seq = scheduler.add_noise(clean_data, noise, timesteps)

                # predict the noise added by scheduler
                noise_pred = model(noisy_seq, timesteps, return_dict=False)
                loss = criterion(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logs
                train_loss_list.append(loss.mean().item())
                global_step += config.batch
                wandb.log({"train_loss/batch": loss.item(), "lr": config.lr, "global_step": global_step})

            # end of one epoch (all data has been used to train model once)
            ## evalueation
            model.eval()
            with torch.no_grad():
                val_loss_list = []
                for val_batch in val_dataloader:
                    val_batch = val_batch[0]
                    clean_data = val_batch.to(device)
                    val_noise = torch.randn_like(val_batch).to(device)
                    val_timesteps = torch.randint(
                        0, scheduler.num_train_timesteps, (val_batch.size(0),), device=device
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
            
    # clean cuda memory
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    print(">>> Training finished.")


# wandb
wandb.require("core")
wandb.login()
sweep_id = wandb.sweep(sweep_config, project="5utr-diffusion-unet-sweep")
wandb.agent(sweep_id, train)

