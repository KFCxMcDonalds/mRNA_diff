import os, warnings, datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from tqdm import tqdm
import wandb

from ..models.utr.unet import UNet1D
from utils.tensor_helper import tensor2rna, write2fasta

def build_dataloader(config):
    # load from pt file
    loaded_dataset = torch.load(config.data_path, weights_only=True)
    dataset = TensorDataset(loaded_dataset)

    train_size = int(config.train_prop * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch, shuffle=config.shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch, shuffle=config.shuffle)

    print(f"=== data building completed. ===")
    print(f'train data: total batch: {len(train_dataloader)}; total samples: {len(train_dataloader.dataset)}')
    print(f'validate data: total batch: {len(val_dataloader)}; total samples: {len(val_dataloader.dataset)}')
    return train_dataloader, val_dataloader
    
def build_model(config):

    model = UNet1D(config)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"=== model build completed. ===")
    print(f"Total number of parameters:{total_params}")
    return model.to(config.device)

def build_scheduler(config):

    if config.scheduler == 'DDPM':
        scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    elif config.scheduler == 'DDIM':
        scheduler = DDIMScheduler(num_train_timesteps=config.num_train_timesteps)
    else:
        scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
        warnings.warn("not sopported scheduler, using DDPM instead.")

    print(f"=== scheduler build completed. ===")
    print(f"Scheduler: {config.scheduler}")
    return scheduler

def build_optimizer(model, config):

    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=float(config.lr))
    else:
        optimizer = optim.AdamW(model.parameters(), lr=float(config.lr))
        warnings.warn("not sopported optimizer, using AdamW instead.")

    print(f"=== optimizer build completed. ===")
    print(f"optimizer: {config.optimizer}")
    return optimizer

def build_wandb_logger(config):
    config_dict = config.__dict__
    wandb.require("core")
    run = wandb.init(
        project = "5utr-UNet1DModel",
        config = config_dict,
    )
    
    wandb.define_metric("global_step")  # every batch
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss/batch", step_metric="global_step")
    wandb.define_metric("lr/batch", step_metric="global_step")
    wandb.define_metric("train_loss/epoch", step_metric="epoch")
    wandb.define_metric("test_loss/epoch", step_metric="epoch")
    return run

def train(config):
    torch.manual_seed(config.seed)

    TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    device = config.device
    
    # components
    train_dataloader, val_dataloader = build_dataloader(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(config)
#     criterion = torch.nn.MSELoss()

    # logger
    run = build_wandb_logger(config)

    global_step = 0  # for wandb log
    best_val_loss = float('inf')
    for epoch in range(config.epoch):
        print(f"=== training epochs: {epoch+1}/{config.epoch} ===")
        model.train()  # switch to train mode
        train_loss_list = []
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config.epoch}", leave=False):
            batch = batch[0]
            clean_data = batch.to(device)

            # sample noise to add to the sequences
            noise = torch.randn_like(batch).to(device)
            
            # sample a random timestep for each sequence
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (batch.size(0),), device=device
            ).long()

            # add noise to the clean sequences
            noisy_seq = scheduler.add_noise(clean_data, noise, timesteps)

            # predict the clean data added by scheduler
            seq_pred = model(noisy_seq, timesteps, return_dict=False)

#             loss = criterion(seq_pred, noise)
            loss_per_sample = model.loss_function(seq_pred, clean_data, 'none')
            loss = loss_per_sample.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logs
#             train_loss_list.append(loss.mean().item())
            train_loss_list.append(loss_per_sample.mean().item())
            global_step += config.batch
            wandb.log({"train_loss/batch": loss_per_sample.mean().item(), "lr/batch": float(config.lr), "global_step": global_step})

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
                    0, scheduler.config.num_train_timesteps, (val_batch.size(0),), device=device
                ).long()
                val_noisy_seq = scheduler.add_noise(clean_data, val_noise, val_timesteps)


                seq_pred = model(val_noisy_seq, val_timesteps, return_dict=False)
#                 val_loss = criterion(val_noise_pred, val_noise)
                val_loss_per_sample = model.loss_function(seq_pred, clean_data, 'none')
                loss = val_loss_per_sample.sum()

                val_loss_list.append(val_loss_per_sample.mean().item())

            # log epoch results
            train_loss = sum(train_loss_list) / len(train_loss_list)
            val_loss = sum(val_loss_list) / len(val_loss_list)
            wandb.log({"train_loss/epoch": train_loss, "test_loss/epoch": val_loss, "epoch": epoch})

            # save the best model for now
            if config.save_flag and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.save_path, TIME+"_best_unet1dmodel.pt"))
        
        # model log
        if config.save_flag and epoch % config.save_model_epochs == 0 and epoch != 0:
            pt_file = os.path.join(config.save_path, TIME+f"_diffusion_unet_epoch_{epoch}.pt")
            torch.save(model.state_dict(), pt_file)

    if config.save_flag:            
        torch.save(model.state_dict(), os.path.join(config.save_path, TIME+"_final_unet_model.pt"))
            
    # clean cuda memory
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    print(">>> Training finished. >>>")


def batch_generate(config, scheduler, model):
    # initialize noise
    noise = torch.randn(config.current_gen_batch, config.in_channels, config.input_length, device=config.device)
    progress_bar = tqdm(total=config.num_train_timesteps, desc='Generating batch', unit='step')

    with torch.no_grad():
        for t in reversed(range(config.num_train_timesteps)):
            clean_seqs = model(noise, t)
            noise = scheduler.step(clean_seqs, t, noise).prev_sample

            progress_bar.update(1)
    progress_bar.close()
    return clean_seqs


def generate(config):
    TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    device = config.device
    gen_batch = config.gen_batch_size
    total_samples = config.gen_num

    # load diffusion model
    model = UNet1D(config).to(device)
    state_dict = torch.load(config.gen_model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()    
    scheduler = build_scheduler(config)

    scheduler.set_timesteps(config.num_train_timesteps)
    file = config.gen_seqs_path + f"{TIME}_unet1dmodel.fasta"

    for i in range(total_samples // gen_batch):
        print(f"=== generating batch {i+1} ===")
        utr_onehot = batch_generate(config, scheduler, model)
        utr5 = [tensor2rna(ele) for ele in utr_onehot]
        write2fasta(utr5, file)
    if total_samples % gen_batch != 0:
        config.current_gen_batch = total_samples % gen_batch
        utr_onehot = batch_generate(config, scheduler, model)
        utr5 = [tensor2rna(ele) for ele in utr_onehot]
        write2fasta(utr5, file)

    print(">>> Generation finished. >>>")
