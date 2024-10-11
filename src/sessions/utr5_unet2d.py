import os, warnings, datetime, io
import pytz
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, random_split, TensorDataset

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from tqdm import tqdm
import wandb

from src.models.utr.unet2d import UNet2D
from src.models.utr.vae import VAE
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
    model = UNet2D(config)
    if config.check_point is not None:
        model.load_state_dict(torch.load(config.check_point))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"=== model build completed. ===")
    if config.check_point is not None:
        print(f"model loaded from {config.check_point}")
    print(f"Total number of parameters:{total_params}")
    return model.to(config.device)

def build_vae_model(config):
    vae = VAE(config)
    vae.load_state_dict(torch.load(config.vae_path, map_location=config.device))
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"=== VAE model build completed. ===")
    print(f"VAE model loaded from {config.vae_path}")
    print(f"Total number of VAE parameters:{total_params}")
    return vae.to(config.device)

def build_scheduler(config):
    if config.scheduler == 'DDPM':
        scheduler = DDPMScheduler(num_train_timesteps=config.timesteps)
    elif config.scheduler == 'DDIM':
        scheduler = DDIMScheduler(num_train_timesteps=config.timesteps)
    else:
        scheduler = DDPMScheduler(num_train_timesteps=config.timesteps)
        warnings.warn("not sopported scheduler, using DDPM instead.")

    print(f"=== scheduler build completed. ===")
    print(f"Scheduler: {config.scheduler}")
    return scheduler

def build_optimizer(model, config):
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=float(config.lr))
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=float(config.lr))
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(config.lr))
        warnings.warn("not sopported optimizer, using Adam instead.")

    print(f"=== optimizer build completed. ===")
    print(f"optimizer: {config.optimizer}")
    return optimizer

def build_wandb_logger(config, model, TIME):
    config_dict = config.__dict__
    config_dict["TIME"] = TIME
    wandb.require("core")
    
    if config.resume_flag:
        runid = config.resume_runid
        run = wandb.init(
            project = config.logger_project,
            name = config.logger_runname,
            resume='must',
            config = config_dict,
            id=runid
        )
    else:
        run = wandb.init(
            project = config.logger_project,
            name = config.logger_runname,
            notes = config.logger_note,
            config = config_dict,
        )
    
    wandb.watch(model, log='all', log_freq=1000, log_graph=True)
    wandb.config.system = {
        "monitor": True
    }

    wandb.define_metric("global_step")  # every batch
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss/batch", step_metric="global_step")
    wandb.define_metric("lr/batch", step_metric="global_step")
    wandb.define_metric("train_loss/epoch", step_metric="epoch")
    wandb.define_metric("val_loss/epoch", step_metric="epoch")

    return run

def train(config):
    torch.manual_seed(config.seed)
    if config.resume_flag:
        TIME = config.resume_TIME
    else:
        TIME = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S"))
    device = config.device
    
    # components
    train_dataloader, val_dataloader = build_dataloader(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(config)
    vae = build_vae_model(config)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.epoch * len(train_dataloader)
    )

    if config.log_flag:
        run = build_wandb_logger(config, model, TIME)

    # ====================
    # ===== training =====
    # ====================

    global_step = 0  # for wandb log
    best_val_loss = float('inf')
    if config.resume_flag:
        start_epoch = config.resume_epoch
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config.epoch):
        model.train()  # switch to train mode
        train_loss_list = []
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config.epoch}", leave=False):
            batch = batch[0]
            clean_data = batch.to(device)
            with torch.no_grad():
                mean, logvar = vae.encoder(clean_data)
                clean_z = vae.reparameterize(mean, logvar)  # sample from latent space
            clean_z = clean_z.to(device)
            # add noise on clean_z
            noise = torch.randn_like(clean_z).to(device)
            
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (batch.size(0),), device=device
            ).long()
            timesteps = timesteps.to(device)

            # add noise to the clean_z
            noisy_z = scheduler.add_noise(clean_z, noise, timesteps)
            # predict the clean data added by scheduler
            noise_pred = model(noisy_z, timesteps, return_dict=False)[0]

            # loss backwards
            loss = F.mse_loss(noise_pred, noise) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # logs
            train_loss_list.append(loss.mean().item())
            global_step += config.batch

            if config.log_flag:
                wandb.log({
                    "train_loss/batch": loss.mean().item(),
                    "lr/batch": float(config.lr),
                    "global_step": global_step
                })

        # end of one epoch (all data has been used to train model once)
        ## evalueation
        model.eval()
        with torch.no_grad():
            val_loss_list = []
            for val_batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{config.epoch}", leave=False):
                val_batch = val_batch[0]
                clean_data = val_batch.to(device)

                mean, logvar = vae.encoder(clean_data)
                clean_z = vae.reparameterize(mean, logvar)  # sample from latent space
                clean_z = clean_z.to(device)
                # add noise on clean_z
                noise = torch.randn_like(clean_z).to(device)

                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (val_batch.size(0),), device=device
                ).long()
                timesteps = timesteps.to(device)

                # add noise to the clean_z
                noisy_z = scheduler.add_noise(clean_z, noise, timesteps)
                # predict the clean data added by scheduler
                noise_pred = model(noisy_z, timesteps, return_dict=False)[0]

                # loss backwards
                v_loss = F.mse_loss(noise_pred, noise) 
                val_loss_list.append(v_loss.mean().item())

            # log epoch results
            if config.log_flag:
                wandb.log({
                    "train_loss/epoch": np.mean(train_loss_list),
                    "val_loss/epoch": np.mean(val_loss_list),
                    "epoch": epoch+1
                })

            # save the best model for now
            if config.save_flag and np.mean(val_loss_list) < best_val_loss:
                best_val_loss = np.mean(val_loss_list)
                torch.save(model.state_dict(), os.path.join(config.save_path, TIME+"_best_unet2d.pt"))
        
        # model log
        if config.save_flag and epoch % config.save_model_epochs == 0 and epoch != 0:
            pt_file = os.path.join(config.save_path, TIME+f"_checkpoint_unet2d_epoch{epoch}.pt")
            torch.save(model.state_dict(), pt_file)

    if config.save_flag:            
        torch.save(model.state_dict(), os.path.join(config.save_path, TIME+"_final_unet2d.pt"))
            
    # clean cuda memory
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    print(">>> Training finished. >>>")
    if config.log_flag:
        run.finish()


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
