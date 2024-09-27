import datetime
import pytz
import warnings
import os

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.optim as optim
import wandb
from tqdm import tqdm

from src.models.utr.vae import VAE
from src.metrics.vae_metric import batch_accuracy
from utils.beta_scheduler import StepBetaScheduler

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
    model = VAE(config)
    # load from check point
    if config.check_point is not None:
        model.load_state_dict(torch.load(config.check_point))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"=== model build completed. ===")
    if config.check_point is not None:
        print(f"model loaded from {config.check_point}")
    print(f"Total number of parameters:{total_params}")
    return model.to(config.device)

def build_optimizer(model, config): 
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=float(config.lr))
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=float(config.lr))
    else:
        optimizer = optim.Adam(model.parameters(), lr=float(config.lr))
        warnings.warn("not sopported optimizer, using AdamW instead.")

    print(f"=== optimizer build completed. ===")
    print(f"optimizer: {config.optimizer}")
    return optimizer

def build_betaScheduler(config): 
    # for kl-divergence weight
    return StepBetaScheduler(config.beta_flag, config.epoch, config.kld_weight)

def build_wandb_logger(config, model, TIME):
    config_dict = config.__dict__
    config_dict["TIME"] = TIME
    wandb.require("core")
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
    wandb.define_metric("train_kld/batch", step_metric="global_step")
    wandb.define_metric("train_ce/batch", step_metric="global_step")
    wandb.define_metric("lr/batch", step_metric="global_step")
    wandb.define_metric("train_loss/epoch", step_metric="epoch")
    wandb.define_metric("train_kld/epoch", step_metric="epoch")
    wandb.define_metric("train_ce/epoch", step_metric="epoch")
    wandb.define_metric("val_loss/epoch", step_metric="epoch")
    wandb.define_metric("val_kld/epoch", step_metric="epoch")
    wandb.define_metric("val_ce/epoch", step_metric="epoch")
    wandb.define_metric("val_acc/epoch", step_metric="epoch")

    return run

def train(config):
    torch.manual_seed(config.seed)
    if config.TIME is None:
        TIME = str(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        TIME = config.TIME
    device = config.device

    # components
    train_dataloader, val_dataloader = build_dataloader(config)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_betaScheduler(config)

    if config.log_flag:
        run = build_wandb_logger(config, model, TIME)

    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(config.epoch):
        model.train()
        train_loss_list = []
        kld_weight = scheduler.step()

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config.epoch}", leave=False):
            batch = batch[0]
            data = batch.to(device)

            # model forward
            recon_data, mu, logvar = model(data)
            # calculate loss
            loss, kld, ce = model.loss_function(data, recon_data, mu, logvar, kld_weight)

            loss.mean().backward()
            optimizer.step()
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
            val_kld_list = []
            val_ce_list = []
            val_acc_list = []
            for val_batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{config.epoch}", leave=False):
                val_batch = val_batch[0]
                data = val_batch.to(device)

                recon_data, mu, logvar = model(data)
                loss, kld, ce = model.loss_function(data, recon_data, mu, logvar, kld_weight)
                # metric
                val_acc = batch_accuracy(recon_data, data)

                val_loss_list.append(loss.mean().item())
                val_kld_list.append(kld.mean().item())
                val_ce_list.append(ce.mean().item())
                val_acc_list.append(val_acc)
            
        # log epoch results
        train_loss = sum(train_loss_list) / len(train_loss_list)
        val_loss = sum(val_loss_list) / len(val_loss_list)
        val_kld = sum(val_kld_list) / len(val_kld_list)
        val_ce = sum(val_ce_list) / len(val_ce_list)
        val_acc = sum(val_acc_list) / len(val_acc_list)

        if config.log_flag:
            wandb.log({"train_loss/epoch": train_loss, 
                   "val_loss/epoch": val_loss, 
                   "val_kld/epoch": val_kld, 
                   "val_ce/epoch": val_ce, 
                   "val_acc/epoch": val_acc, 
                   "epoch": epoch+1})
        
        # save the best model
        if config.save_flag and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.save_path, TIME+"_best_utr5vae.pt"))
    
        # model log
        if config.save_flag and epoch % config.save_model_epochs == 0 and epoch != 0:
            pt_file = os.path.join(config.save_path, TIME+f"_checkpoint_utr5vae_epoch{epoch}.pt")
            torch.save(model.state_dict(), pt_file)
    
    if config.save_flag:            
        torch.save(model.state_dict(), os.path.join(config.save_path, TIME+"_final_utr5vae.pt"))
    
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    print(">>> Training finished. >>>")
    if config.log_flag:
        run.finish()
