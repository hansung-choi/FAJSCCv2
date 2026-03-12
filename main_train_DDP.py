from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train_DDP import *
from model.model_maker import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import os
import torch.optim as optim
import torch
import numpy as np
import logging
import hydra
from omegaconf import DictConfig

def load_trained_model(cfg, logger,model):
    task = cfg.task_name
    data = cfg.data_info.data_name
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    rcpp = str(cfg.rcpp).zfill(3)
    random_seed_num = cfg.random_seed
    metric = cfg.performance_metric
    random_num = str(random_seed_num).zfill(3)
    model_name = cfg.model_name
    save_dir = "../../saved_models/" #"../../../saved_models/"
        
    save_name = f"{random_num}_{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}.pt"
    save_name_backup = f"{random_num}_{task}_{data}_{chan_type}_SNR{str(SNR).zfill(3)}_rcpp{rcpp}_{metric}_{model_name}_backup.pt"
    if model_name in ["ConvJSCCrandomSNR","ResJSCCrandomSNR","SwinJSCCrandomSNR","LICRFJSCCrandomSNR","LAJSCCrandomSNR","FAJSCCrandomSNR"]:
        save_name = f"{random_num}_{task}_{data}_{chan_type}_rcpp{rcpp}_{metric}_{model_name}.pt"
        save_name_backup = f"{random_num}_{task}_{data}_{chan_type}_rcpp{rcpp}_{metric}_{model_name}_backup.pt"
          
    model_info_save_path = save_dir + save_name
    model_backup_info_save_path = save_dir + save_name_backup
    if os.path.exists(model_info_save_path):
        try:
            #model.load_state_dict(torch.load(model_info_save_path))
            ckpt = torch.load(model_info_save_path, map_location="cpu")
            model.load_state_dict(ckpt)
            logger.info(f'The saved model is loaded')
            saved_model_epoch = model.epoch.item()
            logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')
            
        except Exception as ex:
            logger.info(f'Error occured during saved model is loaded')
            logger.info(f'Error info:',ex)
            try:
                ckpt = torch.load(model_backup_info_save_path, map_location="cpu")
                model.load_state_dict(ckpt)
                #model.load_state_dict(torch.load(model_backup_info_save_path))
                logger.info(f'The saved backup model is loaded')
                saved_model_epoch = model.epoch.item()
                logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')
            
            except Exception as ex:
                logger.info(f'Error occured during backup model is loaded')
                logger.info(f'Error info:',ex)

                logger.info(f'Train epoch is initialized, new default model is made')
                model = ModelMaker(cfg)   # make model and set appropriate task name
    else:
        logger.info(f'There is no trained model')
        
    return model


@hydra.main(version_base = '1.1',config_path="configs",config_name='train')
def main(cfg: DictConfig):        
    logger = logging.getLogger(__name__)
    
    use_ddp = "LOCAL_RANK" in os.environ  # torchrun sets this
    cfg.use_ddp = use_ddp
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    if is_main:
        logger.info(f'---------------------------------------------------------------')
        logger.info(f'device: {device}')

    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # set random seed number
    random_seed_num = cfg.random_seed
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)
    
    # make data_info
    data_info = DataMaker(cfg)

    # make model
    model = ModelMaker(cfg)   # make model


    # make criterion
    criterion = LossMaker(cfg)

    saved_model_epoch = 0
    
    if is_main:
        logger.info(f'---'*10)
        logger.info(f'Try loading pretrained model')
        train_data_name = cfg.data_info.data_name
        model = load_trained_model(cfg, logger,model)
        logger.info(f'---'*10)
    
    saved_model_epoch = model.epoch.item()                   
                     
    if use_ddp:
        dist.barrier()  # ensure rank0 finished loading
    
    model = model.to(device)
       
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    # Make all ranks agree before exiting
    if use_ddp:
        t = torch.tensor([saved_model_epoch], device=device, dtype=torch.long)
        dist.broadcast(t, src=0)
        saved_model_epoch = int(t.item())

    should_stop = saved_model_epoch >= cfg.total_max_epoch         
    if should_stop:
        if (not use_ddp) or dist.get_rank() == 0:
            logger.info(f"saved model already exists, total_max_epoch is {cfg.total_max_epoch}")
        if use_ddp:
            dist.destroy_process_group()
        return None

    random_seed_num = int(saved_model_epoch)
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)        
        
    # make optimizer
    base_model = model.module if use_ddp else model
    optimizer= OptimizerMaker(base_model,cfg) 


    # make scheduler
    milestones = [99999999]  #[60, 80]
    if cfg.data_info.data_name in ["Flickr30k"]:
        milestones = [70-saved_model_epoch,75-saved_model_epoch] # milestones = [90,95]

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    
    if is_main:    
        logger.info(hydra_cfg['runtime']['output_dir'])
        logger.info(f'---------------------------------------------------------------')
        logger.info(f'Task: {cfg.task_name}')
        logger.info(f'Data: {cfg.data_info.data_name}')
        logger.info(f'chan_type: {cfg.chan_type}')
        logger.info(f'SNR: {cfg.SNR_info}')
        logger.info(f'rcpp: {cfg.rcpp}')
        logger.info(f'performance_metric: {cfg.performance_metric}')
        logger.info(f'Model: {cfg.model_name}')
        logger.info(f'Learning rate: {cfg.learning_rate}')



    # train model
    train_model(cfg, logger, model, data_info, criterion, optimizer, scheduler)

 







if __name__ == '__main__':
    main()
    
    
    
    
    