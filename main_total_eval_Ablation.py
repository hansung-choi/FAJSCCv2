from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from total_eval import *
import random
import os



@hydra.main(version_base = '1.1',config_path="configs",config_name='model_eval')
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'device: {device}')
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    model_name_list1 = ["SwinJSCC","LAJSCC","FAJSCC","FAPGBJSCC","FAJSCCwoAT","FAJSCCwoDf","FAJSCCwoLA","LAFAJSCC","FALAJSCC"]   
    model_name_collection = [model_name_list1]
    
    rcpp_list=[32]
    SNR_list=[1,4,7,10]  
    rcpp = 32

    # make data_info
    cfg.test_data = "DIV2K"
    data_info = DataMaker(cfg)     
    for model_name_list in model_name_collection:
        total_eval_dict_list = get_total_eval_dict_list(cfg,logger,model_name_list,rcpp_list,SNR_list)

        save_SNR_performance_table_meanstd(cfg, logger, total_eval_dict_list, model_name_list, rcpp, SNR_list,prefix=f'Ablation')
    
        save_gflops_memory_table_meanstd(cfg, logger, total_eval_dict_list, model_name_list, rcpp, 10,prefix=f'Ablation')
            
        total_eval_dict = get_total_eval_dict(cfg,logger,model_name_list,rcpp_list,SNR_list)
        
        save_SNR_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,prefix=f'Ablation_')

        save_gflops_memory_table(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR=10,prefix=f'Ablation_')




        
        
if __name__ == '__main__':
    main()
    














    