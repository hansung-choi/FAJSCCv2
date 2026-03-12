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
    
    model_name_list = ["ConvJSCC"]
    model_name_list = ["FAJSCC","LAJSCC","LICRFJSCC","SwinJSCC","ResJSCC","ConvJSCC","HugeFAJSCC"]   
    model_name_list = ["HugeFAJSCC"]  
    model_name_collection = [model_name_list]
    
    SNR_list = [10]
    rcpp_list=[12,16,24,32]
    #rcpp_list=[12,32]
    rcpp = 32
    SNR= 10

    # make data_info
    cfg.test_data = "Kodak"
    #cfg.test_data = "DIV2K"
    #cfg.test_data = "CLIC"
    data_info = DataMaker(cfg)     
    for model_name_list in model_name_collection:        
        total_eval_dict = get_total_eval_dict(cfg,logger,model_name_list,rcpp_list,SNR_list)

        save_CPP_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR,prefix=f'HugeFA_')
        
        save_CPP_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR,prefix=f'HugeFA_')
        
        
if __name__ == '__main__':
    main()
    















    
