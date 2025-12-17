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

    # set random seed number
    random_seed_num = cfg.random_seed
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)
    
    # make data_info
    data_info = DataMaker(cfg)    
    
    model_type_list = ["FAJSCC","LAJSCC","LICRFJSCC","SwinJSCC","ResJSCC","ConvJSCC"]
    
    FA_list = ["smallFAJSCC","FAJSCC","largeFAJSCC"]
    LA_list = ["smallLAJSCC","LAJSCC","largeLAJSCC"]
    LICRF_list = ["LICRFJSCC","largeLICRFJSCC","hugeLICRFJSCC"]
    Swin_list = ["smallSwinJSCC","SwinJSCC","largeSwinJSCC"]
    Res_list = ["ResJSCC","largeResJSCC","hugeResJSCC"]
    Conv_list = ["ConvJSCC","largeConvJSCC","hugeConvJSCC"]
    
    model_list = FA_list + LA_list + LICRF_list + Swin_list + Res_list + Conv_list    
    rcpp=cfg.rcpp
    SNR=cfg.SNR_info
    rcpp_list=[rcpp]
    SNR_list=[SNR]
    

    total_eval_dict = get_total_eval_dict(cfg,logger,model_list,rcpp_list,SNR_list)


    save_performance_GFlops_Mmemory_plot(cfg,logger,total_eval_dict,model_list,model_type_list,rcpp,SNR)
    
    save_GFlops_performance_plot(cfg,logger,total_eval_dict,model_list,model_type_list,rcpp,SNR)
    
    save_Mmemory_performance_plot(cfg,logger,total_eval_dict,model_list,model_type_list,rcpp,SNR)        
    
    save_SNR_performance_table(cfg,logger,total_eval_dict,model_list,rcpp,SNR_list)
    
    
if __name__ == '__main__':
    main()
    