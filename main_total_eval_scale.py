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
    
    model_type_list = ["FAJSCC","LAJSCC","LICRFJSCC","SwinJSCC","ResJSCC","ConvJSCC"]
    
    FA_list = ["smallFAJSCC","FAJSCC","largeFAJSCC"]
    LA_list = ["smallLAJSCC","LAJSCC","largeLAJSCC"]
    LICRF_list = ["LICRFJSCC","largeLICRFJSCC","hugeLICRFJSCC"]
    Swin_list = ["smallSwinJSCC","SwinJSCC","largeSwinJSCC"]
    Res_list = ["ResJSCC","largeResJSCC","hugeResJSCC"]
    Conv_list = ["ConvJSCC","largeConvJSCC","hugeConvJSCC"]
    
    
    #FA_list = ["smallFAJSCC"]
    #LA_list = ["smallLAJSCC"]
    #LICRF_list = ["largeLICRFJSCC"]
    #Swin_list = ["smallSwinJSCC"]
    #Res_list = ["largeResJSCC"]
    #Conv_list = ["largeConvJSCC"]    
    model_list = FA_list + LA_list + LICRF_list + Swin_list + Res_list + Conv_list

    model_scale_list = ["ConvJSCC","ResJSCC","smallSwinJSCC","LICRFJSCC","smallLAJSCC","smallFAJSCC","largeConvJSCC","largeResJSCC","SwinJSCC","largeLICRFJSCC","LAJSCC","FAJSCC","hugeConvJSCC","hugeResJSCC","largeSwinJSCC","hugeLICRFJSCC","largeLAJSCC","largeFAJSCC"]

    
    rcpp=cfg.rcpp
    SNR=cfg.SNR_info
    rcpp_list=[rcpp]
    SNR_list=[SNR]
    
    # make data_info
    cfg.test_data = "DIV2K"
    data_info = DataMaker(cfg) 
    total_eval_dict = get_total_eval_dict(cfg,logger,model_list,rcpp_list,SNR_list)
    
    save_GFlops_performance_plot(cfg,logger,total_eval_dict,model_list,model_type_list,rcpp,SNR)
    
    save_Mmemory_performance_plot_type(cfg,logger,total_eval_dict,model_list,model_type_list,rcpp,SNR)          

    save_SNR_performance_table(cfg,logger,total_eval_dict,model_scale_list,rcpp,SNR_list,prefix="Scale_")
    
    save_gflops_memory_table(cfg,logger,total_eval_dict,model_scale_list,rcpp,SNR,prefix='Scale_')
    
    total_eval_dict_list = get_total_eval_dict_list(cfg,logger,model_scale_list,rcpp_list,SNR_list)

    save_SNR_performance_table_meanstd(cfg, logger, total_eval_dict_list, model_scale_list, rcpp, SNR_list,prefix='Scale_')
    
    save_gflops_memory_table_meanstd(cfg, logger, total_eval_dict_list,model_scale_list, rcpp, SNR,prefix='Scale_')
    
    save_GFlops_performance_plot_meanstd(cfg,logger,total_eval_dict_list,model_list,model_type_list,rcpp,SNR,prefix='Scale_')
    
    save_Mmemory_performance_plot_meanstd(cfg,logger,total_eval_dict_list,model_list,model_type_list,rcpp,SNR,prefix='Scale_')
    
        
    # make data_info
    cfg.test_data = "Kodak"
    data_info = DataMaker(cfg) 
    total_eval_dict = get_total_eval_dict(cfg,logger,model_list,rcpp_list,SNR_list)
    
    save_GFlops_performance_plot(cfg,logger,total_eval_dict,model_list,model_type_list,rcpp,SNR)
    
    save_Mmemory_performance_plot_type(cfg,logger,total_eval_dict,model_list,model_type_list,rcpp,SNR)          

    save_SNR_performance_table(cfg,logger,total_eval_dict,model_scale_list,rcpp,SNR_list,prefix="Scale_")
    
    save_gflops_memory_table(cfg,logger,total_eval_dict,model_scale_list,rcpp,SNR,prefix='Scale_')
    
    total_eval_dict_list = get_total_eval_dict_list(cfg,logger,model_scale_list,rcpp_list,SNR_list)

    save_SNR_performance_table_meanstd(cfg, logger, total_eval_dict_list, model_scale_list, rcpp, SNR_list,prefix='Scale_')
    
    save_gflops_memory_table_meanstd(cfg, logger, total_eval_dict_list,model_scale_list, rcpp, SNR,prefix='Scale_')
    
    save_GFlops_performance_plot_meanstd(cfg,logger,total_eval_dict_list,model_list,model_type_list,rcpp,SNR,prefix='Scale_')
    
    save_Mmemory_performance_plot_meanstd(cfg,logger,total_eval_dict_list,model_list,model_type_list,rcpp,SNR,prefix='Scale_')
    
    
    
    


        
        
if __name__ == '__main__':
    main()
    














    