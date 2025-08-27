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
    
    model_name_list_r12 = ["FAJSCCr12_00","FAJSCCr12_02","FAJSCCr12_04","FAJSCCr12_05","FAJSCCr12_06","FAJSCCr12_08","FAJSCCr12_10"]

    model_name_list_r1 = ["FAJSCCr1_00","FAJSCCr1_02","FAJSCCr1_04","FAJSCCr1_05","FAJSCCr1_06","FAJSCCr1_08","FAJSCCr1_10"]

    model_name_list_r2 = ["FAJSCCr2_00","FAJSCCr2_02","FAJSCCr2_04","FAJSCCr2_05","FAJSCCr2_06","FAJSCCr2_08","FAJSCCr2_10"] 

    model_type_list_r1 = ["FAJSCCr1"]
    model_type_list_r2 = ["FAJSCCr2"]
    model_type_list_r12 = ["FAJSCCr12"]

         
    
    rcpp_list=[12]
    SNR_list=[1,10]
    
    total_model_list = model_name_list_r12 + model_name_list_r1 + model_name_list_r2
    
    
    total_eval_dict = get_total_eval_dict(cfg,logger,total_model_list,rcpp_list,SNR_list)
    
    
    save_GFlops_performance_ratio_plot(cfg,logger,total_eval_dict,model_name_list_r1,model_name_list_r12,model_name_list_r2,model_name_list3,model_type_list_r1,model_type_list_r12,model_type_list_r2,12,1,postfix="_")
    
    save_GFlops_performance_ratio_plot(cfg,logger,total_eval_dict,model_name_list_r1,model_name_list_r12,model_name_list_r2,model_name_list3,model_type_list_r1,model_type_list_r12,model_type_list_r2,12,10,postfix="_")    
    
    save_SNR_performance_table(cfg,logger,total_eval_dict,total_model_list,12,SNR_list,prefix="Ratio_")
    

    
if __name__ == '__main__':
    main()
    














    