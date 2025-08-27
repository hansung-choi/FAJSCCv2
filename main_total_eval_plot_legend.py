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

    model_name_list = ["FAJSCC (ours)","LAJSCC (ours)","LICRFJSCC","SwinJSCC","ResJSCC","ConvJSCC"]
    save_plot_legend_type1(cfg,logger,model_name_list,plot_name='_main_',ncol=6)
    
    save_plot_legend_ratio_special(cfg,logger,plot_name='_ratio_',ncol=3)

    
    
if __name__ == '__main__':
    main()
    














    