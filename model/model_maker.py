from .common_component import *
from .JSCC import *

def get_model_info(cfg):
    model_info = dict()
    model_info['chan_type'] = cfg.chan_type
    model_info['color_channel'] = cfg.data_info.color_channel
    model_info['rcpp'] = cfg.rcpp #reverse of channel per pixel
    model_info['window_size'] = 8
    model_info['ratio'] = 0.5
    cfg.ratio = model_info['ratio'] #importance ratio
    model_info['gamma'] = 0.5
    cfg.gamma = model_info['gamma']
    model_info['ratio1'] = 0.5
    cfg.ratio1 = model_info['ratio1'] # encoder's importance ratio
    model_info['ratio2'] = 0.5
    cfg.ratio2 = model_info['ratio2'] # decoder's importance ratio
    model_info['swap'] = 0.0

    model_info['window_size_list'] = [8,8,8,8] #[8,8,8,8]
    model_info['num_heads_list'] = [4, 6, 8, 10] ##careful! n_feats_list[i]/num_heads_list[i] should be integer
    model_info['input_resolution'] = cfg.input_resolution
    model_info['n_block_list'] = [2,2,2,2]



    if cfg.model_name in ["ConvJSCC","ConvJSCCfixSNR04","ConvJSCCfixSNR07","ConvJSCCrandomSNR"]:
        model_info['n_feats_list'] = [32,32,32,32]  
    elif cfg.model_name in ["ResJSCC","ResJSCCfixSNR04","ResJSCCfixSNR07","ResJSCCrandomSNR"]:
        model_info['n_feats_list'] = [32,32,32,32]
    elif cfg.model_name in ["SwinJSCC","SwinJSCCfixSNR04","SwinJSCCfixSNR07","SwinJSCCrandomSNR"]:
        model_info['n_feats_list'] = [40,60,80,160] 
    elif cfg.model_name in ["LICRFJSCC","LICRFJSCCfixSNR04","LICRFJSCCfixSNR07","LICRFJSCCrandomSNR"]:
        model_info['n_feats_list'] = [40,60,80,160]
    elif cfg.model_name in ["LAJSCC","LAJSCCfixSNR04","LAJSCCfixSNR07","LAJSCCrandomSNR"]:
        model_info['n_feats_list'] = [40,60,80,160]
    elif cfg.model_name in ["FAJSCC","FAJSCCfixSNR04","FAJSCCfixSNR07","FAJSCCrandomSNR"]:
        model_info['n_feats_list'] = [40,60,80,260]
    elif cfg.model_name in ["FAPGBJSCC","FAJSCCwoAT","FAJSCCwoLA","FAJSCCwoDf"]:
        model_info['n_feats_list'] = [40,60,80,260]
    elif cfg.model_name in ["LAFAJSCC"]:
        model_info['n_feats_list'] = [40,60,80,160]
        model_info['encoder_n_feats_list'] = [40,60,80,160]
        model_info['decoder_n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 0.7
    elif cfg.model_name in ["FALAJSCC"]:
        model_info['n_feats_list'] = [40,60,80,160]
        model_info['encoder_n_feats_list'] = [40,60,80,260]
        model_info['decoder_n_feats_list'] = [40,60,80,160]
        model_info['ratio1'] = 0.7
        
    elif cfg.model_name in ["HugeFAJSCC"]:
        model_info['n_feats_list'] = [128, 192, 256, 320]
        model_info['n_block_list'] = [2,2,6,2]


    elif cfg.model_name == "FAJSCCr12_00":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.0
        model_info['ratio2'] = 0.0
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2']         
    elif cfg.model_name == "FAJSCCr12_02":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.2
        model_info['ratio2'] = 0.2
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr12_04":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.4
        model_info['ratio2'] = 0.4
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr12_05":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.5
        model_info['ratio2'] = 0.5
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr12_06":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.6
        model_info['ratio2'] = 0.6
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr12_08":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.8
        model_info['ratio2'] = 0.8
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr12_10":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 1.0
        model_info['ratio2'] = 1.0
        cfg.ratio1 = model_info['ratio1']
        cfg.ratio2 = model_info['ratio2'] 
        
        
        
    elif cfg.model_name == "FAJSCCr1_00":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.0
        cfg.ratio1 = model_info['ratio1']       
    elif cfg.model_name == "FAJSCCr1_02":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.2
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "FAJSCCr1_04":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.4
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "FAJSCCr1_05":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.5
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "FAJSCCr1_06":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.6
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "FAJSCCr1_08":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 0.8
        cfg.ratio1 = model_info['ratio1']
    elif cfg.model_name == "FAJSCCr1_10":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio1'] = 1.0
        cfg.ratio1 = model_info['ratio1']
        
        
        
    elif cfg.model_name == "FAJSCCr2_00":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 0.0
        cfg.ratio2 = model_info['ratio2']         
    elif cfg.model_name == "FAJSCCr2_02":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 0.2
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr2_04":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 0.4
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr2_05":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 0.5
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr2_06":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 0.6
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr2_08":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 0.8
        cfg.ratio2 = model_info['ratio2'] 
    elif cfg.model_name == "FAJSCCr2_10":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['ratio2'] = 1.0
        cfg.ratio2 = model_info['ratio2'] 

    elif cfg.model_name == "largeConvJSCC":
        model_info['n_feats_list'] = [64,64,64,64]
    elif cfg.model_name == "largeResJSCC":
        model_info['n_feats_list'] = [64,64,64,64]
    elif cfg.model_name == "largeLICRFJSCC":
        model_info['n_feats_list'] = [64,96,128,180]
    elif cfg.model_name == "hugeConvJSCC":
        model_info['n_feats_list'] = [128,128,128,128]
    elif cfg.model_name == "hugeResJSCC":
        model_info['n_feats_list'] = [128,128,128,128]
    elif cfg.model_name == "hugeLICRFJSCC":
        model_info['n_feats_list'] = [80,120,160,240]    
    
        
    elif cfg.model_name == "smallSwinJSCC":
        model_info['n_feats_list'] = [28,42,64,120] 
    elif cfg.model_name == "smallLAJSCC":
        model_info['n_feats_list'] = [28,42,64,120] 
    elif cfg.model_name == "smallFAJSCC":
        model_info['n_feats_list'] = [28,42,64,200] 
    elif cfg.model_name == "largeSwinJSCC":
        model_info['n_feats_list'] = [60,90,120,200]
    elif cfg.model_name == "largeLAJSCC":
        model_info['n_feats_list'] = [60,90,120,200]
    elif cfg.model_name == "largeFAJSCC":
        model_info['n_feats_list'] = [60,90,120,360]


    elif cfg.model_name == "FAJSCCs03":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['swap'] = 0.3       
    elif cfg.model_name == "FAJSCCs06":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['swap'] = 0.6   
    elif cfg.model_name == "FAJSCCs10":
        model_info['n_feats_list'] = [40,60,80,260]
        model_info['swap'] = 1.0 

                
    else:
        raise ValueError(f'n_feats_list for {cfg.model_name} model is not implemented yet')

    return model_info


def ModelMaker(cfg):
    model = None
    model_info = dict()
    
    model_info = get_model_info(cfg)

    if cfg.model_name in ["ConvJSCC","ConvJSCCfixSNR04","ConvJSCCfixSNR07","ConvJSCCrandomSNR","largeConvJSCC","hugeConvJSCC"]:
        model = ConvJSCC(model_info)
    elif cfg.model_name in ["ResJSCC","ResJSCCfixSNR04","ResJSCCfixSNR07","ResJSCCrandomSNR","largeResJSCC","hugeResJSCC"]:
        model = ResJSCC(model_info)
    elif cfg.model_name in ["SwinJSCC","SwinJSCCfixSNR04","SwinJSCCfixSNR07","SwinJSCCrandomSNR","smallSwinJSCC","largeSwinJSCC"]:
        model = SwinJSCC(model_info)
    elif cfg.model_name in ["LICRFJSCC","LICRFJSCCfixSNR04","LICRFJSCCfixSNR07","LICRFJSCCrandomSNR","largeLICRFJSCC","hugeLICRFJSCC"]:
        model = LICRFJSCC(model_info)
    elif cfg.model_name in ["FAJSCC","FAJSCCfixSNR04","FAJSCCfixSNR07","FAJSCCrandomSNR","smallFAJSCC","largeFAJSCC"]:
        model = FAJSCC(model_info)
    elif cfg.model_name == "FAPGBJSCC":
        model = FAPGBJSCC(model_info)
    elif cfg.model_name == "FAJSCCwoAT":
        model = FAJSCCwoAT(model_info)
    elif cfg.model_name == "FAJSCCwoLA":
        model = FAJSCCwoLA(model_info)
    elif cfg.model_name == "FAJSCCwoDf":
        model = FAJSCCwoDf(model_info)
    elif cfg.model_name in ["LAJSCC","LAJSCCfixSNR04","LAJSCCfixSNR07","LAJSCCrandomSNR","smallLAJSCC","largeLAJSCC"]:
        model = LAJSCC(model_info)
    elif cfg.model_name in ["LAFAJSCC"]:
        model = LAFAJSCC(model_info)
    elif cfg.model_name in ["FALAJSCC"]:
        model = FALAJSCC(model_info)
    elif cfg.model_name in ["HugeFAJSCC"]:
        model = HugeFAJSCC(model_info)


    elif cfg.model_name in ["FAJSCCr12_00","FAJSCCr12_02","FAJSCCr12_04","FAJSCCr12_05","FAJSCCr12_06","FAJSCCr12_08","FAJSCCr12_10"]:
        model = FAJSCC(model_info)
    elif cfg.model_name in ["FAJSCCr1_00","FAJSCCr1_02","FAJSCCr1_04","FAJSCCr1_05","FAJSCCr1_06","FAJSCCr1_08","FAJSCCr1_10"]:
        model = FAJSCC(model_info)
    elif cfg.model_name in ["FAJSCCr2_00","FAJSCCr2_02","FAJSCCr2_04","FAJSCCr2_05","FAJSCCr2_06","FAJSCCr2_08","FAJSCCr2_10"]:
        model = FAJSCC(model_info)
    elif cfg.model_name in ["FAJSCCs03","FAJSCCs06","FAJSCCs10"]:
        model = FAJSCC(model_info)        
 
    else:
        raise ValueError(f'{cfg.model_name} model is not implemented yet')
    return model
    
    
 