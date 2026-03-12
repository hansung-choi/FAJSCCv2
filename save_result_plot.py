from data_maker import *
from loss_maker import *
from optimizer_maker import *
from train import *
from model.model_maker import *
from model_eval import *
from matplotlib.pyplot import cm
import csv
import random
import os
import gc

def get_model_save_name(cfg,model_name,rcpp,SNR):
    data = cfg.data_info.data_name
    cfg.model_name = model_name
    get_loss_info(cfg)
    get_task_info(cfg)
    task = cfg.task_name
    cfg.SNR_info = SNR
    chan_type = cfg.chan_type
    SNR = str(cfg.SNR_info).zfill(3)
    cfg.rcpp = rcpp
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    random_seed_num = cfg.random_seed
    random_num = str(random_seed_num).zfill(3)
    
    save_name = f"{task}_{data}_{chan_type}_SNR{SNR}_rcpp{rcpp}_{metric}_{model_name}.pt"


    return save_name    


def save_Mmemory_performance_plot_type(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_Mmemory_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    color_list = ['b-','r-','g-','c-','m-','y-','b--','r--','g--','c--','m--','y--']
    
    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_Mmemory_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mmemory = eval_dict['Mmemory']
            valid_Mmemory_list.append(Mmemory)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_Mmemory_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]}',marker='o',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_Mmemory_list = []
            valid_performance_list = []    
   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('Memory (MB)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})
    ax1.grid(True)

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_Mparams_performance_plot(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_Mparams_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    color_list = ['b-','r-','g-','c-','m-','y-','b--','r--','g--','c--','m--','y--'] 
    
    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_Mparams_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mparams = eval_dict['Mparams']
            valid_Mparams_list.append(Mparams)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_Mparams_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]}',marker='o',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_Mparams_list = []
            valid_performance_list = []    


    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('Params (M)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})
    ax1.grid(True)

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_GFlops_performance_plot(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_GFlops_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    color_list = ['b-','r-','g-','c-','m-','y-','b--','r--','g--','c--','m--','y--']  
    
    plt.rcParams["figure.figsize"] = (14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            if cfg.test_data == "DIV2K":
                key = f"{1536}x{2048}" 
                GFlops = eval_dict[f"GFlops_{key}"]
            else:
                key = f"{512}x{768}"
                GFlops = eval_dict[f"GFlops_{key}"]
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]}',marker='o',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('GFLOPs', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})
    ax1.grid(True)

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')
    
    
    
def save_GFlops_performance_ratio_plot(cfg,logger,total_eval_dict,encoder_side_list,both_side_list,decoder_side_list,fixed_model_list,encoder_type_list,both_type_list,decoder_type_list,rcpp,SNR,postfix=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_GFlops_{metric}_ratio_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if postfix:
        plot_save_name +=postfix
    if len(encoder_side_list)==0:
        return None
    
    for model_type in encoder_type_list:
        plot_save_name += "_" + model_type

    color_list = ['b-','r-','g-','c-','m-','y-','b--','r--','g--','c--','m--','y--'] 
    
    encoder_color_list = ['b^--','r^--']
    both_color_list = ['bd-.','rd-.']
    decoder_color_list = ['bx:','rx:']
    fixed_color_list = ['go-','co-','mo-','ko-']
    
    
    plt.rcParams["figure.figsize"] = (20,8)
    
    fig, ax1 = plt.subplots()
    line_list = []

    num_size = len(encoder_side_list)//len(encoder_type_list)
    th = 0
    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(encoder_side_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            if cfg.test_data == "DIV2K":
                key = f"{1536}x{2048}" 
                GFlops = eval_dict[f"GFlops_{key}"]
            else:
                key = f"{512}x{768}"
                GFlops = eval_dict[f"GFlops_{key}"]

            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,encoder_color_list[m_type_index],label=f'{encoder_type_list[m_type_index]}',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    


    num_size = len(both_side_list)//len(both_type_list)
    th = 0
    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(both_side_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            if cfg.test_data == "DIV2K":
                key = f"{1536}x{2048}" 
                GFlops = eval_dict[f"GFlops_{key}"]
            else:
                key = f"{512}x{768}"
                GFlops = eval_dict[f"GFlops_{key}"]
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,both_color_list[m_type_index],label=f'{both_type_list[m_type_index]}',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    


    num_size = len(decoder_side_list)//len(decoder_type_list)
    th = 0
    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(decoder_side_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            if cfg.test_data == "DIV2K":
                key = f"{1536}x{2048}" 
                GFlops = eval_dict[f"GFlops_{key}"]
            else:
                key = f"{512}x{768}"
                GFlops = eval_dict[f"GFlops_{key}"]
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,decoder_color_list[m_type_index],label=f'{decoder_type_list[m_type_index]}',linewidth=3,markersize=6)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    


    m_type_index = 0    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(fixed_model_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            if cfg.test_data == "DIV2K":
                key = f"{1536}x{2048}" 
                GFlops = eval_dict[f"GFlops_{key}"]
            else:
                key = f"{512}x{768}"
                GFlops = eval_dict[f"GFlops_{key}"]
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
                  
        line = ax1.plot(valid_GFlops_list, valid_performance_list,fixed_color_list[m_type_index],label=f'{fixed_model_list[m_type_index]}',linewidth=3,markersize=6)
        line_list.append(line)
        m_type_index +=1
        valid_GFlops_list = []
        valid_performance_list = []    

    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('GFLOPs', fontsize=20) #'GFLOPs', 'GFlops'
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict = {'fontsize' : 20})
    ax1.grid(True)
    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')    
    
    
    
    
    
    
def save_performance_GFlops_Mmemory_plot(cfg,logger,total_eval_dict,model_name_list,model_type_list,rcpp,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_GFLOPs_Mmemory_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    color_list = ['bo-','ro-','go-','co-','mo-','yo-','bo--','ro--','go--','co--','mo--','yo--']
    
    plt.rcParams["figure.figsize"] = (20,8) #(14,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    valid_GFlops_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            GFlops = eval_dict['GFlops']
            if cfg.test_data == "DIV2K":
                key = f"{1536}x{2048}" 
                GFlops = eval_dict[f"GFlops_{key}"]
            else:
                key = f"{512}x{768}"
                GFlops = eval_dict[f"GFlops_{key}"]
            valid_GFlops_list.append(GFlops)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax1.plot(valid_GFlops_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]} Gflops',linewidth=3,markersize=6,zorder=i)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_GFlops_list = []
            valid_performance_list = []    
        

    ax1.grid(True)
    ax1.set_xlabel('GFLOPs', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)
    
    
    
    num_size = len(model_name_list)//len(model_type_list)
    th = 0
    m_type_index = 0

    color_list = ['bd--','rd--','gd--','cd--','md--']
    
    
    ax2 = ax1.twiny() # ax1.twiny():use same y axis, ax1.twinx():use same x axis

    
    valid_Mmemory_list = []
    valid_performance_list = []    
    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
        eval_dict = total_eval_dict[model_save_name]            
            
        if eval_dict:
            Mmemory = eval_dict['Mmemory']
            valid_Mmemory_list.append(Mmemory)
            performance = eval_dict[metric]
            valid_performance_list.append(performance)
        
        th +=1            
        if th >= num_size:
            line = ax2.plot(valid_Mmemory_list, valid_performance_list,color_list[m_type_index],label=f'{model_type_list[m_type_index]} Memory',linewidth=4,markersize=6,zorder=i+num_size)
            line_list.append(line)
            m_type_index +=1
            th = 0
            valid_Mmemory_list = []
            valid_performance_list = []    

    lines = []
    for line in line_list:
        lines += line


    ax2.set_xlabel('Memory (MB)', fontsize=20)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB\n', fontdict = {'fontsize' : 20})

    
    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')

    
    

def save_SNR_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,prefix=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if prefix:
        plot_save_name = prefix + plot_save_name
    
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


 
    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']
    color_list = ['b-','r-','g-','c-','m-','y-','k-','b--','r--','g--','c--','m--','y--','k--']
    
    plt.rcParams["figure.figsize"] = (12,4) 
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    
    for i, model_name in enumerate(model_name_list):
        valid_SNR_list = []
        valid_performance_list = []
        for SNR in SNR_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]            
            
            if eval_dict:
                valid_SNR_list.append(SNR)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        
        line = ax1.plot(valid_SNR_list, valid_performance_list,color_list[i],label=f'{model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)
        logger.info(f'model_name: {model_name}')
        logger.info(f'valid_SNR_list: {valid_SNR_list}')
        logger.info(f'valid_performance_list: {valid_performance_list}')   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('SNR (dB)', fontsize=11) #try 10,11,12
    ax1.set_ylabel(metric, fontsize=11)
    plt.xticks( fontsize=10)
    plt.yticks(fontsize=10)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}', fontdict = {'fontsize' : 11})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')



def save_CPP_performance_plot(cfg, logger, total_eval_dict, model_name_list, rcpp_list, SNR, prefix=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_CPP_{metric}_at_SNR{str(SNR).zfill(2)}dB"
    if prefix:
        plot_save_name = prefix + plot_save_name

    if len(model_name_list) == 0:
        return None

    for model_name in model_name_list:
        plot_save_name += "_" + model_name

    color_list = ['b-','r-','g-','c-','m-','y-','k-','b--','r--','g--','c--','m--','y--','k--']

    plt.rcParams["figure.figsize"] = (12,4)

    fig, ax1 = plt.subplots()
    line_list = []

    for i, model_name in enumerate(model_name_list):
        valid_cpp_list = []
        valid_performance_list = []
        for rcpp in rcpp_list:
            model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)
            eval_dict = total_eval_dict[model_save_name]

            if eval_dict:
                cpp = 1.0 / rcpp
                valid_cpp_list.append(cpp)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)

        line = ax1.plot(
            valid_cpp_list,
            valid_performance_list,
            color_list[i],
            label=f'{cfg.model_name}',
            marker='o',
            linewidth=3,
            markersize=6
        )
        line_list.append(line)

    lines = []
    for line in line_list:
        lines += line

    ax1.set_xlabel('CPP (= 1/RCPP)', fontsize=11)
    ax1.set_ylabel(metric, fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, SNR={SNR} dB', fontdict={'fontsize': 11})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')




def save_rcpp_performance_plot(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{chan_type}_rcpp_{metric}_at_SNR{str(SNR).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        plot_save_name += "_" + model_name


    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']
    color_list = ['b-','r-','g-','c-','m-','y-','k-','b--','r--','g--','c--','m--','y--','k--']
    
    plt.rcParams["figure.figsize"] = (12,4) 
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        valid_rcpp_list = []
        valid_performance_list = []
        for rcpp in rcpp_list:
            model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
            eval_dict = total_eval_dict[model_save_name]
            if eval_dict:
                valid_rcpp_list.append(rcpp)
                performance = eval_dict[metric]
                valid_performance_list.append(performance)
        valid_cpp_list = 1/np.array(valid_rcpp_list)        
        line = ax1.plot(valid_cpp_list, valid_performance_list,color_list[i],label=f'{cfg.model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('CPP', fontsize=10)
    ax1.set_ylabel(metric, fontsize=10)
    plt.xticks( fontsize=10)
    plt.yticks(fontsize=10)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0,0,0.6,0.8])
    plt.title(f'{cfg.chan_type}, SNR={SNR}dB', fontdict = {'fontsize' : 10})

    save_name = save_folder + plot_save_name + ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')




def save_SNR_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp,SNR_list,prefix=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    table_save_name = f"{cfg.test_data}_{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}"
    if prefix:
        table_save_name = prefix + table_save_name

    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        if len(table_save_name) <=150:
            table_save_name += "_" + model_name

    save_name = save_folder + table_save_name + ".csv"
    
    
    first_line = ["rcpp",rcpp,"metric",metric]
    second_line = ["SNR"]
    second_line.extend(SNR_list)
    second_line.append('Model Storage Size')
    second_line.append('ms/image')
    
    
    with open(save_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
    
        for i, model_name in enumerate(model_name_list):
            valid_performance_list = [model_name]
            for SNR in SNR_list:
                model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
                eval_dict = total_eval_dict[model_save_name]            
            
                if eval_dict:
                    performance = eval_dict[metric]
                    Mmemory = eval_dict['Mmemory']
                    latency = eval_dict['ms/image']
                    valid_performance_list.append(performance)
                else:
                    valid_performance_list.append("None")
            valid_performance_list.append(Mmemory)
            valid_performance_list.append(latency)
            writer.writerow(valid_performance_list)
                    
    f.close()  
        
    logger.info(f'{table_save_name} is saved')

def save_CPP_performance_table(cfg, logger, total_eval_dict, model_name_list, rcpp_list, SNR, prefix=None):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    table_save_name = f"{cfg.test_data}_{chan_type}_CPP_{metric}_at_SNR{str(SNR).zfill(2)}dB"
    if prefix:
        table_save_name = prefix + table_save_name

    if len(model_name_list) == 0:
        return None

    for model_name in model_name_list:
        if len(table_save_name) <= 150:
            table_save_name += "_" + model_name

    save_name = save_folder + table_save_name + ".csv"

    first_line = ["SNR", SNR, "metric", metric]
    second_line = ["CPP"]
    # CPP = 1/RCPP
    second_line.extend([1.0 / rcpp for rcpp in rcpp_list])
    second_line.append('Model Storage Size')
    second_line.append('ms/image')

    with open(save_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)

        for i, model_name in enumerate(model_name_list):
            valid_performance_list = [model_name]

            GFlops = None
            Mmemory = None
            latency = None

            for rcpp in rcpp_list:
                model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)
                eval_dict = total_eval_dict[model_save_name]

                if eval_dict:
                    performance = eval_dict[metric]
                    Mmemory = eval_dict['Mmemory']
                    latency = eval_dict['ms/image']
                    valid_performance_list.append(performance)
                else:
                    valid_performance_list.append("None")
            valid_performance_list.append(Mmemory)
            valid_performance_list.append(latency)
            writer.writerow(valid_performance_list)

    f.close()

    logger.info(f'{table_save_name} is saved')
    
def save_rcpp_performance_table(cfg,logger,total_eval_dict,model_name_list,rcpp_list,SNR):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    table_save_name = f"{chan_type}_rcpp_{metric}_at_SNR{str(SNR).zfill(3)}"
    if len(model_name_list)==0:
        return None
    
    for model_name in model_name_list:
        table_save_name += "_" + model_name

    save_name = save_folder + table_save_name + ".csv"
    
    
    first_line = ["SNR",SNR,"metric",metric]
    second_line = ["rcpp"]
    second_line.extend(rcpp_list)
    second_line.append('GFlops')
    second_line.append('Mmemory')
    
    
    with open(save_name,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
    
        for i, model_name in enumerate(model_name_list):
            valid_performance_list = [model_name]
            for rcpp in rcpp_list:
                model_save_name = get_model_save_name(cfg,model_name,rcpp,SNR)
                eval_dict = total_eval_dict[model_save_name]            
            
                if eval_dict:
                    performance = eval_dict[metric]
                    GFlops = eval_dict['GFlops']
                    Mmemory = eval_dict['Mmemory']
                    valid_performance_list.append(performance)
                else:
                    valid_performance_list.append("None")
            valid_performance_list.append(GFlops)
            valid_performance_list.append(Mmemory)
            writer.writerow(valid_performance_list)
                    
    f.close()  
        
    logger.info(f'{table_save_name} is saved')

def save_gflops_memory_table(
    cfg,
    logger,
    total_eval_dict,
    model_name_list,
    rcpp,
    SNR,   
    prefix= None,
    resolutions=((512,768),(1536,2048)),
    gflops_digits=4,
    mem_digits=4,
):

    save_folder = "../../test_results/"
    os.makedirs(save_folder, exist_ok=True)

    chan_type = getattr(cfg, "chan_type", "chan")
    table_save_name = f"Complexity_{cfg.test_data}_{chan_type}_GFLOPs_PeakMem_at_rcpp{str(rcpp).zfill(3)}_SNR{SNR}"
    if prefix:
        table_save_name = prefix + table_save_name

    save_path = os.path.join(save_folder, table_save_name + ".csv")

    # Header rows (2 lines)
    header1 = ["Resolution"]
    for H, W in resolutions:
        header1 += [f"{H}x{W}", "", ""]  

    header2 = ["Metric"]
    for _ in resolutions:
        header2 += ["GFLOPs", "PeakMemory(MB)"]

    def fmt(x, ndigits):
        if x is None:
            return "None"
        try:
            return f"{float(x):.{ndigits}f}"
        except Exception:
            return "None"

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rcpp", rcpp, "SNR", SNR])   
        writer.writerow(header1)
        writer.writerow(header2)

        for model_name in model_name_list:
            model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

            eval_dict = total_eval_dict.get(model_save_name, None)


            row = [model_name]

            for H, W in resolutions:
                key = f"{H}x{W}"

                if eval_dict:
                    gfl = eval_dict.get(f"GFlops_{key}", None)
                    mem = eval_dict.get(f"max_memory_MB_{key}", None)

                row.append(fmt(gfl, gflops_digits))
                row.append(fmt(mem, mem_digits))

            writer.writerow(row)

    logger.info(f"{table_save_name} is saved: {save_path}")
    return save_path
    
    

def save_plot_legend_type1(cfg,logger,model_name_list,plot_name=None,ncol=5):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"plot_legend"
    if len(model_name_list)==0:
        return None
    
    if not plot_name:
        for model_name in model_name_list:
            plot_save_name += "_" + model_name
    else:
        plot_save_name +=plot_name

    color_list = ['b-','r-','g-','c-','m-','b--','r--','g--','c--','m--']
    color_list = ['b-','r-','g-','c-','m-','y-','k-','b--','r--','g--','c--','m--','y--','k--']


    plt.rcParams["figure.figsize"] = (20,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        line = ax1.plot([10], [10],color_list[i],label=f'{model_name}',marker='o',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('...', fontsize=18)
    ax1.set_ylabel(metric, fontsize=15)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(loc='lower right', ncol=ncol, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0,0,0.6,0.8])

    save_name = save_folder + plot_save_name + f'ncol{ncol}' + "_type1.pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')



def save_plot_legend_type2(cfg,logger,model_name_list,plot_name=None,ncol=5):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"plot_legend"
    if len(model_name_list)==0:
        return None
    
    if not plot_name:
        for model_name in model_name_list:
            plot_save_name += "_" + model_name
    else:
        plot_save_name +=plot_name

    color_list = ['bo-','ro-','go-','co-','mo-','bd--','rd--','gd--','cd--','md--']

    plt.rcParams["figure.figsize"] = (20,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        line = ax1.plot([10], [10],color_list[i],label=f'{model_name}',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('...', fontsize=18)
    ax1.set_ylabel(metric, fontsize=15)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(loc='lower right', ncol=ncol, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0,0,0.6,0.8])


    save_name = save_folder + plot_save_name + f'ncol{ncol}' + "_type3.pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_plot_legend_ratio(cfg,logger,model_name_list,plot_name=None,ncol=5):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"plot_legend"
    if len(model_name_list)==0:
        return None
    
    if not plot_name:
        for model_name in model_name_list:
            plot_save_name += "_" + model_name
    else:
        plot_save_name +=plot_name

    encoder_color_list = ['b^--','r^--']
    both_color_list = ['bd-.','rd-.']
    decoder_color_list = ['bx:','rx:']
    fixed_color_list = ['go-','co-','mo-']
    color_list = ['b^--','r^--','bd-.','rd-.','bx:','rx:','go-','co-','mo-']


    plt.rcParams["figure.figsize"] = (30,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        line = ax1.plot([10], [10],color_list[i],label=f'{model_name}',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('...', fontsize=18)
    ax1.set_ylabel(metric, fontsize=15)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(loc='lower right', ncol=ncol, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0,0,0.6,0.8])

    save_name = save_folder + plot_save_name + f'_ncol{ncol}'+ ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')


def save_plot_legend_ratio_special(cfg,logger,plot_name="special",ncol=5):
    model_name_list1 = ["baseFAJSCC control $ \gamma_{e} $","baseFAJSCC w/o SI control $ \gamma_{e} $"]
    model_name_list2 = ["baseFAJSCC control $ \gamma_{e},\gamma_{d} $","baseFAJSCC control $ \gamma_{e},\gamma_{d} $"]
    model_name_list3 = ["baseFAJSCC control $ \gamma_{d} $","baseFAJSCC w/o SI control $ \gamma_{d} $","baseSwinJSCC"]
    
    model_name_list1 = ["baseFAJSCC varying $ \gamma_{e} $ only","baseFAJSCC w/o SI varying $ \gamma_{e}$ only"]
    model_name_list2 = ["baseFAJSCC varying $ \gamma_{e},\gamma_{d} $","baseFAJSCC varying $ \gamma_{e},\gamma_{d} $"]
    model_name_list3 = ["baseFAJSCC varying $ \gamma_{d}$ only","baseFAJSCC w/o SI varying $ \gamma_{d}$ only","baseSwinJSCC"]

    model_name_list1 = ["baseFAJSCC varying $ \gamma_{e} $","baseFAJSCC w/o SI varying $ \gamma_{e}$"]
    model_name_list2 = ["baseFAJSCC $ \gamma_{e}=\gamma_{d}$","baseFAJSCC $ \gamma_{e}=\gamma_{d} $"]
    model_name_list3 = ["baseFAJSCC varying $ \gamma_{d}$","baseFAJSCC w/o SI varying $ \gamma_{d}$","baseSwinJSCC"]


    
    model_name_list1 = ["baseFAJSCC w/o SA varying $ \gamma_{e} $","baseFAJSCC w/ SA varying $ \gamma_{e}$"]
    model_name_list2 = ["baseFAJSCC w/o SA $ \gamma_{e}=\gamma_{d}$","baseFAJSCC w/ SA $ \gamma_{e}=\gamma_{d} $"]
    model_name_list3 = ["baseFAJSCC w/o SA varying $ \gamma_{d}$","baseFAJSCC w/ SA varying $ \gamma_{d}$","baseSwinJSCC"]

    
    model_name_list = model_name_list1 + model_name_list2 + model_name_list3
    
    model_name_list = ["FAJSCC varying $ \gamma_{e} $","FAJSCC $ \gamma_{e}=\gamma_{d}$","FAJSCC varying $ \gamma_{d}$"]

    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"plot_legend"
    if len(model_name_list)==0:
        return None
    
    if not plot_name:
        for model_name in model_name_list:
            plot_save_name += "_" + model_name
    else:
        plot_save_name +=plot_name

    encoder_color_list = ['b^--','r^--']
    both_color_list = ['bd-.','rd-.']
    decoder_color_list = ['bx:','rx:']
    fixed_color_list = ['go-','co-','mo-']
    color_list = ['b^--','r^--','bd-.','rd-.','bx:','rx:','go-','co-','mo-']
    color_list = ['b^--','bd-.','bx:']
    

    plt.rcParams["figure.figsize"] = (30,8)
    
    fig, ax1 = plt.subplots()
    line_list = []
    
    for i, model_name in enumerate(model_name_list):
        line = ax1.plot([10], [10],color_list[i],label=f'{model_name}',linewidth=3,markersize=6)
        line_list.append(line)   
        
    lines = []
    for line in line_list:
        lines += line


    ax1.set_xlabel('...', fontsize=18)
    ax1.set_ylabel(metric, fontsize=15)
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(loc='lower right', ncol=ncol, bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0,0,0.6,0.8])

    save_name = save_folder + plot_save_name + f'_ncol{ncol}'+ ".pdf"
    if save_name:
        plt.savefig(save_name)
    plt.clf()
    logger.info(f'{plot_save_name} is saved')

# -------------------------
# Helpers
# -------------------------
def _collect_stats(total_eval_dict_list, model_save_name, key):
    """
    Collect values for (model_save_name, key) across seeds.
    Ignore missing/None eval_dict or missing/None key.
    Return (mean, std, n).
    """
    vals = []
    for d in total_eval_dict_list:
        eval_dict = d.get(model_save_name, None)
        if not eval_dict:
            continue
        v = eval_dict.get(key, None)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            pass

    if len(vals) == 0:
        return (None, None, 0)

    arr = np.array(vals, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))  # population std; change to ddof=1 if you want sample std
    return (mean, std, len(vals))


def _fmt_mean_std(mean, std, ndigits=4):
    if mean is None or std is None:
        return "None"
    return f"({mean:.{ndigits}f}, {std:.{ndigits}f})"
    



# =========================================================
# 1) Plot: SNR vs metric (mean with std error bars)
# =========================================================
def save_SNR_performance_plot_meanstd(
    cfg,
    logger,
    total_eval_dict_list,
    model_name_list,
    rcpp,
    SNR_list,
    prefix=None,
):
    save_folder = "../../test_results/"
    os.makedirs(save_folder, exist_ok=True)

    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}_AvgStd"
    if prefix:
        plot_save_name = prefix + plot_save_name

    if len(model_name_list) == 0:
        return None

    #for model_name in model_name_list:
        #plot_save_name += "_" + model_name

    plt.rcParams["figure.figsize"] = (12, 4)
    fig, ax1 = plt.subplots()

    # Keep your original style list (cycled if needed)
    style_list = ['b--','r--','g--','c--','m--','y--','k--','b--','r--','g--','c--','m--','y--','k--']

    for i, model_name in enumerate(model_name_list):
        valid_SNR_list = []
        mean_list = []
        std_list = []

        for SNR in SNR_list:
            model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)
            mean, std, n = _collect_stats(total_eval_dict_list, model_save_name, metric)

            if n > 0:
                valid_SNR_list.append(SNR)
                mean_list.append(mean)
                std_list.append(std)

        logger.info(f"[AvgStd] model_name: {model_name}")
        logger.info(f"[AvgStd] valid_SNR_list: {valid_SNR_list}")
        logger.info(f"[AvgStd] mean_list: {mean_list}")
        logger.info(f"[AvgStd] std_list: {std_list}")

        if len(valid_SNR_list) == 0:
            continue

        style = style_list[i % len(style_list)]
        color = style[0]  # 'b','r',... (matplotlib default color codes)
        linestyle = style[1:] if len(style) > 1 else '-'

        # Mean with std error bars
        ax1.errorbar(
            valid_SNR_list,
            mean_list,
            yerr=std_list,
            fmt='o',
            color=color,
            linestyle=linestyle,
            linewidth=3,
            markersize=6,
            capsize=4,
            label=f"{model_name}",
        )

    ax1.set_xlabel("SNR (dB)", fontsize=11)
    ax1.set_ylabel(metric, fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    ax1.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.6, 0.8])
    plt.title(f"{cfg.chan_type}, CPP=1/{rcpp}", fontdict={'fontsize': 11})

    save_path = os.path.join(save_folder, plot_save_name + ".pdf")
    plt.savefig(save_path)
    plt.clf()

    logger.info(f"{plot_save_name} is saved: {save_path}")
    return save_path


# =========================================================
# 2) Table: SNR vs metric + (GFlops, Mmemory, ms/image)
#     Each cell is "(mean, std)"
# =========================================================
def save_SNR_performance_table_meanstd(
    cfg,
    logger,
    total_eval_dict_list,
    model_name_list,
    rcpp,
    SNR_list,
    prefix=None,
    ndigits_metric=4,
    ndigits_complexity=4,
):
    save_folder = "../../test_results/"
    os.makedirs(save_folder, exist_ok=True)

    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    table_save_name = f"{cfg.test_data}_{chan_type}_SNR_{metric}_at_rcpp{str(rcpp).zfill(3)}_AvgStd"
    if prefix:
        table_save_name = prefix + table_save_name

    if len(model_name_list) == 0:
        return None

    #for model_name in model_name_list:
        #if len(table_save_name) <= 150:
            #table_save_name += "_" + model_name

    save_path = os.path.join(save_folder, table_save_name + ".csv")

    first_line = ["rcpp", rcpp, "metric", metric, "stat", "(mean, std)"]
    second_line = ["SNR"]
    second_line.extend(SNR_list)
    second_line.append("Model Storage Size")
    second_line.append("ms/image")

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)

        for model_name in model_name_list:
            row = [model_name]

            # Per-SNR metric
            for SNR in SNR_list:
                model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)
                mean, std, n = _collect_stats(total_eval_dict_list, model_save_name, metric)
                row.append(_fmt_mean_std(mean, std, ndigits_metric))

            # Complexity keys (same as your original naming)
            rep_SNR = SNR_list[0] if len(SNR_list) > 0 else None

            if rep_SNR is not None:
                model_save_name = get_model_save_name(cfg, model_name, rcpp, rep_SNR)

                m_mean, m_std, _ = _collect_stats(total_eval_dict_list, model_save_name, "Mmemory")
                l_mean, l_std, _ = _collect_stats(total_eval_dict_list, model_save_name, "ms/image")
                
                row.append(_fmt_mean_std(m_mean, m_std, ndigits_complexity))
                row.append(_fmt_mean_std(l_mean, l_std, ndigits_complexity))
            else:
                row += ["None", "None", "None"]

            writer.writerow(row)

    logger.info(f"{table_save_name} is saved: {save_path}")
    return save_path


# =========================================================
# 3) Complexity table by resolution: Latency/GFLOPs/PeakMem
#     Each cell is "(mean, std)"
# =========================================================
def save_gflops_memory_table_meanstd(
    cfg,
    logger,
    total_eval_dict_list,
    model_name_list,
    rcpp,
    SNR,
    prefix=None,
    resolutions=((512, 768), (1536, 2048)),
    gflops_digits=4,
    mem_digits=4,
):
    save_folder = "../../test_results/"
    os.makedirs(save_folder, exist_ok=True)

    chan_type = getattr(cfg, "chan_type", "chan")
    table_save_name = (
        f"Complexity_{cfg.test_data}_{chan_type}_GFLOPs_PeakMem_"
        f"at_rcpp{str(rcpp).zfill(3)}_SNR{SNR}_AvgStd"
    )
    if prefix:
        table_save_name = prefix + table_save_name

    save_path = os.path.join(save_folder, table_save_name + ".csv")

    # Header rows
    header1 = ["Resolution"]
    for H, W in resolutions:
        header1 += [f"{H}x{W}", "", ""]

    header2 = ["Metric"]
    for _ in resolutions:
        header2 += ["GFLOPs", "PeakMemory(MB)"]

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rcpp", rcpp, "SNR", SNR, "stat", "(mean, std)"])
        writer.writerow(header1)
        writer.writerow(header2)

        for model_name in model_name_list:
            model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

            row = [model_name]
            for H, W in resolutions:
                key = f"{H}x{W}"

                gfl_key = f"GFlops_{key}"
                mem_key = f"max_memory_MB_{key}"

                gfl_mean, gfl_std, _ = _collect_stats(total_eval_dict_list, model_save_name, gfl_key)
                mem_mean, mem_std, _ = _collect_stats(total_eval_dict_list, model_save_name, mem_key)

                row.append(_fmt_mean_std(gfl_mean, gfl_std, gflops_digits))
                row.append(_fmt_mean_std(mem_mean, mem_std, mem_digits))

            writer.writerow(row)

    logger.info(f"{table_save_name} is saved: {save_path}")
    return save_path

def _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, key):
    vals = []

    for total_eval_dict in total_eval_dict_list:
        eval_dict = total_eval_dict.get(model_save_name, None)

        if not eval_dict:
            continue

        v = eval_dict.get(key, None)

        if v is None:
            continue

        try:
            vals.append(float(v))
        except Exception:
            pass

    if len(vals) == 0:
        return None, None, 0

    arr = np.array(vals, dtype=np.float64)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))

    return mean, std, len(vals)
    
        
def save_GFlops_performance_plot_meanstd(
    cfg,
    logger,
    total_eval_dict_list,
    model_name_list,
    model_type_list,
    rcpp,
    SNR,
    prefix=None,
    ):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_GFlops_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}_AvgStd"
    if prefix:
        plot_save_name = prefix + plot_save_name

    if len(model_name_list) == 0:
        return None

    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list) // len(model_type_list)
    th = 0
    m_type_index = 0

    color_list = ['b-', 'r-', 'g-', 'c-', 'm-', 'y-', 'b--', 'r--', 'g--', 'c--', 'm--', 'y--']

    plt.rcParams["figure.figsize"] = (10, 8)
    #plt.rcParams["figure.figsize"] = (3.5, 2.4)

    fig, ax1 = plt.subplots()

    valid_gflops_mean_list = []
    valid_gflops_std_list = []
    valid_perf_mean_list = []
    valid_perf_std_list = []

    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

        if cfg.test_data == "DIV2K":
            key = f"{1536}x{2048}"
            gflops_key = f"GFlops_{key}"
        else:
            key = f"{512}x{768}"
            gflops_key = f"GFlops_{key}"

        gflops_mean, gflops_std, gflops_n = _collect_stats_from_dict_list(
            total_eval_dict_list, model_save_name, gflops_key
        )
        perf_mean, perf_std, perf_n = _collect_stats_from_dict_list(
            total_eval_dict_list, model_save_name, metric
        )

        if gflops_n > 0 and perf_n > 0:
            valid_gflops_mean_list.append(gflops_mean)
            valid_gflops_std_list.append(gflops_std)
            valid_perf_mean_list.append(perf_mean)
            valid_perf_std_list.append(perf_std)

        th += 1
        if th >= num_size:
            style = color_list[m_type_index % len(color_list)]
            color = style[0]
            linestyle = style[1:] if len(style) > 1 else '-'

            if len(valid_gflops_mean_list) > 0:
                ax1.errorbar(
                    valid_gflops_mean_list,
                    valid_perf_mean_list,
                    yerr=valid_perf_std_list,
                    fmt='o',
                    color=color,
                    linestyle=linestyle,
                    label=f'{model_type_list[m_type_index]}',
                    linewidth=3,
                    markersize=6,
                    capsize=4,
                )

            logger.info(f"[AvgStd] model_type: {model_type_list[m_type_index]}")
            logger.info(f"[AvgStd] valid_gflops_mean_list: {valid_gflops_mean_list}")
            logger.info(f"[AvgStd] valid_gflops_std_list: {valid_gflops_std_list}")
            logger.info(f"[AvgStd] valid_perf_mean_list: {valid_perf_mean_list}")
            logger.info(f"[AvgStd] valid_perf_std_list: {valid_perf_std_list}")

            m_type_index += 1
            th = 0
            valid_gflops_mean_list = []
            valid_gflops_std_list = []
            valid_perf_mean_list = []
            valid_perf_std_list = []

    ax1.set_xlabel('GFLOPs', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ncol = min(len(model_type_list), 6)
    ax1.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.12),
    ncol=len(model_type_list),
    fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict={'fontsize': 20})
    ax1.grid(True)

    save_name = save_folder + plot_save_name + ".pdf"
    plt.savefig(save_name)
    plt.clf()

    logger.info(f'{plot_save_name} is saved')
    return save_name    




def save_Mmemory_performance_plot_meanstd(
    cfg,
    logger,
    total_eval_dict_list,
    model_name_list,
    model_type_list,
    rcpp,
    SNR,
    prefix=None,
):
    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    plot_save_name = f"{cfg.test_data}_{chan_type}_Mmemory_{metric}_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}_AvgStd"
    if prefix:
        plot_save_name = prefix + plot_save_name

    if len(model_name_list) == 0:
        return None

    for model_type in model_type_list:
        plot_save_name += "_" + model_type

    num_size = len(model_name_list) // len(model_type_list)
    th = 0
    m_type_index = 0

    color_list = ['b-', 'r-', 'g-', 'c-', 'm-', 'y-', 'b--', 'r--', 'g--', 'c--', 'm--', 'y--']

    #plt.rcParams["figure.figsize"] = (14, 8)
    plt.rcParams["figure.figsize"] = (10, 8)

    fig, ax1 = plt.subplots()

    valid_mem_mean_list = []
    valid_mem_std_list = []
    valid_perf_mean_list = []
    valid_perf_std_list = []

    for i, model_name in enumerate(model_name_list):
        model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

        mem_mean, mem_std, mem_n = _collect_stats_from_dict_list(
            total_eval_dict_list, model_save_name, "Mmemory"
        )
        perf_mean, perf_std, perf_n = _collect_stats_from_dict_list(
            total_eval_dict_list, model_save_name, metric
        )

        if mem_n > 0 and perf_n > 0:
            valid_mem_mean_list.append(mem_mean)
            valid_mem_std_list.append(mem_std)
            valid_perf_mean_list.append(perf_mean)
            valid_perf_std_list.append(perf_std)

        th += 1
        if th >= num_size:
            style = color_list[m_type_index % len(color_list)]
            color = style[0]
            linestyle = style[1:] if len(style) > 1 else '-'

            if len(valid_mem_mean_list) > 0:
                ax1.errorbar(
                    valid_mem_mean_list,
                    valid_perf_mean_list,
                    yerr=valid_perf_std_list,
                    fmt='o',
                    color=color,
                    linestyle=linestyle,
                    label=f'{model_type_list[m_type_index]}',
                    linewidth=3,
                    markersize=6,
                    capsize=4,
                )

            logger.info(f"[AvgStd] model_type: {model_type_list[m_type_index]}")
            logger.info(f"[AvgStd] valid_mem_mean_list: {valid_mem_mean_list}")
            logger.info(f"[AvgStd] valid_mem_std_list: {valid_mem_std_list}")
            logger.info(f"[AvgStd] valid_perf_mean_list: {valid_perf_mean_list}")
            logger.info(f"[AvgStd] valid_perf_std_list: {valid_perf_std_list}")

            m_type_index += 1
            th = 0
            valid_mem_mean_list = []
            valid_mem_std_list = []
            valid_perf_mean_list = []
            valid_perf_std_list = []

    ax1.set_xlabel('Memory (MB)', fontsize=20)
    ax1.set_ylabel(metric, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ncol = min(len(model_type_list), 6)
    ax1.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.12),
    ncol=len(model_type_list),
    fontsize=10
    )

    plt.tight_layout(rect=[0, 0, 0.98, 0.90])
    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontdict={'fontsize': 20})
    ax1.grid(True)

    save_name = save_folder + plot_save_name + ".pdf"
    plt.savefig(save_name)
    plt.clf()

    logger.info(f'{plot_save_name} is saved')
    return save_name




def save_GFlops_performance_ratio_plot_meanstd(
    cfg,
    logger,
    total_eval_dict_list,
    encoder_side_list,
    both_side_list,
    decoder_side_list,
    fixed_model_list,
    encoder_type_list,
    both_type_list,
    decoder_type_list,
    rcpp,
    SNR,
    postfix=None,
):

    save_folder = "../../test_results/"
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    chan_type = cfg.chan_type
    metric = cfg.performance_metric

    plot_save_name = f"{cfg.test_data}_{chan_type}_GFlops_{metric}_ratio_at_SNR{str(SNR).zfill(3)}_rcpp{str(rcpp).zfill(3)}_AvgStd"

    if postfix:
        plot_save_name += postfix

    if len(encoder_side_list) == 0:
        return None

    encoder_color_list = ['b^--','r^--']
    both_color_list = ['bd-.','rd-.']
    decoder_color_list = ['bx:','rx:']
    fixed_color_list = ['go-','co-','mo-','ko-']

    plt.rcParams["figure.figsize"] = (10, 8)

    fig, ax1 = plt.subplots()

    line_list = []

    ################################
    # encoder side
    ################################

    num_size = len(encoder_side_list)//len(encoder_type_list)

    th = 0
    m_type_index = 0

    x_mean = []
    y_mean = []
    y_std = []

    for model_name in encoder_side_list:

        model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

        if cfg.test_data == "DIV2K":
            key = f"{1536}x{2048}"
            gflops_key = f"GFlops_{key}"
        else:
            key = f"{512}x{768}"
            gflops_key = f"GFlops_{key}"

        g_mean, g_std, g_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, gflops_key)
        p_mean, p_std, p_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, metric)

        if g_n > 0 and p_n > 0:
            x_mean.append(g_mean)
            y_mean.append(p_mean)
            y_std.append(p_std)

        th += 1

        if th >= num_size:

            style = encoder_color_list[m_type_index]

            ax1.errorbar(
                x_mean,
                y_mean,
                yerr=y_std,
                fmt=style,
                linewidth=2,
                markersize=5,
                capsize=3,
                label=encoder_type_list[m_type_index]
            )

            m_type_index += 1
            th = 0

            x_mean = []
            y_mean = []
            y_std = []

    ################################
    # both side
    ################################

    num_size = len(both_side_list)//len(both_type_list)

    th = 0
    m_type_index = 0

    x_mean = []
    y_mean = []
    y_std = []

    for model_name in both_side_list:

        model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

        if cfg.test_data == "DIV2K":
            key = f"{1536}x{2048}"
            gflops_key = f"GFlops_{key}"
        else:
            key = f"{512}x{768}"
            gflops_key = f"GFlops_{key}"

        g_mean, g_std, g_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, gflops_key)
        p_mean, p_std, p_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, metric)

        if g_n > 0 and p_n > 0:
            x_mean.append(g_mean)
            y_mean.append(p_mean)
            y_std.append(p_std)

        th += 1

        if th >= num_size:

            style = both_color_list[m_type_index]

            ax1.errorbar(
                x_mean,
                y_mean,
                yerr=y_std,
                fmt=style,
                linewidth=2,
                markersize=5,
                capsize=3,
                label=both_type_list[m_type_index]
            )

            m_type_index += 1
            th = 0

            x_mean = []
            y_mean = []
            y_std = []

    ################################
    # decoder side
    ################################

    num_size = len(decoder_side_list)//len(decoder_type_list)

    th = 0
    m_type_index = 0

    x_mean = []
    y_mean = []
    y_std = []

    for model_name in decoder_side_list:

        model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

        if cfg.test_data == "DIV2K":
            key = f"{1536}x{2048}"
            gflops_key = f"GFlops_{key}"
        else:
            key = f"{512}x{768}"
            gflops_key = f"GFlops_{key}"

        g_mean, g_std, g_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, gflops_key)
        p_mean, p_std, p_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, metric)

        if g_n > 0 and p_n > 0:
            x_mean.append(g_mean)
            y_mean.append(p_mean)
            y_std.append(p_std)

        th += 1

        if th >= num_size:

            style = decoder_color_list[m_type_index]

            ax1.errorbar(
                x_mean,
                y_mean,
                yerr=y_std,
                fmt=style,
                linewidth=2,
                markersize=5,
                capsize=3,
                label=decoder_type_list[m_type_index]
            )

            m_type_index += 1
            th = 0

            x_mean = []
            y_mean = []
            y_std = []

    ################################
    # fixed models
    ################################

    for i, model_name in enumerate(fixed_model_list):

        model_save_name = get_model_save_name(cfg, model_name, rcpp, SNR)

        if cfg.test_data == "DIV2K":
            key = f"{1536}x{2048}"
            gflops_key = f"GFlops_{key}"
        else:
            key = f"{512}x{768}"
            gflops_key = f"GFlops_{key}"

        g_mean, g_std, g_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, gflops_key)
        p_mean, p_std, p_n = _collect_stats_from_dict_list(total_eval_dict_list, model_save_name, metric)

        if g_n > 0 and p_n > 0:

            ax1.errorbar(
                [g_mean],
                [p_mean],
                yerr=[p_std],
                fmt=fixed_color_list[i],
                linewidth=2,
                markersize=5,
                capsize=3,
                label=model_name
            )

    ################################

    ax1.set_xlabel('GFLOPs', fontsize=8)
    ax1.set_ylabel(metric, fontsize=8)

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    ax1.legend(
        loc='lower center',
        bbox_to_anchor=(0.5,1.15),
        ncol=3,
        fontsize=7
    )

    plt.title(f'{cfg.chan_type}, CPP=1/{rcpp}, SNR={SNR}dB', fontsize=9)

    ax1.grid(True)

    plt.tight_layout(rect=[0,0,1,0.9])

    save_name = save_folder + plot_save_name + ".pdf"

    plt.savefig(save_name)

    plt.clf()

    logger.info(f'{plot_save_name} is saved')






















