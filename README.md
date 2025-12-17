# FAJSCCv2

![Image](https://github.com/user-attachments/assets/89ad7e46-c1dc-4f51-9ebe-6b9440226829)
[dimension_specialized_operation.pdf](https://github.com/user-attachments/files/22002516/dimension_specialized_operation.pdf)


Implementations of main experiments for the paper "Feature Importance-Aware Deep Joint Source-Channel Coding for Computationally Efficient and Adjustable Image Transmission" 
This is an enhanced version compared to the previous FAJSCC (version 1 of arXiv) by applying axis dimension-specialized computation, selective deformable self-attention, and attention family tree frameworks detailed in the main paper.


## Requirements
1. python 3.8.8
2. pytorch 2.0.1
3. cuda 11.1.1
4. numpy 1.24.4
5. hydra 1.1

## Experiment code manual

### Arguments for terminal execution
1. **chan_type**: The type of communication channel, which can be one of **"AWGN", "Rayleigh"**.
2. **rcpp**: The reciprocal of **cpp** (channel usage per RGB pixels). It can take one of the following discrete values: **12, 16, 24 or 32**.
3. **SNR_info**: The channel SNR value, which can be one of **1, 4, 7, or 10** dB.
4. **performance_metric**: The performance metric to be maximized, which can be one of **"PSNR", "SSIM"**.
5. **data_info**: The dataset name (possible value: **"DIV2K"**).
6. **model_name**: The model name, which can be one of the following: **"ConvJSCC", "ResJSCC", "SwinJSCC", "LAJSCC", "FAJSCC","FAPGBJSCC","FAJSCCwoAT","FAJSCCwoLA","FAJSCCwoDf","LAFAJSCC", or "FALAJSCC"**.


### Example of training a model.

    python3 main_train.py rcpp=12 chan_type=AWGN performance_metric=PSNR SNR_info=4 model_name=ConvJSCC data_info=DIV2K


### Example of experimental results for "Architecture Efficiency".
**You can obtain test results for other settings by simply modifying arguments such as the SNR or rcpp values.**

    python3 main_total_evalGM.py chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K rcpp=32 SNR_info=10

    
### Example of experimental results for "Main Results".
**You can obtain test results for other settings by simply modifying the rcpp value, and models in the main_total_eval.py file**

    python3 main_total_eval.py chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K


### Example of experimental results for "Computation Resource Adjustment" and "Complexity Demands at Encoder and Decoder".
    
    python3 main_total_evalratio.py  chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K


### Example of experimental results for "Visual Inspection".

    python3 main_model_visualize.py  SNR_info=1 rcpp=12 chan_type="AWGN" performance_metric="PSNR" model_name=ResJSCC data_info=DIV2K
