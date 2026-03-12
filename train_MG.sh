#!/bin/sh


set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=127.0.0.1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TORCH_DISTRIBUTED_DEBUG=DETAIL

SET=(12 16 24 32)
SET=(16 24 32)
rcpp1=12
chan_info1="AWGN"
SNR_info4=10

###for rcpp in $SET; do
for rcpp in 12 16 24 32; do
  export MASTER_PORT=$((29500 + RANDOM % 1000))
  echo "=== rcpp=${rcpp}, MASTER_PORT=${MASTER_PORT} ==="

  torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main_train_DDP.py \
    rcpp=${rcpp} chan_type=${chan_info1} SNR_info=${SNR_info4} model_name="HugeFAJSCC" data_info="Flickr30k" performance_metric="PSNR"
done

wait
