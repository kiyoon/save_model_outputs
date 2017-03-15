#!/bin/bash

if [ $# -lt 2 ]
then
	echo "Usage: $0 [input_dir] [output_dir] [train, val, or both]"
	exit
fi

CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" ../siamese_clsfy/batch_32_noaug/oflow/dropped_0/filter_256/split_1/053_epoch-0.0739_loss-0.9783_acc-7.2593_val_loss-0.1033_val_acc.hdf5 1 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" ../siamese_clsfy/batch_32_noaug/oflow/dropped_0/filter_256/split_2/055_epoch-0.0720_loss-0.9780_acc-7.4823_val_loss-0.1019_val_acc.hdf5 2 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" ../siamese_clsfy/batch_32_noaug/oflow/dropped_0/filter_256/split_3/042_epoch-0.0894_loss-0.9740_acc-7.4814_val_loss-0.0868_val_acc.hdf5 3 $3
