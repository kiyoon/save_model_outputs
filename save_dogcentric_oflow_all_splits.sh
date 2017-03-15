#!/bin/bash

if [ $# -lt 2 ]
then
	echo "Usage: $0 [input_dir] [output_dir] [train, val, or both]"
	exit
fi

CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_0/069_epoch-0.0106_loss-0.9971_acc-4.3539_val_loss-0.4153_val_acc.hdf5 0 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_1/068_epoch-0.0325_loss-0.9904_acc-4.5177_val_loss-0.4359_val_acc.hdf5 1 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_2/050_epoch-0.0169_loss-0.9946_acc-4.0133_val_loss-0.4637_val_acc.hdf5 2 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_3/076_epoch-0.0060_loss-0.9980_acc-3.9603_val_loss-0.4258_val_acc.hdf5 3 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_4/098_epoch-0.0075_loss-0.9979_acc-4.4131_val_loss-0.4265_val_acc.hdf5 4 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_5/054_epoch-0.0560_loss-0.9808_acc-3.8983_val_loss-0.4125_val_acc.hdf5 5 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_6/064_epoch-0.0062_loss-0.9983_acc-4.3499_val_loss-0.4300_val_acc.hdf5 6 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_7/093_epoch-0.0034_loss-0.9993_acc-4.0847_val_loss-0.4423_val_acc.hdf5 7 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_8/044_epoch-0.0198_loss-0.9933_acc-3.9679_val_loss-0.4302_val_acc.hdf5 8 $3
CUDA_VISIBLE_DEVICES=0 ./save.py "$1" "$2" /storage/kiyoon/siamese_model/batch_32_noaug/dogcentric/oflow/first_try/split_9/032_epoch-0.0187_loss-0.9938_acc-3.2587_val_loss-0.4317_val_acc.hdf5 9 $3
