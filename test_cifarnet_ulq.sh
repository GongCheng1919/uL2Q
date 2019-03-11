#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10
WEIGHTS=./examples/cifar10/ULQ/model/cifar10_full_iter_70000.caffemodel.h5

# 1bit
$TOOLS/caffe test \
    --model=$MODEL/ULQ/cifar10_full_train_test_ulq1.prototxt \
	--weights=$MODEL/ULQ/model/cifar10_full_ulq1_iter_25000.caffemodel -gpu 0 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq1_test.log& $@
# 2bit
$TOOLS/caffe test \
    --model=$MODEL/ULQ/cifar10_full_train_test_ulq2.prototxt \
	--weights=$MODEL/ULQ/model/cifar10_full_ulq2_iter_25000.caffemodel -gpu 1 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq2_test.log& $@
# 4bit
$TOOLS/caffe test \
    --model=$MODEL/ULQ/cifar10_full_train_test_ulq4.prototxt \
	--weights=$MODEL/ULQ/model/cifar10_full_ulq4_iter_25000.caffemodel -gpu 2 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq4_test.log& $@
# 8bit
$TOOLS/caffe test \
    --model=$MODEL/ULQ/cifar10_full_train_test_ulq8.prototxt \
	--weights=$MODEL/ULQ/model/cifar10_full_ulq8_iter_25000.caffemodel -gpu 3 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq8_test.log& $@
# 16bit
#$TOOLS/caffe test \
#    --model=$MODEL/ULQ/cifar10_full_train_test_ulq16.prototxt \
#	--weights=$MODEL/ULQ/model/cifar10_full_ulq16_iter_25000.caffemodel -gpu 0 \
#	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq16_test.log& $@
# float
$TOOLS/caffe test \
    --model=$MODEL/ULQ/cifar10_full_train_test.prototxt \
	--weights=$WEIGHTS -gpu 1 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_test.log& $@