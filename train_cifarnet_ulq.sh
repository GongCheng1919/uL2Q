#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10
WEIGHTS=./examples/cifar10/ULQ/model/cifar10_full_iter_70000.caffemodel.h5

# 1bit
$TOOLS/caffe train \
    --solver=$MODEL/ULQ/cifar10_full_solver_ulq1.prototxt \
	--weights=$WEIGHTS -gpu 0 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq1_train.log& $@
# 2bit
$TOOLS/caffe train \
    --solver=$MODEL/ULQ/cifar10_full_solver_ulq2.prototxt \
	--weights=$WEIGHTS -gpu 1 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq2_train.log& $@
# 4bit
$TOOLS/caffe train \
    --solver=$MODEL/ULQ/cifar10_full_solver_ulq4.prototxt \
	--weights=$WEIGHTS -gpu 2 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq4_train.log& $@
# 8bit
$TOOLS/caffe train \
    --solver=$MODEL/ULQ/cifar10_full_solver_ulq8.prototxt \
	--weights=$WEIGHTS -gpu 3 \
	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq8_train.log& $@
# 16bit
#$TOOLS/caffe train \
#    --solver=$MODEL/ULQ/cifar10_full_solver_ulq16.prototxt \
#	--weights=$WEIGHTS -gpu 0 \
#	2>&1>&$MODEL/ULQ/log/log_cifarnet_ulq16_train.log& $@