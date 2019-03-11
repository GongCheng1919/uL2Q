#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10/ULQ

# 1bit
$TOOLS/caffe train \
    --solver=$MODEL/vgg7_64_solver_bn_ulq1.prototxt -gpu 0 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq1_train.log& $@
# 2bit
$TOOLS/caffe train \
    --solver=$MODEL/vgg7_64_solver_bn_ulq2.prototxt -gpu 1 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq2_train.log& $@
# 4bit
$TOOLS/caffe train \
    --solver=$MODEL/vgg7_64_solver_bn_ulq4.prototxt -gpu 2 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq4_train.log& $@
# 8bit
$TOOLS/caffe train \
    --solver=$MODEL/vgg7_64_solver_bn_ulq8.prototxt -gpu 3 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq8_train.log& $@
# 16bit
#$TOOLS/caffe train \
#    --solver=$MODEL/vgg7_64_solver_bn_ulq16.prototxt -gpu 3 \
#	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq16_train.log& $@