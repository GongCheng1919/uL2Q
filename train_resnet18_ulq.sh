#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./models/ResNet-18/ULQ

# 1bit
$TOOLS/caffe train \
    --solver=$MODEL/resnet18_ulq1_solver.prototxt -gpu 0 \
	2>&1>&$MODEL/log/log_resnet18_ulq1.log& $@
# 2bit
$TOOLS/caffe train \
    --solver=$MODEL/resnet18_ulq2_solver.prototxt -gpu 1 \
	2>&1>&$MODEL/log/log_resnet18_ulq2.log& $@
# 4bit
$TOOLS/caffe train \
    --solver=$MODEL/resnet18_ulq4_solver.prototxt -gpu 2 \
	2>&1>&$MODEL/log/log_resnet18_ulq4.log& $@
# 8bit
$TOOLS/caffe train \
    --solver=$MODEL/resnet18_ulq8_solver.prototxt -gpu 3 \
	2>&1>&$MODEL/log/log_resnet18_ulq8.log& $@
# 16bit
#$TOOLS/caffe train \
#    --solver=$MODEL/resnet18_ulq1_solver.prototxt -gpu all \
#	2>&1>$MODEL/log/log_vgg7_64_bn_ulq16.log& $@