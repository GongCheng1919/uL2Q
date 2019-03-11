#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/mnist/ULQ

# 1bit
$TOOLS/caffe train \
    --solver=$MODEL/lenet_bn_solver_ulq1.prototxt -gpu 0 \
	2>&1>&$MODEL/log/log_lenet_bn_solver_ulq1.log& $@
# 2bit
$TOOLS/caffe train \
    --solver=$MODEL/lenet_bn_solver_ulq2.prototxt -gpu 0 \
	2>&1>&$MODEL/log/log_lenet_bn_solver_ulq2.log& $@
# 4bit
$TOOLS/caffe train \
    --solver=$MODEL/lenet_bn_solver_ulq4.prototxt -gpu 1 \
	2>&1>&$MODEL/log/log_lenet_bn_solver_ulq4.log& $@
# 8bit
$TOOLS/caffe train \
    --solver=$MODEL/lenet_bn_solver_ulq8.prototxt -gpu 2 \
	2>&1>&$MODEL/log/log_lenet_bn_solver_ulq8.log& $@
# 16bit
$TOOLS/caffe train \
    --solver=$MODEL/lenet_bn_solver_ulq16.prototxt -gpu 3 \
	2>&1>&$MODEL/log/log_lenet_bn_solver_ulq16.log& $@