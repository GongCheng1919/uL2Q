#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/mnist/ULQ

# 1bit
$TOOLS/caffe test \
    --model=$MODEL/lenet_bn_ulq1.prototxt \
	--weights=$MODEL/model/lenet_bn_ulq1_iter_30000.caffemodel -gpu 0 \
	2>&1>&$MODEL/log/log_lenet_bn_ulq1_test.log& $@
# 2bit
$TOOLS/caffe test \
    --model=$MODEL/lenet_bn_ulq2.prototxt \
	--weights=$MODEL/model/lenet_bn_ulq2_iter_30000.caffemodel -gpu 1 \
	2>&1>&$MODEL/log/log_lenet_bn_ulq2_test.log& $@
# 4bit
$TOOLS/caffe test \
     --model=$MODEL/lenet_bn_ulq4.prototxt \
	--weights=$MODEL/model/lenet_bn_ulq4_iter_30000.caffemodel -gpu 2 \
	2>&1>&$MODEL/log/log_lenet_bn_ulq4_test.log& $@
# 8bit
$TOOLS/caffe test \
     --model=$MODEL/lenet_bn_ulq8.prototxt \
	--weights=$MODEL/model/lenet_bn_ulq8_iter_30000.caffemodel -gpu 3 \
	2>&1>&$MODEL/log/log_lenet_bn_ulq8_test.log& $@
# 16bit
$TOOLS/caffe test \
     --model=$MODEL/lenet_bn_ulq16.prototxt \
	--weights=$MODEL/model/lenet_bn_ulq16_iter_30000.caffemodel -gpu 0 \
	2>&1>&$MODEL/log/log_lenet_bn_ulq16_test.log& $@
#float
# 1bit
$TOOLS/caffe test \
    --model=./examples/mnist/lenet_tn.prototxt \
	--weights=./examples/mnist/model/lenet_tn_iter_30000.caffemodel -gpu 1 \
	2>&1>&$MODEL/log/log_lenet_tn_test.log& $@