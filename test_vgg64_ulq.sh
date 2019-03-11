#!/usr/bin/env sh
set -e

TOOLS=./build/tools
MODEL=./examples/cifar10/ULQ

# 1bit
$TOOLS/caffe test \
    --model=$MODEL/vgg7_64_train_test_bn_ulq1.prototxt \
	--weights=$MODEL/model/vgg7_64_bn_ulq1_iter_35000.caffemodel -gpu 0 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq1_test.log& $@
# 2bit
$TOOLS/caffe test \
    --model=$MODEL/vgg7_64_train_test_bn_ulq2.prototxt \
	--weights=$MODEL/model/vgg7_64_bn_ulq2_iter_35000.caffemodel -gpu 1 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq2_test.log& $@
# 4bit
$TOOLS/caffe test \
    --model=$MODEL/vgg7_64_train_test_bn_ulq4.prototxt \
	--weights=$MODEL/model/vgg7_64_bn_ulq4_iter_35000.caffemodel -gpu 2 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq4_test.log& $@
# 8bit
$TOOLS/caffe test \
    --model=$MODEL/vgg7_64_train_test_bn_ulq8.prototxt \
	--weights=$MODEL/model/vgg7_64_bn_ulq8_iter_35000.caffemodel -gpu 3 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq8_test.log& $@
# 16bit
#$TOOLS/caffe test \
#    --model=$MODEL/vgg7_64_train_test_bn_ulq16.prototxt \
#	--weights=$MODEL/model/vgg7_64_bn_ulq16_iter_35000.caffemodel -gpu 0 \
#	2>&1>&$MODEL/log/log_vgg7_64_bn_ulq16_test.log& $@
# float
$TOOLS/caffe test \
    --model=./examples/cifar10/vgg7_64_train_test_bn.prototxt \
	--weights=./examples/cifar10/model/vgg7_64_bn.caffemodel -gpu 1 \
	2>&1>&$MODEL/log/log_vgg7_64_bn_test.log& $@