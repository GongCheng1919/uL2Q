# μL2Q
## Configurations
> **μL2Q** has been integrated into the caffe framework, and its installation process is consistent with the official caffe, see [caffe installation](https://github.com/BVLC/caffe).
## Usage
>  **μL2Q** provides a configurable interface, and the compression scheme can be directly configured in the model training.
In the layer configuration parameters, add the weight compression parameter configuration:
```
weights_compress 
	- ""(default): not use weights compress.
	- "ULQ": use the uL2Q to compress the weights of one layer.
weights_compress_param
	- alpha: keep the mean of layer's weights.
	- beta: keep the std of layer's weights.
	- fixedpos:don't used.
 	- maxbits: config the bit-widths after compression.
```
> An example use as:
```
weights_compress:"ULQ"
weights_compress_param{
    maxbits:2
}
```
# Citation
Please cite **μL2Q** in your publications if it helps your research:
```
@article{cheng2019μL2Q,
  title={μL2Q: An Ultra-Low Loss Quantization Method for DNN},
  author={Cheng, Gong and Ye, Lu and Tao, Li and Xiaofan, Zhang and Cong, Hao and Deming, Chen and Yao, Chen},
  journal={Conference The 2019 International Joint Conference on Neural Networks (IJCNN)},
  year={2019}
}
```
