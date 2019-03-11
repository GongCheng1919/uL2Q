# uL2Q
## Configurations
> **uL2Q** has been integrated into the caffe framework, and its installation process is consistent with the official caffe, see [caffe installation](https://github.com/BVLC/caffe).
## Usage
>  **uL2Q** provides a configurable interface, and the compression scheme can be directly configured in the model training.
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
Please cite **uL2Q** in your publications if it helps your research:
```
```
