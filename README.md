# $\mu$L2Q: An Ultra-Low Loss Quantization Method for DNN Compression

## This method has been merged into [Quantization-caffe](https://github.com/GongCheng1919/Quantization-caffe). Please go to [Quantization-caffe](https://github.com/GongCheng1919/Quantization-caffe) for detailed introduction.

### $\mu$L2Q quantization method
- Firstly, by analyzing the data distribution of the model, 
we find that the weight distribution of most models obeys the 
normal distribution approximately, and the regularization term 
based on theoretical deduction (L2) also shows that the weight 
of the model will be constrained to approach the normal distribution 
in the training process.
![data distribution](data_distribution_analysis.png)
- Based on the analysis of model weight distribution, 
our method quantifies uniformly ($\lambda$ interval) data $\varphi$ with standard normal 
distribution to discrete value set $Q$, and minimize the 
L2 distance before and after quantization.
![ulq_steps](ulq_steps.png)
