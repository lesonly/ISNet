# MAFNet


## dependencies 
```
>= Ubuntu 16.04 
>= Python 3.7
>= Pytorch 1.3.0
OpenCV-Python
```

## preparation 
- download the official pretrained model ([Baidu drive](https://pan.baidu.com/s/1zRhAaGlunIZEOopNSxZNxw 
code：fv6m)) of ResNet-50 implemented in Pytorch if you want to train the network again.
- download or put the RGB saliency benchmark datasets ([Baidu drive](https://pan.baidu.com/s/1kUPZGSe1CN4AOVmB3R3Qxg 
code：sfx6)) in the folder of `dataset` for training or test.

## generate the thicker edge mask
After preparing the data folder, you need to use the mask_edge.py to generate the thicker edge mask for training. Run this command
```
python data2/mask_edge.py
```
## generate the dilated mask
After preparing the data folder, you need to use the mask_edge.py to generate the dilated mask for training. Run this command
```
python data2/mask_regione.py
```

## training
you may revise the `TAG` and `SAVEPATH` defined in the *train.py*. After the preparation, run this command 
```
python train.py
```
make sure  that the GPU memory is enough (the original training is conducted on a one NVIDIA RTX 2080Ti (11G) card with the batch size of 24).

## test
After the preparation, run this commond to generate the final saliency maps.
```
 python test.py 
```

We provide the trained model file ([Baidu drive](https://pan.baidu.com/s/1KdP0doBCiIme4y_j4Y4OPQ code:uht6)), and run this command to check its completeness:
```
cksum model-20210718 
```
you will obtain the result `model-20210718`.

## evaluation
We provide the evaluation code in the folder  "eval_code" for fair comparisons. You may need to revise the `algorithms` , `data_root`, and `maps_root` defined in the `main.m`. 
