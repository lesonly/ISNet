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

## generate the extended boundary mask
After preparing the data folder, you need to use the enlarge_b.py to generate the extended boundary mask mask for training. Run this command
```
python data2/enlarged_b.py
```
## generate the dilated and eroded mask for extended difference loss function
After preparing the data folder, you need to use the dilate_erode.py to generate the dilated and eroded mask for extended difference loss function for training. Run this command
```
python data4/dilate_erode.py
```

## training
you may revise the `TAG` and `SAVEPATH` defined in the *train.py*. After the preparation, run this command 
```
'CUDA_VISIBLE_DEVICES=0,…… python -m torch.distributed.launch --nproc_per_node=4 train.py -b 16'
```
make sure  that the GPU memory is enough (the original training is conducted on a one NVIDIA RTX 2080Ti (11G) card with the batch size of 16).

## test
After the preparation, run this commond to generate the final saliency maps.
```
 python test.py 
```

We provide the trained model file ([Baidu drive](link：https://pan.baidu.com/s/12Q0RGrQFch5oHJ-4xsIbcQ code：p50u), and run this command to check its completeness:
```
cksum MAFNet 
```
you will obtain the result `MAFNet`.

## evaluation

We provide the evaluation code in the folder  "eval_code" for fair comparisons. You may need to revise the `algorithms` , `data_root`, and `maps_root` defined in the `main.m`. 
