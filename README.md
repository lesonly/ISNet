# ISNet


## Dependencies 
```
>= Ubuntu 16.04 
>= Python 3.7
>= Pytorch 1.3.0
OpenCV-Python
```

## Preparation 
- download the official pretrained model ([Baidu drive](https://pan.baidu.com/s/1zRhAaGlunIZEOopNSxZNxw 
code：fv6m)) of ResNet-50 implemented in Pytorch if you want to train the network again.
- download or put the RGB saliency benchmark datasets ([Baidu drive](https://pan.baidu.com/s/1kUPZGSe1CN4AOVmB3R3Qxg 
code：sfx6)) in the folder of `data` for training or test.

## Generate the boundary envelope
After preparing the data folder, you need to use the enlarge_b.py to produce the boundary envelope, which can be used to generate the Expanded Ground Truth for training. Run this command
```
python data4/enlarged_b.py
```
## Generate the dilated and eroded mask for hierarchical difference-aware loss function
After preparing the data folder, you need to u the dilate_erode.py to generate the dilated and eroded mask for hierarchical difference-aware loss function for training. Run this command
```
python data4/dilate_erode.py
```

## Training
you may revise the `TAG` and `SAVEPATH` defined in the *train.py*. After the preparation, run this command 
```
'CUDA_VISIBLE_DEVICES=0,1,…… python -m torch.distributed.launch --nproc_per_node=x train.py -b 16'
```
make sure that the GPU memory is enough (You can adjust the batch according to your GPU memory).

## Test
After the preparation, run this commond to generate the final saliency maps.
```
 python test.py 
```

We provide the trained model file ([Baidu drive](link：https://pan.baidu.com/s/1rPpyKnNQPbyA1RVbz1ipjg?pwd=ys6n  code：ys6n), and run this command to check its completeness:
```
cksum ISNet 
```
you will obtain the result `ISNet`.

## Evaluation

We provide the predicted saliency maps on five benchmark datasets,including PASCAL-S, ECSSD, HKU-S, DUT-OMRON and DUTS-TE. ([Baidu drive](link：https://pan.baidu.com/s/1usIegmAAbt9FJAx_FIMoFg?pwd=dkkt code: dkkt)

You can use the evaluation code in the folder  "eval_code" for fair comparisons, but you may need to revise the `algorithms` , `data_root`, and `maps_root` defined in the `main.m`. 

## Citation

We really hope this repo can contribute the conmunity, and if you find this work useful, please use the following citation:

@article{DBLP:journals/pr/ZhuLG24, <br>
  author       = {Ge Zhu and
                  Jinbao Li and
                  Yahong Guo},
  title        = {Separate first, then segment: An integrity segmentation network for salient object detection}, <br>
  journal      = {Pattern Recognit.}, <br>
  volume       = {150}, <br>
  pages        = {110328}, <br>
  year         = {2024}, <br>
  url          = {https://doi.org/10.1016/j.patcog.2024.110328}, <br>
  doi          = {10.1016/J.PATCOG.2024.110328}, <br>
  timestamp    = {Fri, 31 May 2024 21:07:46 +0200}, <br>
  biburl       = {https://dblp.org/rec/journals/pr/ZhuLG24.bib}, <br>
  bibsource    = {dblp computer science bibliography, https://dblp.org} <br>
}
