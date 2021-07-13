# lightweight-OCR

## Description
Project of lightweight OCR contest.[[url]](https://aistudio.baidu.com/aistudio/competition/detail/75)

**The best accuracy(score) is 0.7254, ranking 36th.**

We tried different algorithms like [CRNN](https://arxiv.org/abs/1507.05717), [RARE](https://arxiv.org/abs/1603.03915v1), [StarNet](http://www.bmva.org/bmvc/2016/papers/paper043/index.html), and finally found [CRNN](https://arxiv.org/abs/1507.05717) is particularly useful in Chinese character recognition. Given this result, we adopted [CRNN](https://arxiv.org/abs/1507.05717) and focus on 1. obtaining a pruned (large-sparse) model from a large model and 2. designing a compact (small-dense) model.

### **Large-sparse Model**
Pruning is a common method to derive a large-sparse network, and it usually contains pre-training a model, pruning strategy, and fine-tuning. PaddleSlim supported L1-norm, L2-norm, and FPGM based pruning strategies. We also adopted [EgaleEye](https://arxiv.org/abs/2007.02491), which proposed adaptive batch normalization to fast and accurately evaluate the network. (See work/prune.py)

We split the original training dataset into two subsets with split percentages of 80% training and 20% validation using random selection.

We use ResNet-18 as the backbone to trained a recognition model on the new training dataset normally from scratch as a baseline. After training, the large model achieved 0.7587 accuracy on the validation dataset. 

Run [work/prune.py](./work/tools/prune.py) to get the sensitivities of each layer.

<img src=./images/res18_64_fpgm_sen.pickle1.png width="100%">

After pruning 90% FLOPs and fine-tuning, we get a large-sparse model with an accuracy of 0.7177 and an allocated size of 8.9MB. [Squeeze-and-Excitation(SE)](https://arxiv.org/abs/1709.01507) can further improve the accuracy of the large-sparse model from 0.7177 to 0.7202 but increase allocated size to 9.8MB.

Overall, these results indicate that pruning a large model(ResNet18) to get an ultra-lightweight model is challenging. In our case, we pruned 90% FLOPs, causing accuracy to drop rapidly from 0.7587 to 0.7177. With an allocated size limited to 10MB, it is not easy to introduce more modules like [SE](https://arxiv.org/abs/1709.01507) into the network to improve accuracy further.

### **Small-dense Model**
In recent years, there has been an increasing amount of literature on designing small-dense models for the optimal trade-off between accuracy and efficiency. PaddleOCR use [MobileNetV3](https://arxiv.org/abs/1905.02244) as backbone to design ultra-lightweight model for Chinese character recognition. In this project, we train MobileNetV3-small-0.5 as our baseline with an accuracy of 0.6836.

#### **Coordinate Attention**
[SE](https://arxiv.org/abs/1709.01507) module are integraed in a [MobileNetV3](https://arxiv.org/abs/1905.02244). Some researchers proposed [Coordinate Attention(CA)](https://arxiv.org/abs/2103.02907) to replace [SE](https://arxiv.org/abs/1709.01507) for a better performance.
We also try to replace [SE](https://arxiv.org/abs/1709.01507) module by [CA](https://arxiv.org/abs/2103.02907) module but the accuracy drop from 0.6836 to 0.6786. However, [CA](https://arxiv.org/abs/2103.02907) module reduces parameters and the allocated size of the model from 5.87MB to 4.49MB and from 6.9MB to 5.9MB, respectively.

<img src=./images/SE_CA.png width="100%">

#### **Meta-ACON**
[MobileNetV3](https://arxiv.org/abs/1905.02244) introduces a new, fast, and quantization-friendly nonlinearity, h-swish function. Recently, [TFNet](https://arxiv.org/abs/2009.04759) proposed a novel activation function called [ACON](https://arxiv.org/abs/2009.04759) that explicitly learns to **AC**tivate the neurons **O**r **N**ot. [ACON-C](https://arxiv.org/abs/2009.04759) function contains three learnable parameters p1, p2, and beta while [MetaACON-C](https://arxiv.org/abs/2009.04759) build a small network to learn beta. We try to replace HSwish and Relu with [MetaACON-C](https://arxiv.org/abs/2009.04759), and we found that it is slower in back-propagation. The accuracy is 0.6787, and the allocated size is 10.5MB. We think we should redesign [MetaACON-C](https://arxiv.org/abs/2009.04759) in 
[MobileNetV3](https://arxiv.org/abs/1905.02244).

#### **FPN**
Inspired by [Feature Pyramid Networks(FPN)](https://arxiv.org/abs/1612.03144) and [Dynamic Feature Pyramid Networks(DyFPN)](https://arxiv.org/abs/2012.00779), we designed four FPNs to aggregate multi-scale feature information in recognition model. 


##### **FPN-A, FPN-B**
We use <a href="https://www.codecogs.com/eqnedit.php?latex=\oplus" target="_blank"><img src="https://latex.codecogs.com/png.latex?\oplus" title="\oplus" /></a> to denote concatenation here. Given a list of input features with different scales <a href="https://www.codecogs.com/eqnedit.php?latex=\{F3,F4,F5\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\{F3,F4,F5\}" title="\{F3,F4,F5\}" /></a>, the output features <a href="https://www.codecogs.com/eqnedit.php?latex=\{P3,P4,P5\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\{P3,P4,P5\}" title="\{P3,P4,P5\}" /></a> are aggregated as 

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=P_5&space;=&space;F_5" target="_blank"><img src="https://latex.codecogs.com/png.latex?P_5&space;=&space;F_5" title="P_5 = F_5" /></a></div>

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=P_l&space;=&space;f_l(F_l&space;\oplus&space;R(F_{l&plus;1})),&space;l&space;=&space;3,4" target="_blank"><img src="https://latex.codecogs.com/png.latex?P_l&space;=&space;f_l(F_l&space;\oplus&space;R(F_{l&plus;1})),&space;l&space;=&space;3,4" title="P_l = f_l(F_l \oplus R(F_{l+1})), l = 3,4" /></a></div>

where <a href="https://www.codecogs.com/eqnedit.php?latex=l" target="_blank"><img src="https://latex.codecogs.com/png.latex?l" title="l" /></a> denotes the level of pyramid. <a href="https://www.codecogs.com/eqnedit.php?latex=f_l" target="_blank"><img src="https://latex.codecogs.com/png.latex?f_l" title="f_l" /></a> denotes <a href="https://www.codecogs.com/eqnedit.php?latex=3\times&space;3" target="_blank"><img src="https://latex.codecogs.com/png.latex?3\times&space;3" title="3\times 3" /></a> convolution operation with different strides. <a href="https://www.codecogs.com/eqnedit.php?latex=R" target="_blank"><img src="https://latex.codecogs.com/png.latex?R" title="R" /></a> denotes the resizing operation i.e. upsampling with a scale factor of (2,1). 

Finally we sums the aggregated features to get output <a href="https://www.codecogs.com/eqnedit.php?latex=O" target="_blank"><img src="https://latex.codecogs.com/png.latex?O" title="O" /></a> as:
<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=O&space;=&space;P_5&space;\oplus&space;P_4&space;\oplus&space;P_3" target="_blank"><img src="https://latex.codecogs.com/png.latex?O&space;=&space;P_5&space;\oplus&space;P_4&space;\oplus&space;P_3" title="O = P_5 \oplus P_4 \oplus P_3" /></a></div>

In FPN-A, input features <a href="https://www.codecogs.com/eqnedit.php?latex=\{F3,F4,F5\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\{F3,F4,F5\}" title="\{F3,F4,F5\}" /></a> come from the first feature in state <a href="https://www.codecogs.com/eqnedit.php?latex=\{3,&space;4,&space;5\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\{3,&space;4,&space;5\}" title="\{3, 4, 5\}" /></a> whereas in FPN-B, input features come from the last feature in state <a href="https://www.codecogs.com/eqnedit.php?latex=\{3,&space;4,&space;5\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\{3,&space;4,&space;5\}" title="\{3, 4, 5\}" /></a>.

FPN-A achieved an accuracy of 0.7161 and allocated 4.8MB, whereas FPN-B achieved an accuracy of 0.7248 and allocated 7.5MB. The structures of FPN-A and FPN-B are shown below.
<img src=./images/FPNAB.png width="100%">

##### **FPN-C**
We redefine aggregated features <a href="https://www.codecogs.com/eqnedit.php?latex=\{P3,&space;P4,&space;P5\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\{P3,&space;P4,&space;P5\}" title="\{P3, P4, P5\}" /></a> and output <a href="https://www.codecogs.com/eqnedit.php?latex=O" target="_blank"><img src="https://latex.codecogs.com/png.latex?O" title="O" /></a> as
<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=P_5&space;=&space;F_5" target="_blank"><img src="https://latex.codecogs.com/png.latex?P_5&space;=&space;F_5" title="P_5 = F_5" /></a></div>
<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=P_l&space;=&space;Conv_{1\times&space;1}(F_l&space;\oplus&space;R(P_{l&plus;1})),&space;l&space;=&space;3,4" target="_blank"><img src="https://latex.codecogs.com/png.latex?P_l&space;=&space;Conv_{1\times&space;1}(F_l&space;\oplus&space;R(P_{l&plus;1})),&space;l&space;=&space;3,4" title="P_l = Conv_{1\times 1}(F_l \oplus R(P_{l+1})), l = 3,4" /></a></div>
<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=O&space;=&space;P_5&space;\oplus&space;f_4(P_4)&space;\oplus&space;f_3(P_3)" target="_blank"><img src="https://latex.codecogs.com/png.latex?O&space;=&space;P_5&space;\oplus&space;f_4(P_4)&space;\oplus&space;f_3(P_3)" title="O = P_5 \oplus f_4(P_4) \oplus f_3(P_3)" /></a></div>
where <a href="https://www.codecogs.com/eqnedit.php?latex=Conv_{1\times&space;1}" target="_blank"><img src="https://latex.codecogs.com/png.latex?Conv_{1\times&space;1}" title="Conv_{1\times 1}" /></a> is <a href="https://www.codecogs.com/eqnedit.php?latex=1\times&space;1" target="_blank"><img src="https://latex.codecogs.com/png.latex?1\times&space;1" title="1\times 1" /></a> convolution operation with different output channels.
Compared to FPN-B, FPN-C improved accuracy to 0.7290 with allocated size of 7.3MB.
The structure of FPN-C is shown below.
<img src=./images/FPNC.png width="100%">

##### **FPN-D**
Inspired by [DyFPN](https://arxiv.org/abs/2012.00779), <a href="https://www.codecogs.com/eqnedit.php?latex=P_l" target="_blank"><img src="https://latex.codecogs.com/png.latex?P_l" title="P_l" /></a> is added by more convolution operations with three different kernel size as:
<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=P_l&space;=&space;Conv_{1\times&space;1}(F_l&space;\oplus&space;R(P_{l&plus;1}))&space;&plus;&space;Conv_{3\times&space;3}(F_l&space;\oplus&space;R(P_{l&plus;1}))&space;&plus;&space;Conv_{5\times&space;5}(F_l&space;\oplus&space;R(P_{l&plus;1})),&space;l&space;=&space;3,4" target="_blank"><img src="https://latex.codecogs.com/png.latex?P_l&space;=&space;Conv_{1\times&space;1}(F_l&space;\oplus&space;R(P_{l&plus;1}))&space;&plus;&space;Conv_{3\times&space;3}(F_l&space;\oplus&space;R(P_{l&plus;1}))&space;&plus;&space;Conv_{5\times&space;5}(F_l&space;\oplus&space;R(P_{l&plus;1})),&space;l&space;=&space;3,4" title="P_l = Conv_{1\times 1}(F_l \oplus R(P_{l+1})) + Conv_{3\times 3}(F_l \oplus R(P_{l+1})) + Conv_{5\times 5}(F_l \oplus R(P_{l+1})), l = 3,4" /></a></div>
FPN-D reached highest accuracy of 0.7319 with allocated size of 8.5MB.
The structure of FPN-D is shown below.
<img src=./images/FPND.png width="100%">

## Overall Directory Structure
The overall directory structure of lightweight-OCR is introduced as follows:

```
lightweight-OCR   
├── 1833844.ipynb
├── data
│   ├── data87683
│   └── data87685
├── output
├── PaddleOCR
├── PaddleSlim
├── README.md
└── work
    ├── configs
    ├── label_dict.txt
    ├── label.txt
    ├── ppocr
    │   └── modeling
    │       ├── backbones
    │       └── necks
    └── tools
        ├── egaleeye_prune.py
        ├── export_pruned_model.py
        ├── infer
        ├── model_summary.py
        ├── prune.py
        └── train.py
```

## Todo list    

## Installation   

### Requirements:
- Python 3.7.10
- CUDA 10.1 
- PaddleOCR-release/2.1
- PaddleSlim-release/2.0.0
- PaddlePaddle-2.0.2

## Instruction

### Train:
Run command
```
python PaddleOCR/tools/train.py -c work/configs/rec_mobilev3_small_1_train
```
### Eval:
Run command
```
python PaddleOCR/tools/eval.py -c work/configs/rec_mobilev3_small_1_train -o Global.checkpoints=./output/rec_mobilev3_small_1.0/best_accuracy
```
### Test:
Run command to export inference model
```
python PaddleOCR/tools/export.py -c work/config/rec_mobilev3_small_1_train.yml -o Global.checkpoints=./output/rec_mobilev3_small_1.0/best_accuracy Global.save_inference_dir=./output/rec_mobilev3_small_1.0/
```
inference model will be exported in output/rec_mobilev3_small_1.0/inference

Change 'rec_model_dir' and run command
```
python PaddleOCR/tools/infer/predict_rec.py --image_dir=./data/test_images/A榜测试数据集/TestAImages/ --rec_char_dict_path=./work/label_dict.txt --rec_model_dir=./output/rec_mobilev3_small_1.0/
```
result file will be saved in output/%Y-%m-%d-%H-%M-%S.log.

## Performance
|Algorithm|Backbone|Neck|Trick|Socre|Model Size|Model Link|
|---|---|---|---|---|---|---|
|CRNN|MobileNetV3-small-1.0|48BiGRU|None|0.6836|6.9MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|SE->CA|0.6786|5.9MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_ca_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|ReLu, H-Swish -> MetaAconC|0.6787|10.5MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_metaaconc_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-A|0.7161|4.8MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpn_small_1.0)|
|CRNN|MobileNetV3-large-0.5|96BiGRU|FPN-A, 200epoch->500epoch|0.7243|7.6MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpn_large_0.5_96bigru)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-A, MaxPool->BlurPool|0.7145|4.8MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpn_bp_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-B|0.7248|7.5MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpnb_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-C|0.7290|7.3MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpnc_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-D|0.7319|8.5MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpnd_small_1.0)|
|CRNN|ResNet18|64BiGRU|Prune 90% FLOPs|0.7177|8.9MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18_64_prune_0.9)|
|CRNN|ResNet18SE|64BiGRU|Prune 91% FLOPs|0.7202|9.8MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18se_64_prune_0.91)|
|CRNN|ResNet18|48BiGRU|Prune 90% FLOPs|0.7076|8.9MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18_48_prune_0.9)|
|CRNN|ResNet18SE|48BiGRU|Prune 90% FLOPs|0.7087|9.1MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18se_48_prune_0.9)|
|RARE|MobileNetV3-samll-0.5|32BiGRU|Remove TPS|0.4329|9.2MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/rare_mv3_small_0.5_32bigru_att)|
|Star|MobileNetV3-samll-1.0|48BiGRU|FPN-B|0.7093|15.4MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/star_mv3_fpnb_small_1.0)|
|CRNN|MobileNetV3-large-0.5|72BiGRU|FPN-D|**0.735**|9.3MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpnd_large_0.5)|
