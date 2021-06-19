# lightweight-OCR

## Description
Project of lightweight OCR contest.[[url]](https://aistudio.baidu.com/aistudio/competition/detail/75)

## Overall directory structure
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
    │       │   └── rec_rednet_vd.py
    │       └── necks
    │           └── rnn.py
    └── tools
        ├── egaleeye_prune.py
        ├── infer_rec.py
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
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-B|0.7248|7.5MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpnb_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-C|0.7290|7.3MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpnc_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-D|**0.7319**|8.5MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpnd_small_1.0)|
|CRNN|MobileNetV3-small-1.0|48BiGRU|FPN-A, MaxPool->BlurPool|0.7145|4.8MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpn_bp_small_1.0)|
|CRNN|MobileNetV3-large-0.5|96BiGRU|FPN-A, 200epoch->500epoch|0.7243|7.6MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_fpn_large_0.5_96bigru)|
|CRNN|ResNet18|64BiGRU|Prune 90% FLOPs|0.7177|8.9MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18_64_prune_0.9)|
|CRNN|ResNet18SE|64BiGRU|Prune 91% FLOPs|0.7202|9.8MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18se_64_prune_0.91)|
|CRNN|ResNet18|48BiGRU|Prune 90% FLOPs|0.7076|8.9MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18_48_prune_0.9)|
|CRNN|ResNet18SE|48BiGRU|Prune 90% FLOPs|0.7087|9.1MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/resnet18se_48_prune_0.9)|
|RARE|MobileNetV3-samll-0.5|32BiGRU|Remove TPS|0.4329|9.2MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/rare_mv3_small_0.5_32bigru_att)|
|Star|MobileNetV3-samll-1.0|48BiGRU|FPN-B|0.7093|15.4MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/star_mv3_fpnb_small_1.0)|
