# lightweight-OCR

## Description
Project of lightweight OCR contest.[[url]](https://aistudio.baidu.com/aistudio/competition/detail/75)

## Overall directory structure
The overall directory structure of SBCR(Steel Billet Character Recognition) is introduced as follows:

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
python PaddleOCR/tools/train.py -c work/configs/rec_mobilev3_0.75_train.yml
```
### Eval:
Run command
```
python PaddleOCR/tools/eval.py -c work/configs/rec_mobilev3_0.75_train.yml -o Global.checkpoints=./output/mobilev3_0.75_48/latest
```
### Test:
Change 'rec_model_dir' and run command
```
python PaddleOCR/tools/infer/predict_rec.py --image_dir=./data/test_images/A榜测试数据集/TestAImages/ --rec_char_dict_path=./work/label_dict.txt --rec_model_dir=./output/rec_mobilev3_small_1.0/
```
result file will be saved in output/%Y-%m-%d-%H-%M-%S.log.

## Performance
|Model|Backbone|Neck|Trick|Socre|Model Size|Model Link|
|---|---|---|---|---|---|---|
|CRNN|MobileNetV3-small-1.0|48GRU|None|0.6836|6.9MB|[link](https://github.com/YukSing12/lightweight-OCR/tree/main/output/mobilev3_small_1.0)|
