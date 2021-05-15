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
