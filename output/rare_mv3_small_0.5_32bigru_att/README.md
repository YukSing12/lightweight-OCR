# Model Summary
```
--------------------------------------------------------------------------------------------------------
   Layer (type)                Input Shape                      Output Shape               Param #    
========================================================================================================
     Conv2D-1               [[1, 3, 32, 320]]                 [1, 8, 16, 160]                216      
    BatchNorm-1             [[1, 8, 16, 160]]                 [1, 8, 16, 160]                32       
   ConvBNLayer-1            [[1, 3, 32, 320]]                 [1, 8, 16, 160]                 0       
     Conv2D-2               [[1, 8, 16, 160]]                 [1, 8, 16, 160]                64       
    BatchNorm-2             [[1, 8, 16, 160]]                 [1, 8, 16, 160]                32       
   ConvBNLayer-2            [[1, 8, 16, 160]]                 [1, 8, 16, 160]                 0       
     Conv2D-3               [[1, 8, 16, 160]]                  [1, 8, 8, 160]                72       
    BatchNorm-3             [[1, 8, 8, 160]]                   [1, 8, 8, 160]                32       
   ConvBNLayer-3            [[1, 8, 16, 160]]                  [1, 8, 8, 160]                 0       
AdaptiveAvgPool2D-1         [[1, 8, 8, 160]]                    [1, 8, 1, 1]                  0       
     Conv2D-4                [[1, 8, 1, 1]]                     [1, 2, 1, 1]                 18       
     Conv2D-5                [[1, 2, 1, 1]]                     [1, 8, 1, 1]                 24       
    SEModule-1              [[1, 8, 8, 160]]                   [1, 8, 8, 160]                 0       
     Conv2D-6               [[1, 8, 8, 160]]                   [1, 8, 8, 160]                64       
    BatchNorm-4             [[1, 8, 8, 160]]                   [1, 8, 8, 160]                32       
   ConvBNLayer-4            [[1, 8, 8, 160]]                   [1, 8, 8, 160]                 0       
  ResidualUnit-1            [[1, 8, 16, 160]]                  [1, 8, 8, 160]                 0       
     Conv2D-7               [[1, 8, 8, 160]]                  [1, 40, 8, 160]                320      
    BatchNorm-5             [[1, 40, 8, 160]]                 [1, 40, 8, 160]                160      
   ConvBNLayer-5            [[1, 8, 8, 160]]                  [1, 40, 8, 160]                 0       
     Conv2D-8               [[1, 40, 8, 160]]                 [1, 40, 4, 160]                360      
    BatchNorm-6             [[1, 40, 4, 160]]                 [1, 40, 4, 160]                160      
   ConvBNLayer-6            [[1, 40, 8, 160]]                 [1, 40, 4, 160]                 0       
     Conv2D-9               [[1, 40, 4, 160]]                 [1, 16, 4, 160]                640      
    BatchNorm-7             [[1, 16, 4, 160]]                 [1, 16, 4, 160]                64       
   ConvBNLayer-7            [[1, 40, 4, 160]]                 [1, 16, 4, 160]                 0       
  ResidualUnit-2            [[1, 8, 8, 160]]                  [1, 16, 4, 160]                 0       
     Conv2D-10              [[1, 16, 4, 160]]                 [1, 48, 4, 160]                768      
    BatchNorm-8             [[1, 48, 4, 160]]                 [1, 48, 4, 160]                192      
   ConvBNLayer-8            [[1, 16, 4, 160]]                 [1, 48, 4, 160]                 0       
     Conv2D-11              [[1, 48, 4, 160]]                 [1, 48, 4, 160]                432      
    BatchNorm-9             [[1, 48, 4, 160]]                 [1, 48, 4, 160]                192      
   ConvBNLayer-9            [[1, 48, 4, 160]]                 [1, 48, 4, 160]                 0       
     Conv2D-12              [[1, 48, 4, 160]]                 [1, 16, 4, 160]                768      
   BatchNorm-10             [[1, 16, 4, 160]]                 [1, 16, 4, 160]                64       
  ConvBNLayer-10            [[1, 48, 4, 160]]                 [1, 16, 4, 160]                 0       
  ResidualUnit-3            [[1, 16, 4, 160]]                 [1, 16, 4, 160]                 0       
     Conv2D-13              [[1, 16, 4, 160]]                 [1, 48, 4, 160]                768      
   BatchNorm-11             [[1, 48, 4, 160]]                 [1, 48, 4, 160]                192      
  ConvBNLayer-11            [[1, 16, 4, 160]]                 [1, 48, 4, 160]                 0       
     Conv2D-14              [[1, 48, 4, 160]]                 [1, 48, 2, 160]               1,200     
   BatchNorm-12             [[1, 48, 2, 160]]                 [1, 48, 2, 160]                192      
  ConvBNLayer-12            [[1, 48, 4, 160]]                 [1, 48, 2, 160]                 0       
AdaptiveAvgPool2D-2         [[1, 48, 2, 160]]                  [1, 48, 1, 1]                  0       
     Conv2D-15               [[1, 48, 1, 1]]                   [1, 12, 1, 1]                 588      
     Conv2D-16               [[1, 12, 1, 1]]                   [1, 48, 1, 1]                 624      
    SEModule-2              [[1, 48, 2, 160]]                 [1, 48, 2, 160]                 0       
     Conv2D-17              [[1, 48, 2, 160]]                 [1, 24, 2, 160]               1,152     
   BatchNorm-13             [[1, 24, 2, 160]]                 [1, 24, 2, 160]                96       
  ConvBNLayer-13            [[1, 48, 2, 160]]                 [1, 24, 2, 160]                 0       
  ResidualUnit-4            [[1, 16, 4, 160]]                 [1, 24, 2, 160]                 0       
     Conv2D-18              [[1, 24, 2, 160]]                 [1, 120, 2, 160]              2,880     
   BatchNorm-14            [[1, 120, 2, 160]]                 [1, 120, 2, 160]               480      
  ConvBNLayer-14            [[1, 24, 2, 160]]                 [1, 120, 2, 160]                0       
     Conv2D-19             [[1, 120, 2, 160]]                 [1, 120, 2, 160]              3,000     
   BatchNorm-15            [[1, 120, 2, 160]]                 [1, 120, 2, 160]               480      
  ConvBNLayer-15           [[1, 120, 2, 160]]                 [1, 120, 2, 160]                0       
AdaptiveAvgPool2D-3        [[1, 120, 2, 160]]                  [1, 120, 1, 1]                 0       
     Conv2D-20              [[1, 120, 1, 1]]                   [1, 30, 1, 1]                3,630     
     Conv2D-21               [[1, 30, 1, 1]]                   [1, 120, 1, 1]               3,720     
    SEModule-3             [[1, 120, 2, 160]]                 [1, 120, 2, 160]                0       
     Conv2D-22             [[1, 120, 2, 160]]                 [1, 24, 2, 160]               2,880     
   BatchNorm-16             [[1, 24, 2, 160]]                 [1, 24, 2, 160]                96       
  ConvBNLayer-16           [[1, 120, 2, 160]]                 [1, 24, 2, 160]                 0       
  ResidualUnit-5            [[1, 24, 2, 160]]                 [1, 24, 2, 160]                 0       
     Conv2D-23              [[1, 24, 2, 160]]                 [1, 120, 2, 160]              2,880     
   BatchNorm-17            [[1, 120, 2, 160]]                 [1, 120, 2, 160]               480      
  ConvBNLayer-17            [[1, 24, 2, 160]]                 [1, 120, 2, 160]                0       
     Conv2D-24             [[1, 120, 2, 160]]                 [1, 120, 2, 160]              3,000     
   BatchNorm-18            [[1, 120, 2, 160]]                 [1, 120, 2, 160]               480      
  ConvBNLayer-18           [[1, 120, 2, 160]]                 [1, 120, 2, 160]                0       
AdaptiveAvgPool2D-4        [[1, 120, 2, 160]]                  [1, 120, 1, 1]                 0       
     Conv2D-25              [[1, 120, 1, 1]]                   [1, 30, 1, 1]                3,630     
     Conv2D-26               [[1, 30, 1, 1]]                   [1, 120, 1, 1]               3,720     
    SEModule-4             [[1, 120, 2, 160]]                 [1, 120, 2, 160]                0       
     Conv2D-27             [[1, 120, 2, 160]]                 [1, 24, 2, 160]               2,880     
   BatchNorm-19             [[1, 24, 2, 160]]                 [1, 24, 2, 160]                96       
  ConvBNLayer-19           [[1, 120, 2, 160]]                 [1, 24, 2, 160]                 0       
  ResidualUnit-6            [[1, 24, 2, 160]]                 [1, 24, 2, 160]                 0       
     Conv2D-28              [[1, 24, 2, 160]]                 [1, 64, 2, 160]               1,536     
   BatchNorm-20             [[1, 64, 2, 160]]                 [1, 64, 2, 160]                256      
  ConvBNLayer-20            [[1, 24, 2, 160]]                 [1, 64, 2, 160]                 0       
     Conv2D-29              [[1, 64, 2, 160]]                 [1, 64, 2, 160]               1,600     
   BatchNorm-21             [[1, 64, 2, 160]]                 [1, 64, 2, 160]                256      
  ConvBNLayer-21            [[1, 64, 2, 160]]                 [1, 64, 2, 160]                 0       
AdaptiveAvgPool2D-5         [[1, 64, 2, 160]]                  [1, 64, 1, 1]                  0       
     Conv2D-30               [[1, 64, 1, 1]]                   [1, 16, 1, 1]                1,040     
     Conv2D-31               [[1, 16, 1, 1]]                   [1, 64, 1, 1]                1,088     
    SEModule-5              [[1, 64, 2, 160]]                 [1, 64, 2, 160]                 0       
     Conv2D-32              [[1, 64, 2, 160]]                 [1, 24, 2, 160]               1,536     
   BatchNorm-22             [[1, 24, 2, 160]]                 [1, 24, 2, 160]                96       
  ConvBNLayer-22            [[1, 64, 2, 160]]                 [1, 24, 2, 160]                 0       
  ResidualUnit-7            [[1, 24, 2, 160]]                 [1, 24, 2, 160]                 0       
     Conv2D-33              [[1, 24, 2, 160]]                 [1, 72, 2, 160]               1,728     
   BatchNorm-23             [[1, 72, 2, 160]]                 [1, 72, 2, 160]                288      
  ConvBNLayer-23            [[1, 24, 2, 160]]                 [1, 72, 2, 160]                 0       
     Conv2D-34              [[1, 72, 2, 160]]                 [1, 72, 2, 160]               1,800     
   BatchNorm-24             [[1, 72, 2, 160]]                 [1, 72, 2, 160]                288      
  ConvBNLayer-24            [[1, 72, 2, 160]]                 [1, 72, 2, 160]                 0       
AdaptiveAvgPool2D-6         [[1, 72, 2, 160]]                  [1, 72, 1, 1]                  0       
     Conv2D-35               [[1, 72, 1, 1]]                   [1, 18, 1, 1]                1,314     
     Conv2D-36               [[1, 18, 1, 1]]                   [1, 72, 1, 1]                1,368     
    SEModule-6              [[1, 72, 2, 160]]                 [1, 72, 2, 160]                 0       
     Conv2D-37              [[1, 72, 2, 160]]                 [1, 24, 2, 160]               1,728     
   BatchNorm-25             [[1, 24, 2, 160]]                 [1, 24, 2, 160]                96       
  ConvBNLayer-25            [[1, 72, 2, 160]]                 [1, 24, 2, 160]                 0       
  ResidualUnit-8            [[1, 24, 2, 160]]                 [1, 24, 2, 160]                 0       
     Conv2D-38              [[1, 24, 2, 160]]                 [1, 144, 2, 160]              3,456     
   BatchNorm-26            [[1, 144, 2, 160]]                 [1, 144, 2, 160]               576      
  ConvBNLayer-26            [[1, 24, 2, 160]]                 [1, 144, 2, 160]                0       
     Conv2D-39             [[1, 144, 2, 160]]                 [1, 144, 1, 160]              3,600     
   BatchNorm-27            [[1, 144, 1, 160]]                 [1, 144, 1, 160]               576      
  ConvBNLayer-27           [[1, 144, 2, 160]]                 [1, 144, 1, 160]                0       
AdaptiveAvgPool2D-7        [[1, 144, 1, 160]]                  [1, 144, 1, 1]                 0       
     Conv2D-40              [[1, 144, 1, 1]]                   [1, 36, 1, 1]                5,220     
     Conv2D-41               [[1, 36, 1, 1]]                   [1, 144, 1, 1]               5,328     
    SEModule-7             [[1, 144, 1, 160]]                 [1, 144, 1, 160]                0       
     Conv2D-42             [[1, 144, 1, 160]]                 [1, 48, 1, 160]               6,912     
   BatchNorm-28             [[1, 48, 1, 160]]                 [1, 48, 1, 160]                192      
  ConvBNLayer-28           [[1, 144, 1, 160]]                 [1, 48, 1, 160]                 0       
  ResidualUnit-9            [[1, 24, 2, 160]]                 [1, 48, 1, 160]                 0       
     Conv2D-43              [[1, 48, 1, 160]]                 [1, 288, 1, 160]             13,824     
   BatchNorm-29            [[1, 288, 1, 160]]                 [1, 288, 1, 160]              1,152     
  ConvBNLayer-29            [[1, 48, 1, 160]]                 [1, 288, 1, 160]                0       
     Conv2D-44             [[1, 288, 1, 160]]                 [1, 288, 1, 160]              7,200     
   BatchNorm-30            [[1, 288, 1, 160]]                 [1, 288, 1, 160]              1,152     
  ConvBNLayer-30           [[1, 288, 1, 160]]                 [1, 288, 1, 160]                0       
AdaptiveAvgPool2D-8        [[1, 288, 1, 160]]                  [1, 288, 1, 1]                 0       
     Conv2D-45              [[1, 288, 1, 1]]                   [1, 72, 1, 1]               20,808     
     Conv2D-46               [[1, 72, 1, 1]]                   [1, 288, 1, 1]              21,024     
    SEModule-8             [[1, 288, 1, 160]]                 [1, 288, 1, 160]                0       
     Conv2D-47             [[1, 288, 1, 160]]                 [1, 48, 1, 160]              13,824     
   BatchNorm-31             [[1, 48, 1, 160]]                 [1, 48, 1, 160]                192      
  ConvBNLayer-31           [[1, 288, 1, 160]]                 [1, 48, 1, 160]                 0       
  ResidualUnit-10           [[1, 48, 1, 160]]                 [1, 48, 1, 160]                 0       
     Conv2D-48              [[1, 48, 1, 160]]                 [1, 288, 1, 160]             13,824     
   BatchNorm-32            [[1, 288, 1, 160]]                 [1, 288, 1, 160]              1,152     
  ConvBNLayer-32            [[1, 48, 1, 160]]                 [1, 288, 1, 160]                0       
     Conv2D-49             [[1, 288, 1, 160]]                 [1, 288, 1, 160]              7,200     
   BatchNorm-33            [[1, 288, 1, 160]]                 [1, 288, 1, 160]              1,152     
  ConvBNLayer-33           [[1, 288, 1, 160]]                 [1, 288, 1, 160]                0       
AdaptiveAvgPool2D-9        [[1, 288, 1, 160]]                  [1, 288, 1, 1]                 0       
     Conv2D-50              [[1, 288, 1, 1]]                   [1, 72, 1, 1]               20,808     
     Conv2D-51               [[1, 72, 1, 1]]                   [1, 288, 1, 1]              21,024     
    SEModule-9             [[1, 288, 1, 160]]                 [1, 288, 1, 160]                0       
     Conv2D-52             [[1, 288, 1, 160]]                 [1, 48, 1, 160]              13,824     
   BatchNorm-34             [[1, 48, 1, 160]]                 [1, 48, 1, 160]                192      
  ConvBNLayer-34           [[1, 288, 1, 160]]                 [1, 48, 1, 160]                 0       
  ResidualUnit-11           [[1, 48, 1, 160]]                 [1, 48, 1, 160]                 0       
     Conv2D-53              [[1, 48, 1, 160]]                 [1, 288, 1, 160]             13,824     
   BatchNorm-35            [[1, 288, 1, 160]]                 [1, 288, 1, 160]              1,152     
  ConvBNLayer-35            [[1, 48, 1, 160]]                 [1, 288, 1, 160]                0       
    MaxPool2D-1            [[1, 288, 1, 160]]                 [1, 288, 1, 80]                 0       
   MobileNetV3-1            [[1, 3, 32, 320]]                 [1, 288, 1, 80]                 0       
     Im2Seq-1               [[1, 288, 1, 80]]                   [1, 80, 288]                  0       
       GRU-1                 [[1, 80, 288]]              [[1, 80, 64], [4, 1, 32]]         80,640     
 EncoderWithRNN-1            [[1, 80, 288]]                     [1, 80, 64]                   0       
 SequenceEncoder-1          [[1, 288, 1, 80]]                   [1, 80, 64]                   0       
     Linear-1                 [[1, 80, 64]]                     [1, 80, 32]                 2,048     
     Linear-2                   [[1, 32]]                         [1, 32]                   1,056     
     Linear-3                 [[1, 80, 32]]                      [1, 80, 1]                  32       
     GRUCell-5            [[1, 4004], [1, 32]]               [[1, 32], [1, 32]]            387,648    
AttentionGRUCell-1  [[1, 32], [1, 80, 64], [1, 3940]] [[[1, 32], [1, 32]], [1, 1, 80]]        0       
     Linear-4                   [[1, 32]]                        [1, 3940]                 130,020    
  AttentionHead-1             [[1, 80, 64]]                    [1, 26, 3940]                  0       
========================================================================================================
Total params: 860,500
Trainable params: 848,180
Non-trainable params: 12,320
--------------------------------------------------------------------------------------------------------
Input size (MB): 0.12
Forward/backward pass size (MB): 29.03
Params size (MB): 3.28
Estimated Total Size (MB): 32.43
--------------------------------------------------------------------------------------------------------
```