# Model summary before pruning
```
-------------------------------------------------------------------------------------
   Layer (type)         Input Shape             Output Shape            Param #    
=====================================================================================
     Conv2D-1        [[1, 3, 32, 320]]        [1, 32, 32, 320]            864      
    BatchNorm-1      [[1, 32, 32, 320]]       [1, 32, 32, 320]            128      
   ConvBNLayer-1     [[1, 3, 32, 320]]        [1, 32, 32, 320]             0       
     Conv2D-2        [[1, 32, 32, 320]]       [1, 32, 32, 320]           9,216     
    BatchNorm-2      [[1, 32, 32, 320]]       [1, 32, 32, 320]            128      
   ConvBNLayer-2     [[1, 32, 32, 320]]       [1, 32, 32, 320]             0       
     Conv2D-3        [[1, 32, 32, 320]]       [1, 64, 32, 320]          18,432     
    BatchNorm-3      [[1, 64, 32, 320]]       [1, 64, 32, 320]            256      
   ConvBNLayer-3     [[1, 32, 32, 320]]       [1, 64, 32, 320]             0       
    MaxPool2D-1      [[1, 64, 32, 320]]       [1, 64, 16, 160]             0       
     Conv2D-4        [[1, 64, 16, 160]]       [1, 64, 16, 160]          36,864     
    BatchNorm-4      [[1, 64, 16, 160]]       [1, 64, 16, 160]            256      
   ConvBNLayer-4     [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
     Conv2D-5        [[1, 64, 16, 160]]       [1, 64, 16, 160]          36,864     
    BatchNorm-5      [[1, 64, 16, 160]]       [1, 64, 16, 160]            256      
   ConvBNLayer-5     [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
AdaptiveAvgPool2D-1  [[1, 64, 16, 160]]        [1, 64, 1, 1]               0       
     Conv2D-6         [[1, 64, 1, 1]]          [1, 16, 1, 1]             1,040     
     Conv2D-7         [[1, 16, 1, 1]]          [1, 64, 1, 1]             1,088     
    SEModule-1       [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
     Conv2D-8        [[1, 64, 16, 160]]       [1, 64, 16, 160]           4,096     
    BatchNorm-6      [[1, 64, 16, 160]]       [1, 64, 16, 160]            256      
   ConvBNLayer-6     [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
   BasicBlock-1      [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
     Conv2D-9        [[1, 64, 16, 160]]       [1, 64, 16, 160]          36,864     
    BatchNorm-7      [[1, 64, 16, 160]]       [1, 64, 16, 160]            256      
   ConvBNLayer-7     [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
     Conv2D-10       [[1, 64, 16, 160]]       [1, 64, 16, 160]          36,864     
    BatchNorm-8      [[1, 64, 16, 160]]       [1, 64, 16, 160]            256      
   ConvBNLayer-8     [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
AdaptiveAvgPool2D-2  [[1, 64, 16, 160]]        [1, 64, 1, 1]               0       
     Conv2D-11        [[1, 64, 1, 1]]          [1, 16, 1, 1]             1,040     
     Conv2D-12        [[1, 16, 1, 1]]          [1, 64, 1, 1]             1,088     
    SEModule-2       [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
   BasicBlock-2      [[1, 64, 16, 160]]       [1, 64, 16, 160]             0       
     Conv2D-13       [[1, 64, 16, 160]]       [1, 128, 8, 160]          73,728     
    BatchNorm-9      [[1, 128, 8, 160]]       [1, 128, 8, 160]            512      
   ConvBNLayer-9     [[1, 64, 16, 160]]       [1, 128, 8, 160]             0       
     Conv2D-14       [[1, 128, 8, 160]]       [1, 128, 8, 160]          147,456    
   BatchNorm-10      [[1, 128, 8, 160]]       [1, 128, 8, 160]            512      
  ConvBNLayer-10     [[1, 128, 8, 160]]       [1, 128, 8, 160]             0       
AdaptiveAvgPool2D-3  [[1, 128, 8, 160]]        [1, 128, 1, 1]              0       
     Conv2D-15        [[1, 128, 1, 1]]         [1, 32, 1, 1]             4,128     
     Conv2D-16        [[1, 32, 1, 1]]          [1, 128, 1, 1]            4,224     
    SEModule-3       [[1, 128, 8, 160]]       [1, 128, 8, 160]             0       
   AvgPool2D-11      [[1, 64, 16, 160]]       [1, 64, 8, 160]              0       
     Conv2D-17       [[1, 64, 8, 160]]        [1, 128, 8, 160]           8,192     
   BatchNorm-11      [[1, 128, 8, 160]]       [1, 128, 8, 160]            512      
  ConvBNLayer-11     [[1, 64, 16, 160]]       [1, 128, 8, 160]             0       
   BasicBlock-3      [[1, 64, 16, 160]]       [1, 128, 8, 160]             0       
     Conv2D-18       [[1, 128, 8, 160]]       [1, 128, 8, 160]          147,456    
   BatchNorm-12      [[1, 128, 8, 160]]       [1, 128, 8, 160]            512      
  ConvBNLayer-12     [[1, 128, 8, 160]]       [1, 128, 8, 160]             0       
     Conv2D-19       [[1, 128, 8, 160]]       [1, 128, 8, 160]          147,456    
   BatchNorm-13      [[1, 128, 8, 160]]       [1, 128, 8, 160]            512      
  ConvBNLayer-13     [[1, 128, 8, 160]]       [1, 128, 8, 160]             0       
AdaptiveAvgPool2D-4  [[1, 128, 8, 160]]        [1, 128, 1, 1]              0       
     Conv2D-20        [[1, 128, 1, 1]]         [1, 32, 1, 1]             4,128     
     Conv2D-21        [[1, 32, 1, 1]]          [1, 128, 1, 1]            4,224     
    SEModule-4       [[1, 128, 8, 160]]       [1, 128, 8, 160]             0       
   BasicBlock-4      [[1, 128, 8, 160]]       [1, 128, 8, 160]             0       
     Conv2D-22       [[1, 128, 8, 160]]       [1, 256, 4, 160]          294,912    
   BatchNorm-14      [[1, 256, 4, 160]]       [1, 256, 4, 160]           1,024     
  ConvBNLayer-14     [[1, 128, 8, 160]]       [1, 256, 4, 160]             0       
     Conv2D-23       [[1, 256, 4, 160]]       [1, 256, 4, 160]          589,824    
   BatchNorm-15      [[1, 256, 4, 160]]       [1, 256, 4, 160]           1,024     
  ConvBNLayer-15     [[1, 256, 4, 160]]       [1, 256, 4, 160]             0       
AdaptiveAvgPool2D-5  [[1, 256, 4, 160]]        [1, 256, 1, 1]              0       
     Conv2D-24        [[1, 256, 1, 1]]         [1, 64, 1, 1]            16,448     
     Conv2D-25        [[1, 64, 1, 1]]          [1, 256, 1, 1]           16,640     
    SEModule-5       [[1, 256, 4, 160]]       [1, 256, 4, 160]             0       
   AvgPool2D-16      [[1, 128, 8, 160]]       [1, 128, 4, 160]             0       
     Conv2D-26       [[1, 128, 4, 160]]       [1, 256, 4, 160]          32,768     
   BatchNorm-16      [[1, 256, 4, 160]]       [1, 256, 4, 160]           1,024     
  ConvBNLayer-16     [[1, 128, 8, 160]]       [1, 256, 4, 160]             0       
   BasicBlock-5      [[1, 128, 8, 160]]       [1, 256, 4, 160]             0       
     Conv2D-27       [[1, 256, 4, 160]]       [1, 256, 4, 160]          589,824    
   BatchNorm-17      [[1, 256, 4, 160]]       [1, 256, 4, 160]           1,024     
  ConvBNLayer-17     [[1, 256, 4, 160]]       [1, 256, 4, 160]             0       
     Conv2D-28       [[1, 256, 4, 160]]       [1, 256, 4, 160]          589,824    
   BatchNorm-18      [[1, 256, 4, 160]]       [1, 256, 4, 160]           1,024     
  ConvBNLayer-18     [[1, 256, 4, 160]]       [1, 256, 4, 160]             0       
AdaptiveAvgPool2D-6  [[1, 256, 4, 160]]        [1, 256, 1, 1]              0       
     Conv2D-29        [[1, 256, 1, 1]]         [1, 64, 1, 1]            16,448     
     Conv2D-30        [[1, 64, 1, 1]]          [1, 256, 1, 1]           16,640     
    SEModule-6       [[1, 256, 4, 160]]       [1, 256, 4, 160]             0       
   BasicBlock-6      [[1, 256, 4, 160]]       [1, 256, 4, 160]             0       
     Conv2D-31       [[1, 256, 4, 160]]       [1, 512, 2, 160]         1,179,648   
   BatchNorm-19      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-19     [[1, 256, 4, 160]]       [1, 512, 2, 160]             0       
     Conv2D-32       [[1, 512, 2, 160]]       [1, 512, 2, 160]         2,359,296   
   BatchNorm-20      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-20     [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
AdaptiveAvgPool2D-7  [[1, 512, 2, 160]]        [1, 512, 1, 1]              0       
     Conv2D-33        [[1, 512, 1, 1]]         [1, 128, 1, 1]           65,664     
     Conv2D-34        [[1, 128, 1, 1]]         [1, 512, 1, 1]           66,048     
    SEModule-7       [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
   AvgPool2D-21      [[1, 256, 4, 160]]       [1, 256, 2, 160]             0       
     Conv2D-35       [[1, 256, 2, 160]]       [1, 512, 2, 160]          131,072    
   BatchNorm-21      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-21     [[1, 256, 4, 160]]       [1, 512, 2, 160]             0       
   BasicBlock-7      [[1, 256, 4, 160]]       [1, 512, 2, 160]             0       
     Conv2D-36       [[1, 512, 2, 160]]       [1, 512, 2, 160]         2,359,296   
   BatchNorm-22      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-22     [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
     Conv2D-37       [[1, 512, 2, 160]]       [1, 512, 2, 160]         2,359,296   
   BatchNorm-23      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-23     [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
AdaptiveAvgPool2D-8  [[1, 512, 2, 160]]        [1, 512, 1, 1]              0       
     Conv2D-38        [[1, 512, 1, 1]]         [1, 128, 1, 1]           65,664     
     Conv2D-39        [[1, 128, 1, 1]]         [1, 512, 1, 1]           66,048     
    SEModule-8       [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
   BasicBlock-8      [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
    MaxPool2D-2      [[1, 512, 2, 160]]       [1, 512, 1, 80]              0       
    ResNetSE-1       [[1, 3, 32, 320]]        [1, 512, 1, 80]              0       
     Im2Seq-1        [[1, 512, 1, 80]]          [1, 80, 512]               0       
       GRU-1           [[1, 80, 512]]    [[1, 80, 128], [4, 1, 64]]     296,448    
 EncoderWithRNN-1      [[1, 80, 512]]           [1, 80, 128]               0       
 SequenceEncoder-1   [[1, 512, 1, 80]]          [1, 80, 128]               0       
     Linear-1          [[1, 80, 128]]          [1, 80, 3939]            508,131    
     CTCHead-1         [[1, 80, 128]]          [1, 80, 3939]               0       
=====================================================================================
Total params: 12,364,963
Trainable params: 12,345,251
Non-trainable params: 19,712
-------------------------------------------------------------------------------------
Input size (MB): 0.12
Forward/backward pass size (MB): 154.06
Params size (MB): 47.17
Estimated Total Size (MB): 201.35
-------------------------------------------------------------------------------------
```
# Pruning
pruning 91.0% FLOPs and skip ['res2a_se_1_weights', 'res2b_se_1_weights', 'res3a_se_1_weights', 'res3b_se_1_weights', 'res4a_se_1_weights', 'res4b_se_1_weights', 'res5a_se_1_weights', 'res5b_se_1_weights', 'res5a_branch2b_weights', 'res5a_branch1_weights'] layer

# Model summary after pruning:
```
-------------------------------------------------------------------------------------
   Layer (type)         Input Shape             Output Shape            Param #    
=====================================================================================
     Conv2D-1        [[1, 3, 32, 320]]        [1, 5, 32, 320]             135      
    BatchNorm-1      [[1, 5, 32, 320]]        [1, 5, 32, 320]             20       
   ConvBNLayer-1     [[1, 3, 32, 320]]        [1, 5, 32, 320]              0       
     Conv2D-2        [[1, 5, 32, 320]]        [1, 8, 32, 320]             360      
    BatchNorm-2      [[1, 8, 32, 320]]        [1, 8, 32, 320]             32       
   ConvBNLayer-2     [[1, 5, 32, 320]]        [1, 8, 32, 320]              0       
     Conv2D-3        [[1, 8, 32, 320]]        [1, 18, 32, 320]           1,296     
    BatchNorm-3      [[1, 18, 32, 320]]       [1, 18, 32, 320]            72       
   ConvBNLayer-3     [[1, 8, 32, 320]]        [1, 18, 32, 320]             0       
    MaxPool2D-1      [[1, 18, 32, 320]]       [1, 18, 16, 160]             0       
     Conv2D-4        [[1, 18, 16, 160]]       [1, 6, 16, 160]             972      
    BatchNorm-4      [[1, 6, 16, 160]]        [1, 6, 16, 160]             24       
   ConvBNLayer-4     [[1, 18, 16, 160]]       [1, 6, 16, 160]              0       
     Conv2D-5        [[1, 6, 16, 160]]        [1, 26, 16, 160]           1,404     
    BatchNorm-5      [[1, 26, 16, 160]]       [1, 26, 16, 160]            104      
   ConvBNLayer-5     [[1, 6, 16, 160]]        [1, 26, 16, 160]             0       
AdaptiveAvgPool2D-1  [[1, 26, 16, 160]]        [1, 26, 1, 1]               0       
     Conv2D-6         [[1, 26, 1, 1]]          [1, 16, 1, 1]              432      
     Conv2D-7         [[1, 16, 1, 1]]          [1, 26, 1, 1]              442      
    SEModule-1       [[1, 26, 16, 160]]       [1, 26, 16, 160]             0       
     Conv2D-8        [[1, 18, 16, 160]]       [1, 26, 16, 160]            468      
    BatchNorm-6      [[1, 26, 16, 160]]       [1, 26, 16, 160]            104      
   ConvBNLayer-6     [[1, 18, 16, 160]]       [1, 26, 16, 160]             0       
   BasicBlock-1      [[1, 18, 16, 160]]       [1, 26, 16, 160]             0       
     Conv2D-9        [[1, 26, 16, 160]]       [1, 6, 16, 160]            1,404     
    BatchNorm-7      [[1, 6, 16, 160]]        [1, 6, 16, 160]             24       
   ConvBNLayer-7     [[1, 26, 16, 160]]       [1, 6, 16, 160]              0       
     Conv2D-10       [[1, 6, 16, 160]]        [1, 26, 16, 160]           1,404     
    BatchNorm-8      [[1, 26, 16, 160]]       [1, 26, 16, 160]            104      
   ConvBNLayer-8     [[1, 6, 16, 160]]        [1, 26, 16, 160]             0       
AdaptiveAvgPool2D-2  [[1, 26, 16, 160]]        [1, 26, 1, 1]               0       
     Conv2D-11        [[1, 26, 1, 1]]          [1, 16, 1, 1]              432      
     Conv2D-12        [[1, 16, 1, 1]]          [1, 26, 1, 1]              442      
    SEModule-2       [[1, 26, 16, 160]]       [1, 26, 16, 160]             0       
   BasicBlock-2      [[1, 26, 16, 160]]       [1, 26, 16, 160]             0       
     Conv2D-13       [[1, 26, 16, 160]]       [1, 13, 8, 160]            3,042     
    BatchNorm-9      [[1, 13, 8, 160]]        [1, 13, 8, 160]             52       
   ConvBNLayer-9     [[1, 26, 16, 160]]       [1, 13, 8, 160]              0       
     Conv2D-14       [[1, 13, 8, 160]]        [1, 52, 8, 160]            6,084     
   BatchNorm-10      [[1, 52, 8, 160]]        [1, 52, 8, 160]             208      
  ConvBNLayer-10     [[1, 13, 8, 160]]        [1, 52, 8, 160]              0       
AdaptiveAvgPool2D-3  [[1, 52, 8, 160]]         [1, 52, 1, 1]               0       
     Conv2D-15        [[1, 52, 1, 1]]          [1, 32, 1, 1]             1,696     
     Conv2D-16        [[1, 32, 1, 1]]          [1, 52, 1, 1]             1,716     
    SEModule-3       [[1, 52, 8, 160]]        [1, 52, 8, 160]              0       
   AvgPool2D-11      [[1, 26, 16, 160]]       [1, 26, 8, 160]              0       
     Conv2D-17       [[1, 26, 8, 160]]        [1, 52, 8, 160]            1,352     
   BatchNorm-11      [[1, 52, 8, 160]]        [1, 52, 8, 160]             208      
  ConvBNLayer-11     [[1, 26, 16, 160]]       [1, 52, 8, 160]              0       
   BasicBlock-3      [[1, 26, 16, 160]]       [1, 52, 8, 160]              0       
     Conv2D-18       [[1, 52, 8, 160]]        [1, 13, 8, 160]            6,084     
   BatchNorm-12      [[1, 13, 8, 160]]        [1, 13, 8, 160]             52       
  ConvBNLayer-12     [[1, 52, 8, 160]]        [1, 13, 8, 160]              0       
     Conv2D-19       [[1, 13, 8, 160]]        [1, 52, 8, 160]            6,084     
   BatchNorm-13      [[1, 52, 8, 160]]        [1, 52, 8, 160]             208      
  ConvBNLayer-13     [[1, 13, 8, 160]]        [1, 52, 8, 160]              0       
AdaptiveAvgPool2D-4  [[1, 52, 8, 160]]         [1, 52, 1, 1]               0       
     Conv2D-20        [[1, 52, 1, 1]]          [1, 32, 1, 1]             1,696     
     Conv2D-21        [[1, 32, 1, 1]]          [1, 52, 1, 1]             1,716     
    SEModule-4       [[1, 52, 8, 160]]        [1, 52, 8, 160]              0       
   BasicBlock-4      [[1, 52, 8, 160]]        [1, 52, 8, 160]              0       
     Conv2D-22       [[1, 52, 8, 160]]        [1, 26, 4, 160]           12,168     
   BatchNorm-14      [[1, 26, 4, 160]]        [1, 26, 4, 160]             104      
  ConvBNLayer-14     [[1, 52, 8, 160]]        [1, 26, 4, 160]              0       
     Conv2D-23       [[1, 26, 4, 160]]        [1, 86, 4, 160]           20,124     
   BatchNorm-15      [[1, 86, 4, 160]]        [1, 86, 4, 160]             344      
  ConvBNLayer-15     [[1, 26, 4, 160]]        [1, 86, 4, 160]              0       
AdaptiveAvgPool2D-5  [[1, 86, 4, 160]]         [1, 86, 1, 1]               0       
     Conv2D-24        [[1, 86, 1, 1]]          [1, 64, 1, 1]             5,568     
     Conv2D-25        [[1, 64, 1, 1]]          [1, 86, 1, 1]             5,590     
    SEModule-5       [[1, 86, 4, 160]]        [1, 86, 4, 160]              0       
   AvgPool2D-16      [[1, 52, 8, 160]]        [1, 52, 4, 160]              0       
     Conv2D-26       [[1, 52, 4, 160]]        [1, 86, 4, 160]            4,472     
   BatchNorm-16      [[1, 86, 4, 160]]        [1, 86, 4, 160]             344      
  ConvBNLayer-16     [[1, 52, 8, 160]]        [1, 86, 4, 160]              0       
   BasicBlock-5      [[1, 52, 8, 160]]        [1, 86, 4, 160]              0       
     Conv2D-27       [[1, 86, 4, 160]]        [1, 26, 4, 160]           20,124     
   BatchNorm-17      [[1, 26, 4, 160]]        [1, 26, 4, 160]             104      
  ConvBNLayer-17     [[1, 86, 4, 160]]        [1, 26, 4, 160]              0       
     Conv2D-28       [[1, 26, 4, 160]]        [1, 86, 4, 160]           20,124     
   BatchNorm-18      [[1, 86, 4, 160]]        [1, 86, 4, 160]             344      
  ConvBNLayer-18     [[1, 26, 4, 160]]        [1, 86, 4, 160]              0       
AdaptiveAvgPool2D-6  [[1, 86, 4, 160]]         [1, 86, 1, 1]               0       
     Conv2D-29        [[1, 86, 1, 1]]          [1, 64, 1, 1]             5,568     
     Conv2D-30        [[1, 64, 1, 1]]          [1, 86, 1, 1]             5,590     
    SEModule-6       [[1, 86, 4, 160]]        [1, 86, 4, 160]              0       
   BasicBlock-6      [[1, 86, 4, 160]]        [1, 86, 4, 160]              0       
     Conv2D-31       [[1, 86, 4, 160]]        [1, 80, 2, 160]           61,920     
   BatchNorm-19      [[1, 80, 2, 160]]        [1, 80, 2, 160]             320      
  ConvBNLayer-19     [[1, 86, 4, 160]]        [1, 80, 2, 160]              0       
     Conv2D-32       [[1, 80, 2, 160]]        [1, 512, 2, 160]          368,640    
   BatchNorm-20      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-20     [[1, 80, 2, 160]]        [1, 512, 2, 160]             0       
AdaptiveAvgPool2D-7  [[1, 512, 2, 160]]        [1, 512, 1, 1]              0       
     Conv2D-33        [[1, 512, 1, 1]]         [1, 128, 1, 1]           65,664     
     Conv2D-34        [[1, 128, 1, 1]]         [1, 512, 1, 1]           66,048     
    SEModule-7       [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
   AvgPool2D-21      [[1, 86, 4, 160]]        [1, 86, 2, 160]              0       
     Conv2D-35       [[1, 86, 2, 160]]        [1, 512, 2, 160]          44,032     
   BatchNorm-21      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-21     [[1, 86, 4, 160]]        [1, 512, 2, 160]             0       
   BasicBlock-7      [[1, 86, 4, 160]]        [1, 512, 2, 160]             0       
     Conv2D-36       [[1, 512, 2, 160]]       [1, 74, 2, 160]           340,992    
   BatchNorm-22      [[1, 74, 2, 160]]        [1, 74, 2, 160]             296      
  ConvBNLayer-22     [[1, 512, 2, 160]]       [1, 74, 2, 160]              0       
     Conv2D-37       [[1, 74, 2, 160]]        [1, 512, 2, 160]          340,992    
   BatchNorm-23      [[1, 512, 2, 160]]       [1, 512, 2, 160]           2,048     
  ConvBNLayer-23     [[1, 74, 2, 160]]        [1, 512, 2, 160]             0       
AdaptiveAvgPool2D-8  [[1, 512, 2, 160]]        [1, 512, 1, 1]              0       
     Conv2D-38        [[1, 512, 1, 1]]         [1, 128, 1, 1]           65,664     
     Conv2D-39        [[1, 128, 1, 1]]         [1, 512, 1, 1]           66,048     
    SEModule-8       [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
   BasicBlock-8      [[1, 512, 2, 160]]       [1, 512, 2, 160]             0       
    MaxPool2D-2      [[1, 512, 2, 160]]       [1, 512, 1, 80]              0       
    ResNetSE-1       [[1, 3, 32, 320]]        [1, 512, 1, 80]              0       
     Im2Seq-1        [[1, 512, 1, 80]]          [1, 80, 512]               0       
       GRU-1           [[1, 80, 512]]    [[1, 80, 128], [4, 1, 64]]     296,448    
 EncoderWithRNN-1      [[1, 80, 512]]           [1, 80, 128]               0       
 SequenceEncoder-1   [[1, 512, 1, 80]]          [1, 80, 128]               0       
     Linear-1          [[1, 80, 128]]          [1, 80, 3939]            508,131    
     CTCHead-1         [[1, 80, 128]]          [1, 80, 3939]               0       
=====================================================================================
Total params: 2,371,780
Trainable params: 2,362,568
Non-trainable params: 9,212
-------------------------------------------------------------------------------------
Input size (MB): 0.12
Forward/backward pass size (MB): 72.53
Params size (MB): 9.05
Estimated Total Size (MB): 81.69
-------------------------------------------------------------------------------------
```