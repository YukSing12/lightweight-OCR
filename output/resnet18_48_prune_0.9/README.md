# Model before pruning
```
----------------------------------------------------------------------------------
  Layer (type)        Input Shape            Output Shape            Param #    
==================================================================================
    Conv2D-1       [[1, 3, 32, 320]]       [1, 32, 32, 320]            864      
   BatchNorm-1     [[1, 32, 32, 320]]      [1, 32, 32, 320]            128      
  ConvBNLayer-1    [[1, 3, 32, 320]]       [1, 32, 32, 320]             0       
    Conv2D-2       [[1, 32, 32, 320]]      [1, 32, 32, 320]           9,216     
   BatchNorm-2     [[1, 32, 32, 320]]      [1, 32, 32, 320]            128      
  ConvBNLayer-2    [[1, 32, 32, 320]]      [1, 32, 32, 320]             0       
    Conv2D-3       [[1, 32, 32, 320]]      [1, 64, 32, 320]          18,432     
   BatchNorm-3     [[1, 64, 32, 320]]      [1, 64, 32, 320]            256      
  ConvBNLayer-3    [[1, 32, 32, 320]]      [1, 64, 32, 320]             0       
   MaxPool2D-1     [[1, 64, 32, 320]]      [1, 64, 16, 160]             0       
    Conv2D-4       [[1, 64, 16, 160]]      [1, 64, 16, 160]          36,864     
   BatchNorm-4     [[1, 64, 16, 160]]      [1, 64, 16, 160]            256      
  ConvBNLayer-4    [[1, 64, 16, 160]]      [1, 64, 16, 160]             0       
    Conv2D-5       [[1, 64, 16, 160]]      [1, 64, 16, 160]          36,864     
   BatchNorm-5     [[1, 64, 16, 160]]      [1, 64, 16, 160]            256      
  ConvBNLayer-5    [[1, 64, 16, 160]]      [1, 64, 16, 160]             0       
    Conv2D-6       [[1, 64, 16, 160]]      [1, 64, 16, 160]           4,096     
   BatchNorm-6     [[1, 64, 16, 160]]      [1, 64, 16, 160]            256      
  ConvBNLayer-6    [[1, 64, 16, 160]]      [1, 64, 16, 160]             0       
  BasicBlock-1     [[1, 64, 16, 160]]      [1, 64, 16, 160]             0       
    Conv2D-7       [[1, 64, 16, 160]]      [1, 64, 16, 160]          36,864     
   BatchNorm-7     [[1, 64, 16, 160]]      [1, 64, 16, 160]            256      
  ConvBNLayer-7    [[1, 64, 16, 160]]      [1, 64, 16, 160]             0       
    Conv2D-8       [[1, 64, 16, 160]]      [1, 64, 16, 160]          36,864     
   BatchNorm-8     [[1, 64, 16, 160]]      [1, 64, 16, 160]            256      
  ConvBNLayer-8    [[1, 64, 16, 160]]      [1, 64, 16, 160]             0       
  BasicBlock-2     [[1, 64, 16, 160]]      [1, 64, 16, 160]             0       
    Conv2D-9       [[1, 64, 16, 160]]      [1, 128, 8, 160]          73,728     
   BatchNorm-9     [[1, 128, 8, 160]]      [1, 128, 8, 160]            512      
  ConvBNLayer-9    [[1, 64, 16, 160]]      [1, 128, 8, 160]             0       
    Conv2D-10      [[1, 128, 8, 160]]      [1, 128, 8, 160]          147,456    
  BatchNorm-10     [[1, 128, 8, 160]]      [1, 128, 8, 160]            512      
 ConvBNLayer-10    [[1, 128, 8, 160]]      [1, 128, 8, 160]             0       
  AvgPool2D-11     [[1, 64, 16, 160]]       [1, 64, 8, 160]             0       
    Conv2D-11      [[1, 64, 8, 160]]       [1, 128, 8, 160]           8,192     
  BatchNorm-11     [[1, 128, 8, 160]]      [1, 128, 8, 160]            512      
 ConvBNLayer-11    [[1, 64, 16, 160]]      [1, 128, 8, 160]             0       
  BasicBlock-3     [[1, 64, 16, 160]]      [1, 128, 8, 160]             0       
    Conv2D-12      [[1, 128, 8, 160]]      [1, 128, 8, 160]          147,456    
  BatchNorm-12     [[1, 128, 8, 160]]      [1, 128, 8, 160]            512      
 ConvBNLayer-12    [[1, 128, 8, 160]]      [1, 128, 8, 160]             0       
    Conv2D-13      [[1, 128, 8, 160]]      [1, 128, 8, 160]          147,456    
  BatchNorm-13     [[1, 128, 8, 160]]      [1, 128, 8, 160]            512      
 ConvBNLayer-13    [[1, 128, 8, 160]]      [1, 128, 8, 160]             0       
  BasicBlock-4     [[1, 128, 8, 160]]      [1, 128, 8, 160]             0       
    Conv2D-14      [[1, 128, 8, 160]]      [1, 256, 4, 160]          294,912    
  BatchNorm-14     [[1, 256, 4, 160]]      [1, 256, 4, 160]           1,024     
 ConvBNLayer-14    [[1, 128, 8, 160]]      [1, 256, 4, 160]             0       
    Conv2D-15      [[1, 256, 4, 160]]      [1, 256, 4, 160]          589,824    
  BatchNorm-15     [[1, 256, 4, 160]]      [1, 256, 4, 160]           1,024     
 ConvBNLayer-15    [[1, 256, 4, 160]]      [1, 256, 4, 160]             0       
  AvgPool2D-16     [[1, 128, 8, 160]]      [1, 128, 4, 160]             0       
    Conv2D-16      [[1, 128, 4, 160]]      [1, 256, 4, 160]          32,768     
  BatchNorm-16     [[1, 256, 4, 160]]      [1, 256, 4, 160]           1,024     
 ConvBNLayer-16    [[1, 128, 8, 160]]      [1, 256, 4, 160]             0       
  BasicBlock-5     [[1, 128, 8, 160]]      [1, 256, 4, 160]             0       
    Conv2D-17      [[1, 256, 4, 160]]      [1, 256, 4, 160]          589,824    
  BatchNorm-17     [[1, 256, 4, 160]]      [1, 256, 4, 160]           1,024     
 ConvBNLayer-17    [[1, 256, 4, 160]]      [1, 256, 4, 160]             0       
    Conv2D-18      [[1, 256, 4, 160]]      [1, 256, 4, 160]          589,824    
  BatchNorm-18     [[1, 256, 4, 160]]      [1, 256, 4, 160]           1,024     
 ConvBNLayer-18    [[1, 256, 4, 160]]      [1, 256, 4, 160]             0       
  BasicBlock-6     [[1, 256, 4, 160]]      [1, 256, 4, 160]             0       
    Conv2D-19      [[1, 256, 4, 160]]      [1, 512, 2, 160]         1,179,648   
  BatchNorm-19     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-19    [[1, 256, 4, 160]]      [1, 512, 2, 160]             0       
    Conv2D-20      [[1, 512, 2, 160]]      [1, 512, 2, 160]         2,359,296   
  BatchNorm-20     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-20    [[1, 512, 2, 160]]      [1, 512, 2, 160]             0       
  AvgPool2D-21     [[1, 256, 4, 160]]      [1, 256, 2, 160]             0       
    Conv2D-21      [[1, 256, 2, 160]]      [1, 512, 2, 160]          131,072    
  BatchNorm-21     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-21    [[1, 256, 4, 160]]      [1, 512, 2, 160]             0       
  BasicBlock-7     [[1, 256, 4, 160]]      [1, 512, 2, 160]             0       
    Conv2D-22      [[1, 512, 2, 160]]      [1, 512, 2, 160]         2,359,296   
  BatchNorm-22     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-22    [[1, 512, 2, 160]]      [1, 512, 2, 160]             0       
    Conv2D-23      [[1, 512, 2, 160]]      [1, 512, 2, 160]         2,359,296   
  BatchNorm-23     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-23    [[1, 512, 2, 160]]      [1, 512, 2, 160]             0       
  BasicBlock-8     [[1, 512, 2, 160]]      [1, 512, 2, 160]             0       
   MaxPool2D-2     [[1, 512, 2, 160]]       [1, 512, 1, 80]             0       
    ResNet-1       [[1, 3, 32, 320]]        [1, 512, 1, 80]             0       
    Im2Seq-1       [[1, 512, 1, 80]]         [1, 80, 512]               0       
      GRU-1          [[1, 80, 512]]    [[1, 80, 96], [4, 1, 48]]     203,904    
EncoderWithRNN-1     [[1, 80, 512]]           [1, 80, 96]               0       
SequenceEncoder-1  [[1, 512, 1, 80]]          [1, 80, 96]               0       
    Linear-1         [[1, 80, 96]]           [1, 80, 3939]           382,083    
    CTCHead-1        [[1, 80, 96]]           [1, 80, 3939]              0       
==================================================================================
Total params: 11,795,811
Trainable params: 11,776,099
Non-trainable params: 19,712
----------------------------------------------------------------------------------
Input size (MB): 0.12
Forward/backward pass size (MB): 135.24
Params size (MB): 45.00
Estimated Total Size (MB): 180.35
----------------------------------------------------------------------------------
```

# Pruning
pruning 90.0% FLOPs and skip ['res5a_branch2b_weights', 'res5a_branch1_weights'] layer

# Model after pruning
```
----------------------------------------------------------------------------------
  Layer (type)        Input Shape            Output Shape            Param #    
==================================================================================
    Conv2D-1       [[1, 3, 32, 320]]        [1, 4, 32, 320]            108      
   BatchNorm-1     [[1, 4, 32, 320]]        [1, 4, 32, 320]            16       
  ConvBNLayer-1    [[1, 3, 32, 320]]        [1, 4, 32, 320]             0       
    Conv2D-2       [[1, 4, 32, 320]]        [1, 3, 32, 320]            108      
   BatchNorm-2     [[1, 3, 32, 320]]        [1, 3, 32, 320]            12       
  ConvBNLayer-2    [[1, 4, 32, 320]]        [1, 3, 32, 320]             0       
    Conv2D-3       [[1, 3, 32, 320]]       [1, 10, 32, 320]            270      
   BatchNorm-3     [[1, 10, 32, 320]]      [1, 10, 32, 320]            40       
  ConvBNLayer-3    [[1, 3, 32, 320]]       [1, 10, 32, 320]             0       
   MaxPool2D-1     [[1, 10, 32, 320]]      [1, 10, 16, 160]             0       
    Conv2D-4       [[1, 10, 16, 160]]       [1, 6, 16, 160]            540      
   BatchNorm-4     [[1, 6, 16, 160]]        [1, 6, 16, 160]            24       
  ConvBNLayer-4    [[1, 10, 16, 160]]       [1, 6, 16, 160]             0       
    Conv2D-5       [[1, 6, 16, 160]]       [1, 22, 16, 160]           1,188     
   BatchNorm-5     [[1, 22, 16, 160]]      [1, 22, 16, 160]            88       
  ConvBNLayer-5    [[1, 6, 16, 160]]       [1, 22, 16, 160]             0       
    Conv2D-6       [[1, 10, 16, 160]]      [1, 22, 16, 160]            220      
   BatchNorm-6     [[1, 22, 16, 160]]      [1, 22, 16, 160]            88       
  ConvBNLayer-6    [[1, 10, 16, 160]]      [1, 22, 16, 160]             0       
  BasicBlock-1     [[1, 10, 16, 160]]      [1, 22, 16, 160]             0       
    Conv2D-7       [[1, 22, 16, 160]]       [1, 6, 16, 160]           1,188     
   BatchNorm-7     [[1, 6, 16, 160]]        [1, 6, 16, 160]            24       
  ConvBNLayer-7    [[1, 22, 16, 160]]       [1, 6, 16, 160]             0       
    Conv2D-8       [[1, 6, 16, 160]]       [1, 22, 16, 160]           1,188     
   BatchNorm-8     [[1, 22, 16, 160]]      [1, 22, 16, 160]            88       
  ConvBNLayer-8    [[1, 6, 16, 160]]       [1, 22, 16, 160]             0       
  BasicBlock-2     [[1, 22, 16, 160]]      [1, 22, 16, 160]             0       
    Conv2D-9       [[1, 22, 16, 160]]       [1, 13, 8, 160]           2,574     
   BatchNorm-9     [[1, 13, 8, 160]]        [1, 13, 8, 160]            52       
  ConvBNLayer-9    [[1, 22, 16, 160]]       [1, 13, 8, 160]             0       
    Conv2D-10      [[1, 13, 8, 160]]        [1, 33, 8, 160]           3,861     
  BatchNorm-10     [[1, 33, 8, 160]]        [1, 33, 8, 160]            132      
 ConvBNLayer-10    [[1, 13, 8, 160]]        [1, 33, 8, 160]             0       
  AvgPool2D-11     [[1, 22, 16, 160]]       [1, 22, 8, 160]             0       
    Conv2D-11      [[1, 22, 8, 160]]        [1, 33, 8, 160]            726      
  BatchNorm-11     [[1, 33, 8, 160]]        [1, 33, 8, 160]            132      
 ConvBNLayer-11    [[1, 22, 16, 160]]       [1, 33, 8, 160]             0       
  BasicBlock-3     [[1, 22, 16, 160]]       [1, 33, 8, 160]             0       
    Conv2D-12      [[1, 33, 8, 160]]        [1, 13, 8, 160]           3,861     
  BatchNorm-12     [[1, 13, 8, 160]]        [1, 13, 8, 160]            52       
 ConvBNLayer-12    [[1, 33, 8, 160]]        [1, 13, 8, 160]             0       
    Conv2D-13      [[1, 13, 8, 160]]        [1, 33, 8, 160]           3,861     
  BatchNorm-13     [[1, 33, 8, 160]]        [1, 33, 8, 160]            132      
 ConvBNLayer-13    [[1, 13, 8, 160]]        [1, 33, 8, 160]             0       
  BasicBlock-4     [[1, 33, 8, 160]]        [1, 33, 8, 160]             0       
    Conv2D-14      [[1, 33, 8, 160]]        [1, 26, 4, 160]           7,722     
  BatchNorm-14     [[1, 26, 4, 160]]        [1, 26, 4, 160]            104      
 ConvBNLayer-14    [[1, 33, 8, 160]]        [1, 26, 4, 160]             0       
    Conv2D-15      [[1, 26, 4, 160]]        [1, 96, 4, 160]          22,464     
  BatchNorm-15     [[1, 96, 4, 160]]        [1, 96, 4, 160]            384      
 ConvBNLayer-15    [[1, 26, 4, 160]]        [1, 96, 4, 160]             0       
  AvgPool2D-16     [[1, 33, 8, 160]]        [1, 33, 4, 160]             0       
    Conv2D-16      [[1, 33, 4, 160]]        [1, 96, 4, 160]           3,168     
  BatchNorm-16     [[1, 96, 4, 160]]        [1, 96, 4, 160]            384      
 ConvBNLayer-16    [[1, 33, 8, 160]]        [1, 96, 4, 160]             0       
  BasicBlock-5     [[1, 33, 8, 160]]        [1, 96, 4, 160]             0       
    Conv2D-17      [[1, 96, 4, 160]]        [1, 26, 4, 160]          22,464     
  BatchNorm-17     [[1, 26, 4, 160]]        [1, 26, 4, 160]            104      
 ConvBNLayer-17    [[1, 96, 4, 160]]        [1, 26, 4, 160]             0       
    Conv2D-18      [[1, 26, 4, 160]]        [1, 96, 4, 160]          22,464     
  BatchNorm-18     [[1, 96, 4, 160]]        [1, 96, 4, 160]            384      
 ConvBNLayer-18    [[1, 26, 4, 160]]        [1, 96, 4, 160]             0       
  BasicBlock-6     [[1, 96, 4, 160]]        [1, 96, 4, 160]             0       
    Conv2D-19      [[1, 96, 4, 160]]        [1, 57, 2, 160]          49,248     
  BatchNorm-19     [[1, 57, 2, 160]]        [1, 57, 2, 160]            228      
 ConvBNLayer-19    [[1, 96, 4, 160]]        [1, 57, 2, 160]             0       
    Conv2D-20      [[1, 57, 2, 160]]       [1, 512, 2, 160]          262,656    
  BatchNorm-20     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-20    [[1, 57, 2, 160]]       [1, 512, 2, 160]             0       
  AvgPool2D-21     [[1, 96, 4, 160]]        [1, 96, 2, 160]             0       
    Conv2D-21      [[1, 96, 2, 160]]       [1, 512, 2, 160]          49,152     
  BatchNorm-21     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-21    [[1, 96, 4, 160]]       [1, 512, 2, 160]             0       
  BasicBlock-7     [[1, 96, 4, 160]]       [1, 512, 2, 160]             0       
    Conv2D-22      [[1, 512, 2, 160]]      [1, 114, 2, 160]          525,312    
  BatchNorm-22     [[1, 114, 2, 160]]      [1, 114, 2, 160]            456      
 ConvBNLayer-22    [[1, 512, 2, 160]]      [1, 114, 2, 160]             0       
    Conv2D-23      [[1, 114, 2, 160]]      [1, 512, 2, 160]          525,312    
  BatchNorm-23     [[1, 512, 2, 160]]      [1, 512, 2, 160]           2,048     
 ConvBNLayer-23    [[1, 114, 2, 160]]      [1, 512, 2, 160]             0       
  BasicBlock-8     [[1, 512, 2, 160]]      [1, 512, 2, 160]             0       
   MaxPool2D-2     [[1, 512, 2, 160]]       [1, 512, 1, 80]             0       
    ResNet-1       [[1, 3, 32, 320]]        [1, 512, 1, 80]             0       
    Im2Seq-1       [[1, 512, 1, 80]]         [1, 80, 512]               0       
      GRU-1          [[1, 80, 512]]    [[1, 80, 96], [4, 1, 48]]     203,904    
EncoderWithRNN-1     [[1, 80, 512]]           [1, 80, 96]               0       
SequenceEncoder-1  [[1, 512, 1, 80]]          [1, 80, 96]               0       
    Linear-1         [[1, 80, 96]]           [1, 80, 3939]           382,083    
    CTCHead-1        [[1, 80, 96]]           [1, 80, 3939]              0       
==================================================================================
Total params: 2,104,710
Trainable params: 2,095,642
Non-trainable params: 9,068
----------------------------------------------------------------------------------
Input size (MB): 0.12
Forward/backward pass size (MB): 52.56
Params size (MB): 8.03
Estimated Total Size (MB): 60.71
----------------------------------------------------------------------------------
```