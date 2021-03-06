# Model summary
```
--------------------------------------------------------------------------------------------
    Layer (type)             Input Shape               Output Shape            Param #    
============================================================================================
      Conv2D-1            [[1, 3, 32, 320]]          [1, 16, 32, 320]            432      
     BatchNorm-1          [[1, 16, 32, 320]]         [1, 16, 32, 320]            64       
    ConvBNLayer-1         [[1, 3, 32, 320]]          [1, 16, 32, 320]             0       
      Conv2D-2            [[1, 16, 16, 160]]         [1, 32, 16, 160]           4,608     
     BatchNorm-2          [[1, 32, 16, 160]]         [1, 32, 16, 160]            128      
    ConvBNLayer-2         [[1, 16, 16, 160]]         [1, 32, 16, 160]             0       
      Conv2D-3             [[1, 32, 8, 80]]           [1, 64, 8, 80]           18,432     
     BatchNorm-3           [[1, 64, 8, 80]]           [1, 64, 8, 80]             256      
    ConvBNLayer-3          [[1, 32, 8, 80]]           [1, 64, 8, 80]              0       
      Conv2D-4             [[1, 64, 4, 40]]           [1, 128, 4, 40]          73,728     
     BatchNorm-4          [[1, 128, 4, 40]]           [1, 128, 4, 40]            512      
    ConvBNLayer-4          [[1, 64, 4, 40]]           [1, 128, 4, 40]             0       
      Linear-1                [[1, 128]]                  [1, 64]               8,256     
      Linear-2                [[1, 64]]                   [1, 40]               2,600     
LocalizationNetwork-1     [[1, 3, 32, 320]]             [1, 20, 2]                0       
      Linear-3                [[1, 40]]                   [1, 6]                 246      
   GridGenerator-1    [[1, 20, 2], [None, None]]       [1, 10240, 2]              0       
        TPS-1             [[1, 3, 32, 320]]           [1, 3, 32, 320]             0       
      Conv2D-5            [[1, 3, 32, 320]]          [1, 16, 16, 160]            432      
     BatchNorm-5          [[1, 16, 16, 160]]         [1, 16, 16, 160]            64       
    ConvBNLayer-5         [[1, 3, 32, 320]]          [1, 16, 16, 160]             0       
      Conv2D-6            [[1, 16, 16, 160]]         [1, 16, 16, 160]            256      
     BatchNorm-6          [[1, 16, 16, 160]]         [1, 16, 16, 160]            64       
    ConvBNLayer-6         [[1, 16, 16, 160]]         [1, 16, 16, 160]             0       
      Conv2D-7            [[1, 16, 16, 160]]         [1, 16, 16, 160]            144      
     BatchNorm-7          [[1, 16, 16, 160]]         [1, 16, 16, 160]            64       
    ConvBNLayer-7         [[1, 16, 16, 160]]         [1, 16, 16, 160]             0       
 AdaptiveAvgPool2D-2      [[1, 16, 16, 160]]           [1, 16, 1, 1]              0       
      Conv2D-8             [[1, 16, 1, 1]]             [1, 4, 1, 1]              68       
      Conv2D-9              [[1, 4, 1, 1]]             [1, 16, 1, 1]             80       
     SEModule-1           [[1, 16, 16, 160]]         [1, 16, 16, 160]             0       
      Conv2D-10           [[1, 16, 16, 160]]         [1, 16, 16, 160]            256      
     BatchNorm-8          [[1, 16, 16, 160]]         [1, 16, 16, 160]            64       
    ConvBNLayer-8         [[1, 16, 16, 160]]         [1, 16, 16, 160]             0       
   ResidualUnit-1         [[1, 16, 16, 160]]         [1, 16, 16, 160]             0       
      Conv2D-11           [[1, 16, 16, 160]]         [1, 72, 16, 160]           1,152     
     BatchNorm-9          [[1, 72, 16, 160]]         [1, 72, 16, 160]            288      
    ConvBNLayer-9         [[1, 16, 16, 160]]         [1, 72, 16, 160]             0       
      Conv2D-12           [[1, 72, 16, 160]]          [1, 72, 8, 160]            648      
    BatchNorm-10          [[1, 72, 8, 160]]           [1, 72, 8, 160]            288      
   ConvBNLayer-10         [[1, 72, 16, 160]]          [1, 72, 8, 160]             0       
      Conv2D-13           [[1, 72, 8, 160]]           [1, 24, 8, 160]           1,728     
    BatchNorm-11          [[1, 24, 8, 160]]           [1, 24, 8, 160]            96       
   ConvBNLayer-11         [[1, 72, 8, 160]]           [1, 24, 8, 160]             0       
   ResidualUnit-2         [[1, 16, 16, 160]]          [1, 24, 8, 160]             0       
      Conv2D-14           [[1, 24, 8, 160]]           [1, 88, 8, 160]           2,112     
    BatchNorm-12          [[1, 88, 8, 160]]           [1, 88, 8, 160]            352      
   ConvBNLayer-12         [[1, 24, 8, 160]]           [1, 88, 8, 160]             0       
      Conv2D-15           [[1, 88, 8, 160]]           [1, 88, 8, 160]            792      
    BatchNorm-13          [[1, 88, 8, 160]]           [1, 88, 8, 160]            352      
   ConvBNLayer-13         [[1, 88, 8, 160]]           [1, 88, 8, 160]             0       
      Conv2D-16           [[1, 88, 8, 160]]           [1, 24, 8, 160]           2,112     
    BatchNorm-14          [[1, 24, 8, 160]]           [1, 24, 8, 160]            96       
   ConvBNLayer-14         [[1, 88, 8, 160]]           [1, 24, 8, 160]             0       
   ResidualUnit-3         [[1, 24, 8, 160]]           [1, 24, 8, 160]             0       
      Conv2D-17           [[1, 24, 8, 160]]           [1, 96, 8, 160]           2,304     
    BatchNorm-15          [[1, 96, 8, 160]]           [1, 96, 8, 160]            384      
   ConvBNLayer-15         [[1, 24, 8, 160]]           [1, 96, 8, 160]             0       
      Conv2D-18           [[1, 96, 8, 160]]           [1, 96, 4, 160]           2,400     
    BatchNorm-16          [[1, 96, 4, 160]]           [1, 96, 4, 160]            384      
   ConvBNLayer-16         [[1, 96, 8, 160]]           [1, 96, 4, 160]             0       
 AdaptiveAvgPool2D-3      [[1, 96, 4, 160]]            [1, 96, 1, 1]              0       
      Conv2D-19            [[1, 96, 1, 1]]             [1, 24, 1, 1]            2,328     
      Conv2D-20            [[1, 24, 1, 1]]             [1, 96, 1, 1]            2,400     
     SEModule-2           [[1, 96, 4, 160]]           [1, 96, 4, 160]             0       
      Conv2D-21           [[1, 96, 4, 160]]           [1, 40, 4, 160]           3,840     
    BatchNorm-17          [[1, 40, 4, 160]]           [1, 40, 4, 160]            160      
   ConvBNLayer-17         [[1, 96, 4, 160]]           [1, 40, 4, 160]             0       
   ResidualUnit-4         [[1, 24, 8, 160]]           [1, 40, 4, 160]             0       
      Conv2D-22           [[1, 40, 4, 160]]          [1, 240, 4, 160]           9,600     
    BatchNorm-18          [[1, 240, 4, 160]]         [1, 240, 4, 160]            960      
   ConvBNLayer-18         [[1, 40, 4, 160]]          [1, 240, 4, 160]             0       
      Conv2D-23           [[1, 240, 4, 160]]         [1, 240, 4, 160]           6,000     
    BatchNorm-19          [[1, 240, 4, 160]]         [1, 240, 4, 160]            960      
   ConvBNLayer-19         [[1, 240, 4, 160]]         [1, 240, 4, 160]             0       
 AdaptiveAvgPool2D-4      [[1, 240, 4, 160]]          [1, 240, 1, 1]              0       
      Conv2D-24            [[1, 240, 1, 1]]            [1, 60, 1, 1]           14,460     
      Conv2D-25            [[1, 60, 1, 1]]            [1, 240, 1, 1]           14,640     
     SEModule-3           [[1, 240, 4, 160]]         [1, 240, 4, 160]             0       
      Conv2D-26           [[1, 240, 4, 160]]          [1, 40, 4, 160]           9,600     
    BatchNorm-20          [[1, 40, 4, 160]]           [1, 40, 4, 160]            160      
   ConvBNLayer-20         [[1, 240, 4, 160]]          [1, 40, 4, 160]             0       
   ResidualUnit-5         [[1, 40, 4, 160]]           [1, 40, 4, 160]             0       
      Conv2D-27           [[1, 40, 4, 160]]          [1, 240, 4, 160]           9,600     
    BatchNorm-21          [[1, 240, 4, 160]]         [1, 240, 4, 160]            960      
   ConvBNLayer-21         [[1, 40, 4, 160]]          [1, 240, 4, 160]             0       
      Conv2D-28           [[1, 240, 4, 160]]         [1, 240, 4, 160]           6,000     
    BatchNorm-22          [[1, 240, 4, 160]]         [1, 240, 4, 160]            960      
   ConvBNLayer-22         [[1, 240, 4, 160]]         [1, 240, 4, 160]             0       
 AdaptiveAvgPool2D-5      [[1, 240, 4, 160]]          [1, 240, 1, 1]              0       
      Conv2D-29            [[1, 240, 1, 1]]            [1, 60, 1, 1]           14,460     
      Conv2D-30            [[1, 60, 1, 1]]            [1, 240, 1, 1]           14,640     
     SEModule-4           [[1, 240, 4, 160]]         [1, 240, 4, 160]             0       
      Conv2D-31           [[1, 240, 4, 160]]          [1, 40, 4, 160]           9,600     
    BatchNorm-23          [[1, 40, 4, 160]]           [1, 40, 4, 160]            160      
   ConvBNLayer-23         [[1, 240, 4, 160]]          [1, 40, 4, 160]             0       
   ResidualUnit-6         [[1, 40, 4, 160]]           [1, 40, 4, 160]             0       
      Conv2D-32           [[1, 40, 4, 160]]          [1, 120, 4, 160]           4,800     
    BatchNorm-24          [[1, 120, 4, 160]]         [1, 120, 4, 160]            480      
   ConvBNLayer-24         [[1, 40, 4, 160]]          [1, 120, 4, 160]             0       
      Conv2D-33           [[1, 120, 4, 160]]         [1, 120, 4, 160]           3,000     
    BatchNorm-25          [[1, 120, 4, 160]]         [1, 120, 4, 160]            480      
   ConvBNLayer-25         [[1, 120, 4, 160]]         [1, 120, 4, 160]             0       
 AdaptiveAvgPool2D-6      [[1, 120, 4, 160]]          [1, 120, 1, 1]              0       
      Conv2D-34            [[1, 120, 1, 1]]            [1, 30, 1, 1]            3,630     
      Conv2D-35            [[1, 30, 1, 1]]            [1, 120, 1, 1]            3,720     
     SEModule-5           [[1, 120, 4, 160]]         [1, 120, 4, 160]             0       
      Conv2D-36           [[1, 120, 4, 160]]          [1, 48, 4, 160]           5,760     
    BatchNorm-26          [[1, 48, 4, 160]]           [1, 48, 4, 160]            192      
   ConvBNLayer-26         [[1, 120, 4, 160]]          [1, 48, 4, 160]             0       
   ResidualUnit-7         [[1, 40, 4, 160]]           [1, 48, 4, 160]             0       
      Conv2D-37           [[1, 48, 4, 160]]          [1, 144, 4, 160]           6,912     
    BatchNorm-27          [[1, 144, 4, 160]]         [1, 144, 4, 160]            576      
   ConvBNLayer-27         [[1, 48, 4, 160]]          [1, 144, 4, 160]             0       
      Conv2D-38           [[1, 144, 4, 160]]         [1, 144, 4, 160]           3,600     
    BatchNorm-28          [[1, 144, 4, 160]]         [1, 144, 4, 160]            576      
   ConvBNLayer-28         [[1, 144, 4, 160]]         [1, 144, 4, 160]             0       
 AdaptiveAvgPool2D-7      [[1, 144, 4, 160]]          [1, 144, 1, 1]              0       
      Conv2D-39            [[1, 144, 1, 1]]            [1, 36, 1, 1]            5,220     
      Conv2D-40            [[1, 36, 1, 1]]            [1, 144, 1, 1]            5,328     
     SEModule-6           [[1, 144, 4, 160]]         [1, 144, 4, 160]             0       
      Conv2D-41           [[1, 144, 4, 160]]          [1, 48, 4, 160]           6,912     
    BatchNorm-29          [[1, 48, 4, 160]]           [1, 48, 4, 160]            192      
   ConvBNLayer-29         [[1, 144, 4, 160]]          [1, 48, 4, 160]             0       
   ResidualUnit-8         [[1, 48, 4, 160]]           [1, 48, 4, 160]             0       
      Conv2D-42           [[1, 48, 4, 160]]          [1, 288, 4, 160]          13,824     
    BatchNorm-30          [[1, 288, 4, 160]]         [1, 288, 4, 160]           1,152     
   ConvBNLayer-30         [[1, 48, 4, 160]]          [1, 288, 4, 160]             0       
      Conv2D-43           [[1, 288, 4, 160]]         [1, 288, 2, 160]           7,200     
    BatchNorm-31          [[1, 288, 2, 160]]         [1, 288, 2, 160]           1,152     
   ConvBNLayer-31         [[1, 288, 4, 160]]         [1, 288, 2, 160]             0       
 AdaptiveAvgPool2D-8      [[1, 288, 2, 160]]          [1, 288, 1, 1]              0       
      Conv2D-44            [[1, 288, 1, 1]]            [1, 72, 1, 1]           20,808     
      Conv2D-45            [[1, 72, 1, 1]]            [1, 288, 1, 1]           21,024     
     SEModule-7           [[1, 288, 2, 160]]         [1, 288, 2, 160]             0       
      Conv2D-46           [[1, 288, 2, 160]]          [1, 96, 2, 160]          27,648     
    BatchNorm-32          [[1, 96, 2, 160]]           [1, 96, 2, 160]            384      
   ConvBNLayer-32         [[1, 288, 2, 160]]          [1, 96, 2, 160]             0       
   ResidualUnit-9         [[1, 48, 4, 160]]           [1, 96, 2, 160]             0       
      Conv2D-47           [[1, 96, 2, 160]]          [1, 576, 2, 160]          55,296     
    BatchNorm-33          [[1, 576, 2, 160]]         [1, 576, 2, 160]           2,304     
   ConvBNLayer-33         [[1, 96, 2, 160]]          [1, 576, 2, 160]             0       
      Conv2D-48           [[1, 576, 2, 160]]         [1, 576, 2, 160]          14,400     
    BatchNorm-34          [[1, 576, 2, 160]]         [1, 576, 2, 160]           2,304     
   ConvBNLayer-34         [[1, 576, 2, 160]]         [1, 576, 2, 160]             0       
 AdaptiveAvgPool2D-9      [[1, 576, 2, 160]]          [1, 576, 1, 1]              0       
      Conv2D-49            [[1, 576, 1, 1]]           [1, 144, 1, 1]           83,088     
      Conv2D-50            [[1, 144, 1, 1]]           [1, 576, 1, 1]           83,520     
     SEModule-8           [[1, 576, 2, 160]]         [1, 576, 2, 160]             0       
      Conv2D-51           [[1, 576, 2, 160]]          [1, 96, 2, 160]          55,296     
    BatchNorm-35          [[1, 96, 2, 160]]           [1, 96, 2, 160]            384      
   ConvBNLayer-35         [[1, 576, 2, 160]]          [1, 96, 2, 160]             0       
   ResidualUnit-10        [[1, 96, 2, 160]]           [1, 96, 2, 160]             0       
      Conv2D-52           [[1, 96, 2, 160]]          [1, 576, 2, 160]          55,296     
    BatchNorm-36          [[1, 576, 2, 160]]         [1, 576, 2, 160]           2,304     
   ConvBNLayer-36         [[1, 96, 2, 160]]          [1, 576, 2, 160]             0       
      Conv2D-53           [[1, 576, 2, 160]]         [1, 576, 2, 160]          14,400     
    BatchNorm-37          [[1, 576, 2, 160]]         [1, 576, 2, 160]           2,304     
   ConvBNLayer-37         [[1, 576, 2, 160]]         [1, 576, 2, 160]             0       
AdaptiveAvgPool2D-10      [[1, 576, 2, 160]]          [1, 576, 1, 1]              0       
      Conv2D-54            [[1, 576, 1, 1]]           [1, 144, 1, 1]           83,088     
      Conv2D-55            [[1, 144, 1, 1]]           [1, 576, 1, 1]           83,520     
     SEModule-9           [[1, 576, 2, 160]]         [1, 576, 2, 160]             0       
      Conv2D-56           [[1, 576, 2, 160]]          [1, 96, 2, 160]          55,296     
    BatchNorm-38          [[1, 96, 2, 160]]           [1, 96, 2, 160]            384      
   ConvBNLayer-38         [[1, 576, 2, 160]]          [1, 96, 2, 160]             0       
   ResidualUnit-11        [[1, 96, 2, 160]]           [1, 96, 2, 160]             0       
      Conv2D-57           [[1, 144, 4, 160]]          [1, 48, 2, 160]          62,208     
    BatchNorm-39          [[1, 48, 2, 160]]           [1, 48, 2, 160]            192      
   ConvBNLayer-39         [[1, 144, 4, 160]]          [1, 48, 2, 160]             0       
      Conv2D-58           [[1, 72, 8, 160]]           [1, 24, 2, 160]          15,552     
    BatchNorm-40          [[1, 24, 2, 160]]           [1, 24, 2, 160]            96       
   ConvBNLayer-40         [[1, 72, 8, 160]]           [1, 24, 2, 160]             0       
      Conv2D-59           [[1, 168, 2, 160]]         [1, 576, 2, 160]          96,768     
    BatchNorm-41          [[1, 576, 2, 160]]         [1, 576, 2, 160]           2,304     
   ConvBNLayer-41         [[1, 168, 2, 160]]         [1, 576, 2, 160]             0       
     MaxPool2D-4          [[1, 576, 2, 160]]          [1, 576, 1, 80]             0       
  MobileNetV3_FPN-1       [[1, 3, 32, 320]]           [1, 576, 1, 80]             0       
      Im2Seq-1            [[1, 576, 1, 80]]            [1, 80, 576]               0       
        GRU-1               [[1, 80, 576]]       [[1, 80, 96], [4, 1, 48]]     222,336    
  EncoderWithRNN-1          [[1, 80, 576]]              [1, 80, 96]               0       
  SequenceEncoder-1       [[1, 576, 1, 80]]             [1, 80, 96]               0       
      Linear-4              [[1, 80, 96]]              [1, 80, 3939]           382,083    
      CTCHead-1             [[1, 80, 96]]              [1, 80, 3939]              0       
============================================================================================
Total params: 1,767,023
Trainable params: 1,741,487
Non-trainable params: 25,536
--------------------------------------------------------------------------------------------
Input size (MB): 0.12
Forward/backward pass size (MB): 111.04
Params size (MB): 6.74
Estimated Total Size (MB): 117.90
--------------------------------------------------------------------------------------------
```