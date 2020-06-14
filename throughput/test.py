from statistic import *

import torch
from torchvision import models
from thop import profile


#model = resnet50()
model = models.densenet161()
input = torch.randn(1, 3, 224, 224)

with Throughput(model, (input, )) as tp:
    # compute ...
    time.sleep(1)

print("average throughput is: %s GFLOPS" % (tp.val()))
