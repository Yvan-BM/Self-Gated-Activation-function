import torch.nn as nn
import torch.nn.init as init


def _weights_init(m):
  classname = m.__class__.__name__
  #print(classname)
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
  def __init__(self, lambd):
    super(LambdaLayer, self).__init__()
    self.lambd = lambd

  def forward(self, x):
    return self.lambd(x)