import torch

'''_author = Yvan Tamdjo'''

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))


def swish(x):
  return x * sigmoid(x)