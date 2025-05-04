import torch.nn as nn
import torchvision.models as models

class GoogLeNetEncoder(nn.Module):
  def __init__(self, output_size=14):
    super(GoogLeNetEncoder, self).__init__()
    self.output_size = output_size

    googlenet = models.googlenet(pretrained=True)
    self.model = nn.Sequential(*list(googlenet.children())[:-2])
    self.pool = nn.AdaptiveAvgPool2d((output_size, output_size))
    self.fine_tune()

  def forward(self, x):
    out = self.model(x)
    out = self.pool(out)
    out = out.permute(0, 2, 3, 1)
    return out

  def fine_tune(self, fine_tune=True):
    for param in self.model.parameters():
      param.requires_grad = False

    if fine_tune:
      for child in list(self.model.children())[6:]:
        for param in child.parameters():
          param.requires_grad = fine_tune