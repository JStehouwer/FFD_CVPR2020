from imageio import imread
import torch
from torchvision import transforms

def get_templates():
  templates_list = []
  for i in range(10):
    img = imread('./MCT/template{:d}.png'.format(i))
    templates_list.append(transforms.functional.to_tensor(img)[0:1,0:19,0:19])
  templates = torch.stack(templates_list).cuda()
  templates = templates.squeeze(1)
  return templates

