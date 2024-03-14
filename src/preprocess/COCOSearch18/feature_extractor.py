from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import PIL
import os
from os.path import join, isdir, isfile
import numpy as np
import argparse

class ResNetCOCO(nn.Module):
    def __init__(self, device="cuda:0"):
        super(ResNetCOCO, self).__init__()
        self.resnet = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        bs, ch, _, _ = x.size()
        x = x.view(bs, ch, -1).permute(0, 2, 1)

        return x
def image_data(dataset_path, device='cuda:0', overwrite=False):
    resize_dim = (384 * 2, 512 * 2)
    src_path = join(dataset_path, 'images/')
    target_path = join(dataset_path, '../image_features/')
    resize = T.Resize(resize_dim)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    folders = [i for i in os.listdir(src_path) if isdir(join(src_path, i))]

    bbone = ResNetCOCO(device=device).to(device)
    for folder in folders:
        if not (os.path.exists(target_path) and os.path.isdir(target_path)):
            os.mkdir(target_path)
        files = [i for i in os.listdir(join(src_path, folder)) if
                 isfile(join(src_path, folder, i)) and i.endswith('.jpg')]
        for f in files:
            if overwrite == False and os.path.exists(join(target_path, f.replace('jpg', 'pth'))):
                continue
            PIL_image = PIL.Image.open(join(src_path, folder, f))
            tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0)

            features = bbone(tensor_image).squeeze().detach().cpu()
            torch.save(features, join(target_path, f.replace('jpg', 'pth')))




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gazeformer Feature Extractor Utils', add_help=False)
    parser.add_argument('--dataset_path', default='/home/COCO/TA', type=str)
    parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    image_data(dataset_path=args.dataset_path, device=device, overwrite=True)
