from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
import PIL
import os
from os.path import join, isdir, isfile
import numpy as np
import argparse
import json


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
    src_path = join(dataset_path, 'stimuli/')
    target_path = join(dataset_path, 'image_features/')
    resize = T.Resize(resize_dim)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    bbone = ResNetCOCO(device=device).to(device).eval()
    files = [i for i in os.listdir(src_path) if isfile(join(src_path, i)) and i.endswith('.jpg')]
    for f in files:
        if overwrite == False and os.path.exists(join(target_path, f.replace('jpg', 'pth'))):
            continue
        PIL_image = PIL.Image.open(join(src_path, f))
        tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0)

        features = bbone(tensor_image).squeeze().detach().cpu()
        torch.save(features, join(target_path, f.replace('jpg', 'pth')))


def text_data(dataset_path, device='cuda:0', lm_model='sentence-transformers/stsb-roberta-base-v2'):
    # src_path = join(dataset_path, 'images/')
    # tasks = [' '.join(i.split('_')) for i in os.listdir(src_path) if isdir(join(src_path, i))]

    lm = SentenceTransformer(lm_model, device=device).eval()
    embed_dict = {}
    qid_to_question = {}
    for value in ["train", "validation", "test"]:
        fixations_file = join(dataset_path, "processed_data", "AiR_fixations_{}.json".format(value))
        with open(fixations_file) as json_file:
            fixations = json.load(json_file)
        for fixation in fixations:
            qid_to_question[fixation["question_id"]] = fixation["question"]

    for qid, question in qid_to_question.items():
        embed_dict[qid] = lm.encode(question)
    with open(join(dataset_path, 'embeddings.npy'), 'wb') as f:
        np.save(f, embed_dict, allow_pickle=True)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gazeformer Feature Extractor Utils', add_help=False)
    parser.add_argument('--dataset_path', default='/srv/data/AiR', type=str)
    parser.add_argument('--lm_model', default='sentence-transformers/stsb-roberta-base-v2', type=str)
    parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    # image_data(dataset_path = args.dataset_path, device = device, overwrite = True)
    text_data(dataset_path=args.dataset_path, device=device, lm_model=args.lm_model)