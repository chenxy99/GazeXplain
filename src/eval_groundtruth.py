import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import os
import argparse
from tqdm import tqdm
import json
import copy

from accelerate import Accelerator
from accelerate.utils import tqdm

from lib.evaluator_groundtruth import eval
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="Scanpath evaluation")
parser.add_argument('--save_metric', action='store_false')
parser.add_argument('--split', default='test')
parser.add_argument("--test_batch", type=int, default=16, help="Batch size")
parser.add_argument('--datasets', default=["AiR-D", "OSIE", "COCO-TP", "COCO-TA"], nargs='+', help='used dataset')
parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation")
parser.add_argument('--tiny', action="store_true", help='use the tiny dataset in debug')
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
parser.add_argument(
    "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16", "fp8"],
    help="Whether to use mixed precision. Choose"
         "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
         "and an Nvidia Ampere GPU.",
)
parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")

parser.add_argument('--dataset_dir', default="/media/ECCV_2024", help='feature folder')
parser.add_argument("--max_explanation_length", type=int, default=20, help="Max explanation length")
parser.add_argument("--min_explanation_length", type=int, default=5, help="Min explanation length")

# image information
parser.add_argument("--width", type=int, default=512, help="Width of input data")
parser.add_argument("--height", type=int, default=384, help="Height of input data")
parser.add_argument('--im_h', default=24, type=int, help="Height of feature map input to encoder")
parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")

# fixation
parser.add_argument("--blur_sigma", type=float, default=None, help="Standard deviation for Gaussian kernel")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
args = parser.parse_args()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
# These five lines control all the major sources of randomness.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def main():
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    eval(accelerator, args, split=args.split)


if __name__ == "__main__":
    main()
