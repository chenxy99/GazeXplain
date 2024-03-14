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

from lib.explanation_alignment_evaluator import eval
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

    weights_bases = [
        './runs/ALL_runX_baseline',
    ]

    for base in weights_bases:
        accelerator.print('Evaluating {}...'.format(base))

        # update the argument
        opt = copy.deepcopy(args)
        hparams_file = os.path.join(base, "hparams.json")
        # read hparams
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        for k, v in hparams.items():
            if not hasattr(opt, k):
                accelerator.print('Warning: key %s not in args' % k)
            setattr(opt, k, v)

        opt.test_batch = args.test_batch
        opt.datasets = args.datasets
        opt.eval_repeat_num = args.eval_repeat_num
        opt.tiny = args.tiny


        model_path = os.path.join(base, "checkpoints/ckpt_best")
        if opt.save_metric:  # Save the final metric for the current best model
            save_path = os.path.join(base, opt.split)
        else:
            save_path = None

        eval(accelerator, model_path, opt, split=opt.split, save_path=save_path)


if __name__ == "__main__":
    main()
