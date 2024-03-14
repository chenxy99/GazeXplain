from __future__ import print_function
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")
    parser.add_argument("--mode", type=str, default="train", help="Selecting running mode (default: train)")
    parser.add_argument('--stimuli_dir', default="/home/AiR/stimuli", help='stimuli folder')
    parser.add_argument('--fixation_dir', default="/home/AiR/processed_data", help='fixation folder')
    parser.add_argument('--feature_dir', default="/home/AiR/image_features", help='feature folder')
    parser.add_argument('--dataset_dir', default="/home/", help='feature folder')
    parser.add_argument('--datasets', default=["AiR-D", "OSIE", "COCO-TP", "COCO-TA"], nargs='+', help='used dataset')
    parser.add_argument('--tiny', action="store_true", help='use the tiny dataset in debug')

    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--test_batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--pct_start", type=float, default=0.05, help="The percentage of the cycle "
                                                                     "(in number of steps) spent"
                                                                     "increasing the learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--resume_dir", type=str, default="", help="Resume from a specific directory")
    parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation")
    parser.add_argument("--dropout", type=float, default=0.2, help="The dropout rate applied to the model")
    parser.add_argument("--checkpoint_every", type=int, default=1, help="how often to save a model checkpoint (in epochs)?")

    # image information
    parser.add_argument("--width", type=int, default=512, help="Width of input data")
    parser.add_argument("--height", type=int, default=384, help="Height of input data")
    parser.add_argument('--im_h', default=24, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")

    # fixation
    parser.add_argument("--blur_sigma", type=float, default=None, help="Standard deviation for Gaussian kernel")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")

    # model
    parser.add_argument('--patch_size', default=16, type=int,
                        help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
    parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
    parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
    parser.add_argument('--lm_hidden_dim', default=768, type=int,
                        help="Dimensionality of target embeddings from language model")
    parser.add_argument('--encoder_dropout', default=0.1, type=float, help="Encoder dropout rate")
    parser.add_argument('--decoder_dropout', default=0.2, type=float, help="Decoder and fusion step dropout rate")
    parser.add_argument('--cls_dropout', default=0.4, type=float, help="Final scanpath prediction dropout rate")
    parser.add_argument("--activation", default="relu", type=str, help="activation function in transformer")
    parser.add_argument("--normalize_before", action="store_true",
                        help="Normalize before in transformer")
    parser.add_argument("--return_intermediate_dec", action="store_true",
                        help="Return the intermediate data from decoder")

    # accelerate settings
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--project_dir", type=str, default="./runs/runX",
        help="Location on where to store experiment tracking logs` and relevent project information",
    )
    parser.add_argument(
        "--project_name", type=str, default="Baseline",
        help="Project name where the log information are stored",
    )

    # scst settings
    parser.add_argument("--warmup_epoch", type=int, default=1, help="Epoch when finishing warn up strategy")
    parser.add_argument("--start_rl_epoch", type=int, default=8, help="Epoch when starting reinforcement learning")
    parser.add_argument("--rl_sample_number", type=int, default=3,
                        help="Number of samples used in policy gradient update")
    parser.add_argument("--rl_lr_initial_decay", type=float, default=0.1, help="Initial decay of learning rate of rl")
    parser.add_argument("--checkpoint_every_rl", type=int, default=1,
                        help="how often to save a model checkpoint (in epochs) when using SCST?")
    parser.add_argument("--supervised_save", type=bool, default=True,
                        help="Copy the files before start the policy gradient update")

    # explanation modules
    parser.add_argument("--explanation", action="store_true", help="Use explanation module")
    parser.add_argument("--max_generation_length",  type=int, default=20, help="Max length when generate the explanation")
    parser.add_argument("--num_explanation_beams", type=int, default=3, help="Number of beam in explanation")



    # loss hyperparameters
    parser.add_argument("--lambda_duration", type=float, default=1.0, help="Hyper-parameter for duration loss term")
    parser.add_argument("--lambda_lm", type=float, default=1.0, help="Hyper-parameter for language model loss term")
    parser.add_argument("--lambda_alignment", type=float, default=1.0, help="Hyper-parameter for language model loss term")

    parser.add_argument("--max_explanation_length", type=int, default=20, help="Max explanation length")
    parser.add_argument("--min_explanation_length", type=int, default=5, help="Min explanation length")


    # log
    parser.add_argument("--resume_wandb", action="store_true", help="If passed, resume the wandb.")

    # config
    parser.add_argument('--cfg', type=str, default=None,
                        help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from lib.utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k, v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    return args
