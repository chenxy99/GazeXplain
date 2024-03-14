# GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths

This code implements the prediction of visual scanpath along with its corresponding natural language explanations in three different tasks (3 different datasets) with two different architecture:

- Free-viewing: the prediction of scanpath for looking at some salient or important object in the given image. (OSIE)
- Visual Question Answering:  the prediction of scanpath during human performing general tasks, e.g., visual question answering, to reflect their attending and reasoning processes. (AiR-D)
- Visual search: the prediction of scanpath during the search of the given target object to reflect the goal-directed behavior under target present and absent conditions. (COCO-Search18 Target-Present and Target-Absent)

Reference
------------------
If you use our code or data, please cite our paper:
```text
Anonymous submission for ECCV 2024, paper ID 1308.
```

Disclaimer
------------------
For the ScanMatch evaluation metric, we adopt the part of [`GazeParser`](http://gazeparser.sourceforge.net/) package. 
We adopt the implementation of SED and STDE from [`VAME`](https://github.com/dariozanca/VAME) as two of our evaluation metrics mentioned in the [`Visual Attention Models`](https://ieeexplore.ieee.org/document/9207438). 
More specific, we adopt the evaluation metrics provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer), respectively.
Based on the [`checkpoint`](https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/checkpointing.py) implementation from [`updown-baseline`](https://github.com/nocaps-org/updown-baseline), we slightly modify it to accommodate our pipeline.

Requirements
------------------

- Python 3.10
- PyTorch 2.1.2 (along with torchvision)

- We also provide the conda environment ``environment.yml``, you can directly run

```bash
$ conda env create -f environment.yml
```

to create the same environment where we successfully run our codes.

Datasets
------------------

To process the data, you can follow the instructions provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer). 
For handling the SS cluster, you can refer to [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer) and [`Target-absent-Human-Attention`](https://github.com/cvlab-stonybrook/Target-absent-Human-Attention).
More specifically, you can run the following scripts to process the data

```bash
$ python ./src/preprocess/${dataset}/preprocess_fixations.py
```

```bash
$ python ./src/preprocess/${dataset}/feature_extractor.py
```


We structure `<dataset_root>` as follows

Training your own network on ALL the datasets
------------------

We set all the corresponding hyper-parameters in ``opt.py``. 

The `train_explanation_alignment.py` script will dump checkpoints into the folder specified by `--log_root` (default = `./runs/`). You can also set the other hyper-parameters in `opt.py` or define them in the `bash/train.sh`.

- `--datasets` Folder to the dataset, e.g., `<dataset_root>`.
- `--epoch` The number of total epochs.
- `--start_rl_epoch` Start to use reinforcement learning at the predefined epoch.

You can also use the following commands to train your own network. Then you can run the following commands to evaluate the performance of your trained model on test split.
```bash
$ sh bash/train.sh
```

Evaluate on test split
------------------
```bash
$ sh bash/test.sh
```
