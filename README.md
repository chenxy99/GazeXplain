# GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths

This code implements the prediction of visual scanpath along with its corresponding natural language explanations in three different tasks (3 different datasets) with two different architecture:

- Free-viewing: the prediction of scanpath for looking at some salient or important object in the given image. (OSIE)
- Visual Question Answering:  the prediction of scanpath during human performing general tasks, e.g., visual question answering, to reflect their attending and reasoning processes. (AiR-D)
- Visual search: the prediction of scanpath during the search of the given target object to reflect the goal-directed behavior under target present and absent conditions. (COCO-Search18 Target-Present and Target-Absent)

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2408.02788)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/chenxy99/GazeXplain)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=kTv-dDUBOd8)

:fire: News <a name="news"></a>
------------------
- `[2024/07]` GazeXplain code and [datasets](#datasets) initially released.

:mega: Overview
------------------
![overall_structure](./asset/teaser.png)
We introduce ``GazeXplain``, a novel scanpath explanation task to understand human visual attention. We provide ground-truth explanations on various eye-tracking datasets and develop a model architecture for predicting scanpaths and generating natural language explanations.
This example reveals how observers strategically investigate a scene to find out if the person is walking on the sidewalk. Fixations (circles) start centrally, locating a driving car, then shifting to the sidewalk to find the person, and finally looking down to confirm if the person is walking. By annotating observers' scanpaths with detailed explanations, we enable a deeper understanding of the what and why behind fixations, providing insights into human decision-making and task performance.


:bowing_man: Disclaimer
------------------
For the ScanMatch evaluation metric, we adopt the part of [`GazeParser`](http://gazeparser.sourceforge.net/) package. 
We adopt the implementation of SED and STDE from [`VAME`](https://github.com/dariozanca/VAME) as two of our evaluation metrics mentioned in the [`Visual Attention Models`](https://ieeexplore.ieee.org/document/9207438). 
More specific, we adopt the evaluation metrics provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer), respectively.
Based on the [`checkpoint`](https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/checkpointing.py) implementation from [`updown-baseline`](https://github.com/nocaps-org/updown-baseline), we slightly modify it to accommodate our pipeline.

:white_check_mark: Requirements
------------------

- Python 3.10
- PyTorch 2.1.2 (along with torchvision)

- We also provide the conda environment ``environment.yml``, you can directly run

```bash
$ conda env create -f environment.yml
```

to create the same environment where we successfully run our codes.

:bookmark_tabs: Datasets <a name="datasets"></a>
------------------

Our GazeXplain dataset is released! You can download the dataset from [`Link`](https://drive.google.com/drive/folders/13-0j4wkCmab_8Uge30bwCd-vJ1k6gxzO?usp=sharing). 
This dataset contains the explanations of visual scanpaths in three different scanpath datasets (OSIE, AiR-D, COCO-Search18).

:computer: Preprocess
------------------

To process the data, you can follow the instructions provided in [`Scanpath`](https://github.com/chenxy99/Scanpaths) and [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer). 
For handling the SS cluster, you can refer to [`Gazeformer`](https://github.com/cvlab-stonybrook/Gazeformer) and [`Target-absent-Human-Attention`](https://github.com/cvlab-stonybrook/Target-absent-Human-Attention).
More specifically, you can run the following scripts to process the data.

```bash
$ python ./src/preprocess/${dataset}/preprocess_fixations.py
```

```bash
$ python ./src/preprocess/${dataset}/feature_extractor.py
```


We structure `<dataset_root>` as follows

:runner: Training your own network on ALL the datasets
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

:bullettrain_front:	Evaluate on test split
------------------
For inference, we provide the [`pretrained model`](https://drive.google.com/file/d/10WfTJOeF4LjsmILUTb0Z0tVOgdu0P21Q/view?usp=sharing), and you can directly run the following command to evaluate the performance of the pretrained model on test split.
```bash
$ sh bash/test.sh
```

:black_nib: Citation
------------------
If you use our code or data, please cite our paper:
```text
@inproceedings{xianyu:2024:gazexplain,
    Author         = {Xianyu Chen and Ming Jiang and Qi Zhao},
    Title          = {GazeXplain: Learning to Predict Natural Language Explanations of Visual Scanpaths},
    booktitle      = {Proceedings of the European Conference on Computer Vision (ECCV)},
    Year           = {2024}
}
```