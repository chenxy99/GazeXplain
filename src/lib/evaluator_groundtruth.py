"""Evaluation"""
from __future__ import print_function

import json
import os
import random
import time
import torch
import numpy as np
from torch import Tensor

from collections import OrderedDict

import scipy.stats
import sys
import copy
import random


import tempfile
from json import encoder

from torch.utils.data import DataLoader

from lib.dataset.dataset import UnifiedScanpath
from lib.evaluation.evaluator import Evaluator

from accelerate.utils import tqdm

encoder.FLOAT_REPR = lambda o: format(o, '.3f')



def get_evaluation(accelerator, data_loader, opt):
    """
    Get prediction
    """
    # Initialize the Evaluator
    evaluator = Evaluator(opt)

    # transform the gather scanpath prediction to JSON format file
    if accelerator.is_main_process:
        # get the fixation information
        scanpath_dataset = ["AiR-D", "OSIE", "COCO-TP", "COCO-TA"]
        dataset_to_idx = {
            "AiR-D": 0,
            "OSIE": 1,
            "COCO-TP": 2,
            "COCO-TA": 3,
        }
        json_gt_scanpaths = data_loader.dataset.fixations

        # normalize data
        # reshape to the same size 384 x 512
        for value in json_gt_scanpaths:
            value["X"] = (np.array(value["X"]) / value["width"] * opt.width).tolist()
            value["Y"] = (np.array(value["Y"]) / value["height"] * opt.height).tolist()

        # collect for saliency evaluation
        gt_fixation_dict = {}
        for iter in range(len(json_gt_scanpaths)):
            cur_dataset = json_gt_scanpaths[iter]["dataset"]
            if cur_dataset == "AiR-D":
                key = "{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["question_id"])
            elif cur_dataset == "OSIE":
                key = "{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["name"])
            elif cur_dataset == "COCO-TP":
                key = "{}-{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["task"], json_gt_scanpaths[iter]["name"])
            elif cur_dataset == "COCO-TA":
                key = "{}-{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["task"], json_gt_scanpaths[iter]["name"])
            else:
                raise "Invalid dataset"
            gt_fixation_dict.setdefault(key, []).append(json_gt_scanpaths[iter])

        # random assign for saliency prediction
        split_1_gt_fixation_dict = {}
        split_2_gt_fixation_dict = {}

        for key, value in gt_fixation_dict.items():
            split_1_num = len(value) // 2
            random.shuffle(value)
            split_1_gt_fixation_dict[key] = value[:split_1_num]
            split_2_gt_fixation_dict[key] = value[split_1_num:]

        targets = []
        predictions = []
        for key, value in gt_fixation_dict.items():
            # evaluation on the prediction and ground truth
            for idx in range(len(value)):
                for index in range(len(value)):
                    if idx < index:
                        targets.append(value[idx])
                        predictions.append(value[index])

        image_size = np.zeros((len(predictions), 2), np.int64)
        image_size[:, 0] = 384
        image_size[:, 1] = 512
        dataset_idx = [dataset_to_idx[_["dataset"]] for _ in targets]
        batch = {
            "image_size": image_size,
            "dataset_idx": dataset_idx,
            "fixation_info": targets
        }

        explanation_res = {}
        explanation_gts = {}
        for key, value in gt_fixation_dict.items():
            # evaluation on the prediction and ground truth
            for idx in range(len(value)):
                cur_key = "{}-{}".format(key, idx)
                explanation_res.setdefault(cur_key, []).append(
                    {
                        "caption": " ".join(value[idx]["explanation"]).strip(),
                    }
                )
                for index in range(len(value)):
                    if idx != index:
                        explanation_gts.setdefault(cur_key, []).append(
                            {
                                "caption": " ".join(value[index]["explanation"]).strip(),
                            }
                        )


        all_saliency_evals = evaluator.eval_saliency(split_1_gt_fixation_dict, split_2_gt_fixation_dict)
        evaluation_scores = evaluator.measure_gt(targets, predictions, batch)
        # evaluation on captioning
        gts = explanation_gts
        res = explanation_res
        evaluator.explanation_evaluation(gts, res)

        dataset_evaluation_scores_dict = {}
        for iter, evaluation_score in enumerate(evaluation_scores):
            dataset_evaluation_scores_dict.setdefault(scanpath_dataset[dataset_idx[iter]], []).append(
                evaluation_score)

        dataset_caption_evaluation_scores_dict = {}
        for cur_dataset, evaluation_score in evaluator.scanpath_eval.imgToEval.items():
            if "AiR-D" in cur_dataset:
                key = "AiR-D"
            elif "OSIE" in cur_dataset:
                key = "OSIE"
            elif "COCO-TP" in cur_dataset:
                key = "COCO-TP"
            elif "COCO-TA" in cur_dataset:
                key = "COCO-TA"
            else:
                raise "Invalid dataset"
            dataset_caption_evaluation_scores_dict.setdefault(key, []).append(
                evaluation_score)


        dataset_gather_scores_dict = {}
        for dataset_name, evaluation_score in dataset_evaluation_scores_dict.items():
            if len(evaluation_score) > 0:
                if dataset_name in ["AiR-D", "OSIE"]:
                    saliency_eval = np.array( [v for k, v in all_saliency_evals.items() if dataset_name in k])
                    gather_score = {
                        "scanmatch_score": np.array([_["scanmatch_score"] for _ in evaluation_score]),
                        "multimatch_score": np.array([_["multimatch_score"] for _ in evaluation_score]),
                        "sed_score": np.array([_["sed_score"] for _ in evaluation_score]),
                        "stde_score": np.array([_["stde_score"] for _ in evaluation_score]),
                        "SS_score": np.array([_["SS_score"] for _ in evaluation_score]),
                        "CC": saliency_eval[:, 0],
                        "AUC": saliency_eval[:, 1],
                        "NSS": saliency_eval[:, 2],
                        "sAUC": saliency_eval[:, 3],
                        "KLD": saliency_eval[:, 4],
                        "SIM": saliency_eval[:, 5],
                    }
                else:
                    saliency_eval = np.array([v for k, v in all_saliency_evals.items() if dataset_name in k])
                    gather_score = {
                        "scanmatch_score": np.array([_["scanmatch_score"] for _ in evaluation_score]),
                        "multimatch_score": np.array([_["multimatch_score"] for _ in evaluation_score]),
                        "sed_score": np.array([_["sed_score"] for _ in evaluation_score]),
                        "stde_score": np.array([_["stde_score"] for _ in evaluation_score]),
                        "SS_score": np.array([_["SS_score"] for _ in evaluation_score]),
                        "SSS_score": np.array([_["SSS_score"] for _ in evaluation_score]),
                        "CC": saliency_eval[:, 0],
                        "AUC": saliency_eval[:, 1],
                        "NSS": saliency_eval[:, 2],
                        "sAUC": saliency_eval[:, 3],
                        "KLD": saliency_eval[:, 4],
                        "SIM": saliency_eval[:, 5],
                    }
                dataset_gather_scores_dict[dataset_name] = gather_score

        for dataset_name, evaluation_score in dataset_caption_evaluation_scores_dict.items():
            if len(evaluation_score) > 0:
                gather_score = dataset_gather_scores_dict.get(dataset_name, {})
                gather_score["Bleu_4"] = np.array([_["Bleu_4"] for _ in evaluation_score])
                gather_score["METEOR"] = np.array([_["METEOR"] for _ in evaluation_score])
                gather_score["ROUGE_L"] = np.array([_["ROUGE_L"] for _ in evaluation_score])
                gather_score["CIDEr"] = np.array([_["CIDEr"] for _ in evaluation_score])
                gather_score["CIDEr-R"] = np.array([_["CIDEr-R"] for _ in evaluation_score])
                dataset_gather_scores_dict[dataset_name] = gather_score

        dataset_cur_metrics_dict = {}
        for dataset_name, gather_score in dataset_gather_scores_dict.items():
            if dataset_name in ["AiR-D", "OSIE"]:
                cur_metric = {
                    "metrics/SM without Dur": gather_score["scanmatch_score"][:, 0].mean(),
                    "metrics/SM with Dur": gather_score["scanmatch_score"][:, 1].mean(),
                    "metrics/MM Vector": gather_score["multimatch_score"][:, 0].mean(),
                    "metrics/MM Direction": gather_score["multimatch_score"][:, 1].mean(),
                    "metrics/MM Length": gather_score["multimatch_score"][:, 2].mean(),
                    "metrics/MM Position": gather_score["multimatch_score"][:, 3].mean(),
                    "metrics/MM Duration": gather_score["multimatch_score"][:, 4].mean(),
                    "metrics/MM": gather_score["multimatch_score"].mean(),
                    "metrics/SED": gather_score["sed_score"].mean(),
                    "metrics/STDE": gather_score["stde_score"].mean(),
                    "metrics/SS without Dur": gather_score["SS_score"][:, 0].mean(),
                    "metrics/SS with Dur": gather_score["SS_score"][:, 1].mean(),
                    "metrics/CC": gather_score["CC"].mean(),
                    "metrics/AUC": gather_score["AUC"].mean(),
                    "metrics/NSS": gather_score["NSS"].mean(),
                    "metrics/sAUC": gather_score["sAUC"].mean(),
                    "metrics/KLD": gather_score["KLD"].mean(),
                    "metrics/SIM": gather_score["SIM"].mean(),
                    "metrics/Bleu_4": gather_score["Bleu_4"].mean(),
                    "metrics/METEOR": gather_score["METEOR"].mean(),
                    "metrics/ROUGE_L": gather_score["ROUGE_L"].mean(),
                    "metrics/CIDEr": gather_score["CIDEr"].mean(),
                    "metrics/CIDEr-R": gather_score["CIDEr-R"].mean(),
                }
            else:
                cur_metric = {
                    "metrics/SM without Dur": gather_score["scanmatch_score"][:, 0].mean(),
                    "metrics/SM with Dur": gather_score["scanmatch_score"][:, 1].mean(),
                    "metrics/MM Vector": gather_score["multimatch_score"][:, 0].mean(),
                    "metrics/MM Direction": gather_score["multimatch_score"][:, 1].mean(),
                    "metrics/MM Length": gather_score["multimatch_score"][:, 2].mean(),
                    "metrics/MM Position": gather_score["multimatch_score"][:, 3].mean(),
                    "metrics/MM Duration": gather_score["multimatch_score"][:, 4].mean(),
                    "metrics/MM": gather_score["multimatch_score"].mean(),
                    "metrics/SED": gather_score["sed_score"].mean(),
                    "metrics/STDE": gather_score["stde_score"].mean(),
                    "metrics/SS without Dur": gather_score["SS_score"][:, 0].mean(),
                    "metrics/SS with Dur": gather_score["SS_score"][:, 1].mean(),
                    "metrics/SSS without Dur": gather_score["SSS_score"][:, 0].mean(),
                    "metrics/SSS with Dur": gather_score["SSS_score"][:, 1].mean(),
                    "metrics/CC": gather_score["CC"].mean(),
                    "metrics/AUC": gather_score["AUC"].mean(),
                    "metrics/NSS": gather_score["NSS"].mean(),
                    "metrics/sAUC": gather_score["sAUC"].mean(),
                    "metrics/KLD": gather_score["KLD"].mean(),
                    "metrics/SIM": gather_score["SIM"].mean(),
                    "metrics/Bleu_4": gather_score["Bleu_4"].mean(),
                    "metrics/METEOR": gather_score["METEOR"].mean(),
                    "metrics/ROUGE_L": gather_score["ROUGE_L"].mean(),
                    "metrics/CIDEr": gather_score["CIDEr"].mean(),
                    "metrics/CIDEr-R": gather_score["CIDEr-R"].mean(),
                }
            dataset_cur_metrics_dict[dataset_name] = cur_metric

        # for all the dataset
        saliency_eval = np.array(list(all_saliency_evals.values()))
        gather_scores = {
            "scanmatch_score": np.array([_["scanmatch_score"] for _ in evaluation_scores]),
            "multimatch_score": np.array([_["multimatch_score"] for _ in evaluation_scores]),
            "sed_score": np.array([_["sed_score"] for _ in evaluation_scores]),
            "stde_score": np.array([_["stde_score"] for _ in evaluation_scores]),
            "SS_score": np.array([_["SS_score"] for _ in evaluation_scores]),
            "SSS_score": np.array([_["SSS_score"] for _ in evaluation_scores if "SSS_score" in _ and not np.isnan(sum(_["SSS_score"]))]),
            "CC": saliency_eval[:, 0],
            "AUC": saliency_eval[:, 1],
            "NSS": saliency_eval[:, 2],
            "sAUC": saliency_eval[:, 3],
            "KLD": saliency_eval[:, 4],
            "SIM": saliency_eval[:, 5],
            "Bleu_4": np.array([_["Bleu_4"] for _ in list(evaluator.scanpath_eval.imgToEval.values())]),
            "METEOR": np.array([_["METEOR"] for _ in list(evaluator.scanpath_eval.imgToEval.values())]),
            "ROUGE_L": np.array([_["ROUGE_L"] for _ in list(evaluator.scanpath_eval.imgToEval.values())]),
            "CIDEr": np.array([_["CIDEr"] for _ in list(evaluator.scanpath_eval.imgToEval.values())]),
            "CIDEr-R": np.array([_["CIDEr-R"] for _ in list(evaluator.scanpath_eval.imgToEval.values())]),
        }

        cur_metrics = {
            "metrics/SM without Dur": gather_scores["scanmatch_score"][:, 0].mean(),
            "metrics/SM with Dur": gather_scores["scanmatch_score"][:, 1].mean(),
            "metrics/MM Vector": gather_scores["multimatch_score"][:, 0].mean(),
            "metrics/MM Direction": gather_scores["multimatch_score"][:, 1].mean(),
            "metrics/MM Length": gather_scores["multimatch_score"][:, 2].mean(),
            "metrics/MM Position": gather_scores["multimatch_score"][:, 3].mean(),
            "metrics/MM Duration": gather_scores["multimatch_score"][:, 4].mean(),
            "metrics/MM": gather_scores["multimatch_score"].mean(),
            "metrics/SED": gather_scores["sed_score"].mean(),
            "metrics/STDE": gather_scores["stde_score"].mean(),
            "metrics/SS without Dur": gather_scores["SS_score"][:, 0].mean(),
            "metrics/SS with Dur": gather_scores["SS_score"][:, 1].mean(),
            "metrics/SSS without Dur": gather_scores["SSS_score"][:, 0].mean() if gather_scores["SSS_score"].shape[0] != 0 else np.nan,
            "metrics/SSS with Dur": gather_scores["SSS_score"][:, 1].mean() if gather_scores["SSS_score"].shape[0] != 0 else np.nan,
            "metrics/CC": gather_scores["CC"].mean(),
            "metrics/AUC": gather_scores["AUC"].mean(),
            "metrics/NSS": gather_scores["NSS"].mean(),
            "metrics/sAUC": gather_scores["sAUC"].mean(),
            "metrics/KLD": gather_scores["KLD"].mean(),
            "metrics/SIM": gather_scores["SIM"].mean(),
            "metrics/Bleu_4": gather_scores["Bleu_4"].mean(),
            "metrics/METEOR": gather_scores["METEOR"].mean(),
            "metrics/ROUGE_L": gather_scores["ROUGE_L"].mean(),
            "metrics/CIDEr": gather_scores["CIDEr"].mean(),
            "metrics/CIDEr-R": gather_scores["CIDEr-R"].mean(),
        }

        dataset_cur_metrics_dict["all"] = cur_metrics

        return dataset_cur_metrics_dict

    else:
        return None

def eval(accelerator, opt, split='dev'):
    """
    Evaluate a trained model on either dev or test.
    """
    eval_dataset = UnifiedScanpath(split=split, opt=opt)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_dataset.collate_func,
        drop_last=False
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    eval_dataloader = accelerator.prepare(eval_dataloader)

    dataset_cur_metrics_dict = get_evaluation(accelerator, eval_dataloader, opt)

    if accelerator.is_main_process:
        for dataset, cur_metrics in dataset_cur_metrics_dict.items():
            accelerator.print("-" * 40)
            accelerator.print("{:30}".format(dataset))
            for key, value in cur_metrics.items():
                accelerator.print("{:30}: {:.3f}".format(key, value))
        accelerator.print("-" * 40)







