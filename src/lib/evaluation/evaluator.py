"""Evaluation"""
from __future__ import print_function

import copy
import gzip
from typing import List

import numpy as np
from torch import Tensor

import logging
import os

from json import encoder

from accelerate.utils import tqdm

from lib.evaluation.metrics import multimatch
from lib.evaluation.evaltools.scanmatch import ScanMatch
from lib.evaluation.evaltools.visual_attention_metrics import string_edit_distance, scaled_time_delay_embedding_similarity
from lib.evaluation import saliency
from lib.evaluation.pycocoevalcap.eval_scanpath import ScanpathEval

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

logger = logging.getLogger(__name__)

def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0

def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))


class Evaluator(object):
    def __init__(self, opt):
        self.opt = opt
        self.datasets = ["AiR-D", "OSIE", "COCO-TP", "COCO-TA"]
        self.AiR_fix_clusters = np.load(os.path.join(opt.dataset_dir, "AiR", "processed_data", 'clusters.npy'),
                                        allow_pickle=True).item()
        self.OSIE_fix_clusters = np.load(os.path.join(opt.dataset_dir, "OSIE", "processed", 'clusters.npy'),
                                        allow_pickle=True).item()
        self.COCOTP_fix_clusters = np.load(os.path.join(opt.dataset_dir, "COCO/TP", "processed", 'clusters.npy'),
                                           allow_pickle=True).item()
        self.COCOTA_fix_clusters = np.load(os.path.join(opt.dataset_dir, "COCO/TA", "processed", 'clusters.npy'),
                                           allow_pickle=True).item()


    def explanation_evaluation(self, gts, preds):
        self.scanpath_eval = ScanpathEval(gts, preds)
        self.scanpath_eval.evaluate()


    def transform(self, preds: Tensor or List[Tensor])-> List:
        if isinstance(preds, Tensor):
            predictions = []
            np_preds = preds.cpu().numpy()
            for idx in range(np_preds.shape[0]):
                X = np_preds[idx][:, 0]
                Y = np_preds[idx][:, 1]
                T = np_preds[idx][:, 2]
                fixation_length = len(X[X!=-1])
                prediction = {
                    "X": X[:fixation_length].tolist(),
                    "Y": Y[:fixation_length].tolist(),
                    "T": T[:fixation_length].tolist(),
                }
                predictions.append(prediction)
        else:
            predictions = []
            np_preds = [_.cpu().numpy() for _ in preds]
            for pred in np_preds:
                X = pred[:, 0]
                Y = pred[:, 1]
                T = pred[:, 2]
                fixation_length = len(X[X!=-1])
                prediction = {
                    "X": X[:fixation_length].tolist(),
                    "Y": Y[:fixation_length].tolist(),
                    "T": T[:fixation_length].tolist(),
                }
                predictions.append(prediction)

        return predictions

    def measure(self, gts: List, preds: List, batch: dict):
        # evaluation order is SM, MM, SED, STDE
        image_size = batch["image_size"]
        dataset_idx = batch["dataset_idx"]
        fixation_info = batch["fixation_info"]
        scores = []
        for idx, (gt, pred) in enumerate(zip(gts, preds)):
            im_h, im_w = image_size[idx, 0].item(), image_size[idx, 1].item()

            # make sure the fixation length is not less than 3
            _gt = copy.deepcopy(gt)
            _pred = copy.deepcopy(pred)
            if len(_gt["X"]) < 3:
                for _ in range(3 - len(_gt["X"])):
                    _gt["X"].append(1)
                    _gt["Y"].append(1)
                    _gt["T"].append(1)
            if len(_pred["X"]) < 3:
                for _ in range(3 - len(_pred["X"])):
                    _pred["X"].append(1)
                    _pred["Y"].append(1)
                    _pred["T"].append(1)

            # get scanmatch
            scanmatch_score = self.ScanMatch(_pred, _gt, im_w, im_h)

            # get multimatch
            # [VecSim, DirSim, LenSim, PosSim, DurSim]
            multimatch_score = multimatch(_pred, _gt, (im_w, im_h))

            # get SED and STDE
            gt_vector = np.array([_gt["X"], _gt["Y"], _gt["T"]]).T
            pred_vector = np.array([_pred["X"], _pred["Y"], _pred["T"]]).T
            stimulus = np.zeros((im_h, im_w, 3), dtype=np.float32)
            sed_score = string_edit_distance(stimulus, gt_vector, pred_vector)
            stde_score = scaled_time_delay_embedding_similarity(gt_vector, pred_vector, stimulus)

            # get SS score
            dataset = self.datasets[dataset_idx[idx]]
            if dataset == "AiR-D":
                clusters = self.AiR_fix_clusters
            elif dataset == "OSIE":
                clusters = self.OSIE_fix_clusters
            elif dataset == "COCO-TP":
                clusters = self.COCOTP_fix_clusters
            elif dataset == "COCO-TA":
                clusters = self.COCOTA_fix_clusters
            else:
                raise "Invalid dataset"
            SS = self.compute_SS(_pred, clusters, truncate=self.opt.max_length,
                                 fixation_info=fixation_info[idx], dataset=dataset)
            SS_Time = self.compute_SS_Time(_pred, clusters, truncate=self.opt.max_length,
                                           fixation_info=fixation_info[idx], dataset=dataset)
            SS_score = [SS, SS_Time]

            # get SemSS score
            if dataset in ["COCO-TP", "COCO-TA"]:
                segmentation_map_dir = os.path.join(self.opt.dataset_dir, "COCO/TP", "semantic_seq_full/segmentation_maps")
                SSS = self.compute_SSS(_pred, _gt, fixation_info=fixation_info[idx], truncate=self.opt.max_length,
                                       segmentation_map_dir=segmentation_map_dir)
                SSS_Time = self.compute_SSS_Time(_pred, _gt, fixation_info=fixation_info[idx], truncate=self.opt.max_length,
                                                 segmentation_map_dir=segmentation_map_dir)
                SSS_score = [SSS, SSS_Time]

                scores.append({
                    "scanmatch_score": scanmatch_score,
                    "multimatch_score": multimatch_score,
                    "sed_score": sed_score,
                    "stde_score": stde_score,
                    "SS_score": SS_score,
                    "SSS_score": SSS_score,
                })
            else:
                scores.append({
                    "scanmatch_score": scanmatch_score,
                    "multimatch_score": multimatch_score,
                    "sed_score": sed_score,
                    "stde_score": stde_score,
                    "SS_score": SS_score
                })
        return scores

    def measure_gt(self, gts: List, preds: List, batch: dict):
        # evaluation order is SM, MM, SED, STDE
        image_size = batch["image_size"]
        dataset_idx = batch["dataset_idx"]
        fixation_info = batch["fixation_info"]
        scores = []
        pbar = tqdm(total=len(gts))
        for idx, (gt, pred) in enumerate(zip(gts, preds)):
            im_h, im_w = image_size[idx, 0].item(), image_size[idx, 1].item()

            # make sure the fixation length is not less than 3
            _gt = copy.deepcopy(gt)
            _pred = copy.deepcopy(pred)
            if len(_gt["X"]) < 3:
                for _ in range(3 - len(_gt["X"])):
                    _gt["X"].append(1)
                    _gt["Y"].append(1)
                    _gt["T"].append(1)
            if len(_pred["X"]) < 3:
                for _ in range(3 - len(_pred["X"])):
                    _pred["X"].append(1)
                    _pred["Y"].append(1)
                    _pred["T"].append(1)

            _gt["Y"] = _gt["Y"][:len(_gt["X"])]
            _gt["T"] = _gt["T"][:len(_gt["X"])]

            _pred["Y"] = _pred["Y"][:len(_pred["X"])]
            _pred["T"] = _pred["T"][:len(_pred["X"])]

            # get scanmatch
            scanmatch_score = self.ScanMatch(_pred, _gt, im_w, im_h)

            # get multimatch
            # [VecSim, DirSim, LenSim, PosSim, DurSim]
            multimatch_score = multimatch(_pred, _gt, (im_w, im_h))

            # get SED and STDE
            gt_vector = np.array([_gt["X"], _gt["Y"], _gt["T"]]).T
            pred_vector = np.array([_pred["X"], _pred["Y"], _pred["T"]]).T
            stimulus = np.zeros((im_h, im_w, 3), dtype=np.float32)
            sed_score = string_edit_distance(stimulus, gt_vector, pred_vector)
            stde_score = scaled_time_delay_embedding_similarity(gt_vector, pred_vector, stimulus)

            # get SS score
            dataset = self.datasets[dataset_idx[idx]]
            if dataset == "AiR-D":
                clusters = self.AiR_fix_clusters
            elif dataset == "OSIE":
                clusters = self.OSIE_fix_clusters
            elif dataset == "COCO-TP":
                clusters = self.COCOTP_fix_clusters
            elif dataset == "COCO-TA":
                clusters = self.COCOTA_fix_clusters
            else:
                raise "Invalid dataset"
            SS = self.compute_SS(_pred, clusters, truncate=self.opt.max_length,
                                 fixation_info=fixation_info[idx], dataset=dataset)
            SS_Time = self.compute_SS_Time(_pred, clusters, truncate=self.opt.max_length,
                                           fixation_info=fixation_info[idx], dataset=dataset)
            SS_score = [SS, SS_Time]

            # get SemSS score
            if dataset in ["COCO-TP", "COCO-TA"]:
                segmentation_map_dir = os.path.join(self.opt.dataset_dir, "COCO/TP", "semantic_seq_full/segmentation_maps")
                SSS = self.compute_SSS(_pred, _gt, fixation_info=fixation_info[idx], truncate=self.opt.max_length,
                                       segmentation_map_dir=segmentation_map_dir)
                SSS_Time = self.compute_SSS_Time(_pred, _gt, fixation_info=fixation_info[idx], truncate=self.opt.max_length,
                                                 segmentation_map_dir=segmentation_map_dir)
                SSS_score = [SSS, SSS_Time]

                scores.append({
                    "scanmatch_score": scanmatch_score,
                    "multimatch_score": multimatch_score,
                    "sed_score": sed_score,
                    "stde_score": stde_score,
                    "SS_score": SS_score,
                    "SSS_score": SSS_score,
                })
            else:
                scores.append({
                    "scanmatch_score": scanmatch_score,
                    "multimatch_score": multimatch_score,
                    "sed_score": sed_score,
                    "stde_score": stde_score,
                    "SS_score": SS_score
                })
            pbar.update()
        return scores

    # evaluation saliency
    def eval_saliency(self, prediction_fixation_dict: dict, gt_fixation_dict: dict):
        all_evals = {}
        # generate the shufmap
        AiRshufmap = np.zeros((self.opt.height, self.opt.width))
        OSIEshufmap = np.zeros((self.opt.height, self.opt.width))
        COCOTPshufmap = np.zeros((self.opt.height, self.opt.width))
        COCOTAshufmap = np.zeros((self.opt.height, self.opt.width))
        # construct the shufmap
        for key, values in gt_fixation_dict.items():
            if "AiR-D" in key:
                shufmap = AiRshufmap
            elif "OSIE" in key:
                shufmap = OSIEshufmap
            elif "COCO-TP" in key:
                shufmap = COCOTPshufmap
            elif "COCO-TA" in key:
                shufmap = COCOTAshufmap
            else:
                raise "Invalid Dataset"

            for value in values:
                for idx in range(len(value["X"])):
                    shufmap[int(value["Y"][idx] - 1), int(value["X"][idx] - 1)] = 1

        # generate the map
        for key in prediction_fixation_dict.keys():
            predict_fixations = prediction_fixation_dict[key]
            gt_fixations = gt_fixation_dict[key]

            gt_saliency = np.zeros((self.opt.height, self.opt.width))
            predict_saliency = np.zeros((self.opt.height, self.opt.width))
            gt_fixation_for_eval = {
                "rows": [],
                "cols": []
            }

            for gt_fixation in gt_fixations:
                for idx in range(len(gt_fixation["X"])):
                    y_value = min(int(gt_fixation["Y"][idx] - 1), self.opt.height - 1)
                    x_value = min(int(gt_fixation["X"][idx] - 1), self.opt.width - 1)
                    gt_saliency[y_value, x_value] += 1
                    gt_fixation_for_eval["rows"].append(y_value)
                    gt_fixation_for_eval["cols"].append(x_value)

            for predict_fixation in predict_fixations:
                for idx in range(len(predict_fixation["X"])):
                    y_value = min(int(predict_fixation["Y"][idx]), self.opt.height - 1)
                    x_value = min(int(predict_fixation["X"][idx]), self.opt.width - 1)
                    predict_saliency[y_value, x_value] += 1


            gt_saliency = saliency.filter_heatmap(gt_saliency)
            predict_saliency = saliency.filter_heatmap(predict_saliency)

            if "AiR-D" in key:
                shufmap = AiRshufmap
            elif "OSIE" in key:
                shufmap = OSIEshufmap
            elif "COCO-TP" in key:
                shufmap = COCOTPshufmap
            elif "COCO-TA" in key:
                shufmap = COCOTAshufmap
            else:
                raise "Invalid Dataset"

            saliency_scores = []
            saliency_scores.append(saliency.cal_cc_score(predict_saliency, gt_saliency))
            saliency_scores.append(saliency.cal_auc_score(predict_saliency, gt_fixation_for_eval))
            saliency_scores.append(saliency.cal_nss_score(predict_saliency, gt_fixation_for_eval))
            saliency_scores.append(saliency.cal_sauc_score(predict_saliency, gt_fixation_for_eval, shufmap))
            saliency_scores.append(saliency.cal_kld_score(predict_saliency, gt_saliency))
            saliency_scores.append(saliency.cal_sim_score(predict_saliency, gt_saliency))


            all_evals[key] = saliency_scores

        return all_evals

    def measure_scanmatch(self, gts: List, preds: List, image_size: Tensor):
        # evaluation order is SM, MM, SED, STDE
        scores = []
        for idx, (gt, pred) in enumerate(zip(gts, preds)):
            im_h, im_w = image_size[idx, 0].item(), image_size[idx, 1].item()

            # make sure the fixation length is not less than 3
            _gt = copy.deepcopy(gt)
            _pred = copy.deepcopy(pred)
            if len(_gt["X"]) < 3:
                for idx in range(3 - len(_gt["X"])):
                    _gt["X"].append(1)
                    _gt["Y"].append(1)
                    _gt["T"].append(1)
            if len(_pred["X"]) < 3:
                for idx in range(3 - len(_pred["X"])):
                    _pred["X"].append(1)
                    _pred["Y"].append(1)
                    _pred["T"].append(1)

            # get scanmatch
            scanmatch_score = self.ScanMatch(_pred, _gt, im_w, im_h)
            scores.append(scanmatch_score)

        return scores


    def ScanMatch(self, pred, gt, im_w, im_h):
        # create a ScanMatch object
        ScanMatchwithoutDuration = ScanMatch(Xres=im_w, Yres=im_h, Xbin=16, Ybin=12, Offset=(0, 0),
                                             Threshold=3.5)

        gt_vector = np.array([gt["X"], gt["Y"], gt["T"]]).T
        pred_vector = np.array([pred["X"], pred["Y"], pred["T"]]).T
        sequence_gt = ScanMatchwithoutDuration.fixationToSequence(gt_vector).astype(np.int32)
        sequence_pred = ScanMatchwithoutDuration.fixationToSequence(pred_vector).astype(np.int32)
        (score_1, align_1, f_1) = ScanMatchwithoutDuration.match(sequence_gt, sequence_pred)

        # create a ScanMatch object
        ScanMatchwithDuration = ScanMatch(Xres=im_w, Yres=im_h, Xbin=16, Ybin=12, Offset=(0, 0),
                                          TempBin=50, Threshold=3.5)

        sequence_gt = ScanMatchwithDuration.fixationToSequence(gt_vector).astype(np.int32)
        sequence_pred = ScanMatchwithDuration.fixationToSequence(pred_vector).astype(np.int32)
        (score_2, align_2, f_2) = ScanMatchwithDuration.match(sequence_gt, sequence_pred)

        return score_1, score_2

    def scanpath2clusters(self, meanshift, scanpath):
        string = []
        xs = scanpath['X']
        ys = scanpath['Y']
        for i in range(len(xs)):
            symbol = meanshift.predict([[xs[i], ys[i]]])[0]
            string.append(symbol)
        return string

    def compute_SS(self, pred, clusters, truncate, fixation_info, dataset):
        if dataset == "AiR-D":
            key = '{}-{}-{}'.format(fixation_info['split'], fixation_info['question_id'], fixation_info['image_id'][:-4])
        elif dataset == "OSIE":
            key = '{}-{}'.format(fixation_info['split'], fixation_info['name'][:-4])
        elif dataset == "COCO-TP":
            key = '{}-{}-{}-{}'.format(fixation_info['split'], fixation_info['condition'], fixation_info['task'],
                                       fixation_info['name'][:-4])
        elif dataset == "COCO-TA":
            key = '{}-{}-{}-{}'.format(fixation_info['split'], fixation_info['condition'], fixation_info['task'],
                                       fixation_info['name'][:-4])
        else:
            raise "Invalid dataset"

        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        subject = fixation_info['subject']
        pred_string = self.scanpath2clusters(cluster, pred)

        gt = strings[subject]
        if len(gt) > 0:
            pred_string = pred_string[:truncate] if len(pred_string) > truncate else pred_string
            gt = gt[:truncate] if len(gt) > truncate else gt
            score = nw_matching(pred_string, gt)
        else:
            score = 0

        return score


    def compute_SS_Time(self, pred, clusters, truncate, fixation_info, dataset, tempbin = 50):
        if dataset == "AiR-D":
            key = '{}-{}-{}'.format(fixation_info['split'], fixation_info['question_id'], fixation_info['image_id'][:-4])
        elif dataset == "OSIE":
            key = '{}-{}'.format(fixation_info['split'], fixation_info['name'][:-4])
        elif dataset == "COCO-TP":
            key = '{}-{}-{}-{}'.format(fixation_info['split'], fixation_info['condition'], fixation_info['task'],
                                       fixation_info['name'][:-4])
        elif dataset == "COCO-TA":
            key = '{}-{}-{}-{}'.format(fixation_info['split'], fixation_info['condition'], fixation_info['task'],
                                       fixation_info['name'][:-4])
        else:
            raise "Invalid dataset"

        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        subject = fixation_info['subject']
        pred_string = self.scanpath2clusters(cluster, pred)

        gt = strings[subject]
        if len(gt) > 0:
            time_string = fixation_info["T"]
            gt = gt[:truncate] if len(gt) > truncate else gt
            pred_string = pred_string[:truncate] if len(pred_string) > truncate else pred_string
            gtime_string = time_string[:truncate] if len(time_string) > truncate else time_string
            ptime_string = pred['T'][:truncate] if len(pred['T']) > truncate else pred['T']

            pred_time = []
            gt_time = []
            for p, t_p in zip(pred_string, ptime_string):
                pred_time.extend([p for _ in range(int(t_p / tempbin))])
            for g, t_g in zip(gt, gtime_string):
                gt_time.extend([g for _ in range(int(t_g / tempbin))])

            score = nw_matching(pred_time, gt_time)
        else:
            score = 0

        return score

    def scanpath2categories(self, seg_map, scanpath):
        string = []
        xs = scanpath['X']
        ys = scanpath['Y']
        ts = scanpath['T']
        for x, y, t in zip(xs, ys, ts):
            symbol = str(int(seg_map[int(y), int(x)]))
            string.append((symbol, t))
        return string

    def compute_SSS(self, pred, gt, fixation_info, truncate, segmentation_map_dir):
        if not os.path.exists(os.path.join(segmentation_map_dir, fixation_info['name'][:-3] + 'npy.gz')):
            return np.nan
        with gzip.GzipFile(os.path.join(segmentation_map_dir, fixation_info['name'][:-3] + 'npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()

        gt_fixations = copy.deepcopy(gt)
        pred_fixations = copy.deepcopy(pred)
        # normalize to fit the segmentation map
        gt_fixations["X"] = (np.array(gt_fixations["X"]) / self.opt.width * 512).tolist()
        gt_fixations["Y"] = (np.array(gt_fixations["Y"]) / self.opt.height * 320).tolist()
        pred_fixations["X"] = (np.array(pred_fixations["X"]) / self.opt.width * 512).tolist()
        pred_fixations["Y"] = (np.array(pred_fixations["Y"]) / self.opt.height * 320).tolist()

        gt_strings = self.scanpath2categories(segmentation_map, gt_fixations)
        pred_strings = self.scanpath2categories(segmentation_map, pred_fixations)

        pred_strings = pred_strings[:truncate] if len(pred_strings) > truncate else pred_strings
        pred_noT = [i[0] for i in pred_strings]

        gt_strings = gt_strings[:truncate] if len(gt_strings) > truncate else gt_strings
        gt_noT = [i[0] for i in gt_strings]

        if len(gt_strings) > 0:
            score = nw_matching(pred_noT, gt_noT)
        else:
            score = np.nan

        return score

    def compute_SSS_Time(self, pred, gt, fixation_info, truncate, segmentation_map_dir, tempbin=50):
        if not os.path.exists(os.path.join(segmentation_map_dir, fixation_info['name'][:-3] + 'npy.gz')):
            return np.nan
        with gzip.GzipFile(os.path.join(segmentation_map_dir, fixation_info['name'][:-3] + 'npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()

        gt_fixations = copy.deepcopy(gt)
        pred_fixations = copy.deepcopy(pred)
        # normalize to fit the segmentation map
        gt_fixations["X"] = (np.array(gt_fixations["X"]) / self.opt.width * 512).tolist()
        gt_fixations["Y"] = (np.array(gt_fixations["Y"]) / self.opt.height * 320).tolist()
        pred_fixations["X"] = (np.array(pred_fixations["X"]) / self.opt.width * 512).tolist()
        pred_fixations["Y"] = (np.array(pred_fixations["Y"]) / self.opt.height * 320).tolist()

        gt_strings = self.scanpath2categories(segmentation_map, gt_fixations)
        pred_strings = self.scanpath2categories(segmentation_map, pred_fixations)

        pred_strings = pred_strings[:truncate] if len(pred_strings) > truncate else pred_strings
        pred_T = []
        for p in pred_strings:
            pred_T.extend([p[0] for _ in range(int(p[1] / tempbin))])

        gt_strings = gt_strings[:truncate] if len(gt_strings) > truncate else gt_strings
        gt_T = []
        for g in gt_strings:
            gt_T.extend([g[0] for _ in range(int(g[1] / tempbin))])

        if len(gt_strings) > 0:
            score = nw_matching(pred_T, gt_T)
        else:
            score = np.nan

        return score