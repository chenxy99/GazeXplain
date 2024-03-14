import argparse
import os.path

import h5py
import torch
import numpy as np
from accelerate import accelerator
from torch.utils.data import Dataset, DataLoader
from os.path import join
import json
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as filters
import pickle
import cv2
from accelerate.utils import tqdm
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, BertTokenizer, RobertaTokenizerFast, \
    BertTokenizerFast

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

epsilon = 1e-7

class UnifiedScanpath(Dataset):
    """
    get unified scanpath data
    """

    def __init__(self, split, opt, datasets=None):
        self.opt = opt
        self.split = split
        self.action_map = (opt.im_h, opt.im_w)

        self.dataset2idx = {
            "AiR-D": 0,
            "OSIE": 1,
            "COCO-TP": 2,
            "COCO-TA": 3,
        }

        explanation_gts = {}

        ########## for air dataset ##########
        air_fixation_dir = os.path.join(self.opt.dataset_dir, "AiR", "processed_data")
        with open(join(air_fixation_dir, "AiR_fixations_{}.json".format(self.split)), "r") as f:
            air_fixations = json.load(f)

        if self.opt.tiny:
            air_fixations = air_fixations[:100]

        for iter, fixation in enumerate(air_fixations):
            fixation["dataset"] = "AiR-D"
            # idx of the current dataset
            fixation["idx"] = iter

        # add duration
        for fixation in air_fixations:
            fixation["T"] = [t_end - t_start for t_start, t_end in zip(fixation["T_start"], fixation["T_end"])]

        # explanation
        air_explanation_dir = os.path.join(self.opt.dataset_dir, "AiR", "processed_data")
        with open(join(air_explanation_dir, "explanation_manually.json"), "r") as f:
            air_explanations = json.load(f)

        air_explanation_dict = {}
        for explanation in air_explanations:
            brief_explanation = [_ for _ in explanation["explanation"]]
            # keep the first sentence
            brief_explanation = [_.split(". ")[0] for _ in brief_explanation]
            brief_explanation = [_ if _[-1] == "." else _ + "." for _ in brief_explanation]
            words_each_explanation = [len(_.split(" ")) for _ in brief_explanation]
            for iter, length_explanation in enumerate(words_each_explanation):
                if length_explanation >= self.opt.max_explanation_length:
                    sub_explanation = brief_explanation[iter].split(", ")
                    words_each_sub_explanation = [len(_.split(" ")) for _ in sub_explanation]
                    for select_idx in range(len(words_each_sub_explanation)):
                        if sum(words_each_sub_explanation[:select_idx + 1]) >= self.opt.min_explanation_length:
                            re_explanation = ", ".join(sub_explanation[:select_idx + 1])
                            if re_explanation[-1] != ".":
                                re_explanation = re_explanation + "."
                            brief_explanation[iter] = re_explanation
                            break
            air_explanation_dict.setdefault(explanation["question_id"], {})[explanation["subject"]] = brief_explanation
            explanation_gts.setdefault(
                "{}-{}".format(explanation["question_id"], explanation["subject"]), []).append({
                "caption": " ".join(brief_explanation),
                "question_id": explanation["question_id"],
                "subject": explanation["subject"]
            })

        for fixation in air_fixations:
            if fixation["question_id"] in air_explanation_dict:
                if fixation["subject"] in air_explanation_dict[fixation["question_id"]]:
                    fixation["explanation"] = air_explanation_dict[fixation["question_id"]][fixation["subject"]]


        # task description
        text_template = "Question: {} Answer: {}."
        for fixation in air_fixations:
            task_description = text_template.format(fixation["question"], fixation["subject_answer"])
            fixation["task_description"] = task_description

        ########## for OSIE dataset ##########
        osie_fixation_dir = os.path.join(self.opt.dataset_dir, "OSIE", "processed")
        with open(join(osie_fixation_dir, "fixations.json"), "r") as f:
            osie_fixations = json.load(f)

        osie_fixations = [_ for _ in osie_fixations if _["split"] == self.split]

        if self.opt.tiny:
            osie_fixations = osie_fixations[:100]


        for iter, fixation in enumerate(osie_fixations):
            fixation["dataset"] = "OSIE"
            # idx of the current dataset
            fixation["idx"] = iter
            fixation["height"] = 600
            fixation["width"] = 800

        # explanation
        osie_explanation_dir = os.path.join(self.opt.dataset_dir, "OSIE", "processed")
        with open(join(osie_explanation_dir, "explanation_manually.json"), "r") as f:
            osie_explanations = json.load(f)

        osie_explanation_dict = {}
        for explanation in osie_explanations:
            brief_explanation = [_ for _ in explanation["explanation"]]
            # keep the first sentence
            brief_explanation = [_.split(". ")[0] for _ in brief_explanation]
            brief_explanation = [_ if _[-1] == "." else _ + "." for _ in brief_explanation]
            words_each_explanation = [len(_.split(" ")) for _ in brief_explanation]
            for iter, length_explanation in enumerate(words_each_explanation):
                if length_explanation >= self.opt.max_explanation_length:
                    sub_explanation = brief_explanation[iter].split(", ")
                    words_each_sub_explanation = [len(_.split(" ")) for _ in sub_explanation]
                    for select_idx in range(len(words_each_sub_explanation)):
                        if sum(words_each_sub_explanation[:select_idx + 1]) >= self.opt.min_explanation_length:
                            re_explanation = ", ".join(sub_explanation[:select_idx + 1])
                            if re_explanation[-1] != ".":
                                re_explanation = re_explanation + "."
                            brief_explanation[iter] = re_explanation
                            break
            osie_explanation_dict.setdefault(explanation["name"], {})[explanation["subject"]] = brief_explanation
            explanation_gts.setdefault(
                "{}-{}".format(explanation["name"][:-4], explanation["subject"]), []).append({
                "caption": " ".join(brief_explanation),
                "name": explanation["name"],
                "subject": explanation["subject"]
            })

        for fixation in osie_fixations:
            if fixation["name"] in osie_explanation_dict:
                if fixation["subject"] in osie_explanation_dict[fixation["name"]]:
                    fixation["explanation"] = osie_explanation_dict[fixation["name"]][fixation["subject"]]


        # task description
        text_template = "Question: What do you see in the image?"
        for fixation in osie_fixations:
            task_description = text_template
            fixation["task_description"] = task_description

        ########## for COCO-Search18 TP ##########
        cocosearch18_TP_fixation_dir = os.path.join(self.opt.dataset_dir, "COCO", "TP", "fixations")
        with open(join(cocosearch18_TP_fixation_dir, "coco_search18_fixations_TP_{}.json".format(self.split)), "r") as f:
            cocosearch18_TP_fixations = json.load(f)

        if self.opt.tiny:
            cocosearch18_TP_fixations = cocosearch18_TP_fixations[:100]

        for iter, fixation in enumerate(cocosearch18_TP_fixations):
            fixation["dataset"] = "COCO-TP"
            # idx of the current dataset
            fixation["idx"] = iter
            fixation["height"] = 320
            fixation["width"] = 512

        # explanation
        cocotp_explanation_dir = os.path.join(self.opt.dataset_dir, "COCO/TP", "processed")
        with open(join(cocotp_explanation_dir, "explanation_manually.json"), "r") as f:
            cocotp_explanations = json.load(f)

        cocotp_explanation_dict = {}
        for explanation in cocotp_explanations:
            brief_explanation = [_ for _ in explanation["explanation"]]
            # keep the first sentence
            brief_explanation = [_.split(". ")[0] for _ in brief_explanation]
            brief_explanation = [_ if _[-1] == "." else _ + "." for _ in brief_explanation]
            words_each_explanation = [len(_.split(" ")) for _ in brief_explanation]
            for iter, length_explanation in enumerate(words_each_explanation):
                if length_explanation >= self.opt.max_explanation_length:
                    sub_explanation = brief_explanation[iter].split(", ")
                    words_each_sub_explanation = [len(_.split(" ")) for _ in sub_explanation]
                    for select_idx in range(len(words_each_sub_explanation)):
                        if sum(words_each_sub_explanation[:select_idx + 1]) >= self.opt.min_explanation_length:
                            re_explanation = ", ".join(sub_explanation[:select_idx + 1])
                            if re_explanation[-1] != ".":
                                re_explanation = re_explanation + "."
                            brief_explanation[iter] = re_explanation
                            break
            cocotp_explanation_dict.setdefault("{}-{}".format(explanation["task"], explanation["name"]), {})[explanation["subject"]] = brief_explanation
            explanation_gts.setdefault(
                "TP-{}-{}-{}".format(explanation["task"], explanation["name"][:-4], explanation["subject"]), []).append({
                "caption": " ".join(brief_explanation),
                "task": explanation["task"],
                "name": explanation["name"],
                "subject": explanation["subject"]
            })

        for fixation in cocosearch18_TP_fixations:
            fixation["explanation"] = cocotp_explanation_dict["{}-{}".format(fixation["task"], fixation["name"])][fixation["subject"]]

        # task description
        text_template = "Question: Is there a {} in the image? Answer: {}."
        for fixation in cocosearch18_TP_fixations:
            if fixation["fixOnTarget"]:
                subject_answer = "yes"
            else:
                subject_answer = "no"
            task_description = text_template.format(fixation["task"], subject_answer)
            fixation["task_description"] = task_description


        ########## for COCO-Search18 TA ##########
        cocosearch18_TA_fixation_dir = os.path.join(self.opt.dataset_dir, "COCO", "TA", "fixations")
        if self.split in ["train", "validation"]:
            with open(join(cocosearch18_TA_fixation_dir, "coco_search18_fixations_TA_trainval.json"), "r") as f:
                cocosearch18_TA_fixations = json.load(f)

            if self.split == "train":
                cocosearch18_TA_fixations = [_ for _ in cocosearch18_TA_fixations if _["split"] == "train"]
            else:
                cocosearch18_TA_fixations = [_ for _ in cocosearch18_TA_fixations if _["split"] == "valid"]

        else:
            with open(join(cocosearch18_TA_fixation_dir, "coco_search18_fixations_TA_test.json"), "r") as f:
                cocosearch18_TA_fixations = json.load(f)

        if self.opt.tiny:
            cocosearch18_TA_fixations = cocosearch18_TA_fixations[:100]

        for iter, fixation in enumerate(cocosearch18_TA_fixations):
            fixation["dataset"] = "COCO-TA"
            # idx of the current dataset
            fixation["idx"] = iter
            fixation["height"] = 1050
            fixation["width"] = 1680


        # explanation
        cocota_explanation_dir = os.path.join(self.opt.dataset_dir, "COCO/TA", "processed")
        with open(join(cocota_explanation_dir, "explanation_manually.json"), "r") as f:
            cocota_explanations = json.load(f)

        cocota_explanation_dict = {}
        for explanation in cocota_explanations:
            brief_explanation = [_ for _ in explanation["explanation"]]
            # keep the first sentence
            brief_explanation = [_.split(". ")[0] for _ in brief_explanation]
            brief_explanation = [_ if _[-1] == "." else _ + "." for _ in brief_explanation]
            words_each_explanation = [len(_.split(" ")) for _ in brief_explanation]
            for iter, length_explanation in enumerate(words_each_explanation):
                if length_explanation >= self.opt.max_explanation_length:
                    sub_explanation = brief_explanation[iter].split(", ")
                    words_each_sub_explanation = [len(_.split(" ")) for _ in sub_explanation]
                    for select_idx in range(len(words_each_sub_explanation)):
                        if sum(words_each_sub_explanation[:select_idx + 1]) >= self.opt.min_explanation_length:
                            re_explanation = ", ".join(sub_explanation[:select_idx + 1])
                            if re_explanation[-1] != ".":
                                re_explanation = re_explanation + "."
                            brief_explanation[iter] = re_explanation
                            break
            cocota_explanation_dict.setdefault("{}-{}".format(explanation["task"], explanation["name"]), {})[explanation["subject"]] = brief_explanation
            explanation_gts.setdefault(
                "TA-{}-{}-{}".format(explanation["task"], explanation["name"][:-4], explanation["subject"]), []).append({
                "caption": " ".join(brief_explanation),
                "task": explanation["task"],
                "name": explanation["name"],
                "subject": explanation["subject"]
            })

        for fixation in cocosearch18_TA_fixations:
            fixation["explanation"] = cocota_explanation_dict["{}-{}".format(fixation["task"], fixation["name"])][fixation["subject"]]

        # task description
        text_template = "Question: Is there a {} in the image? Answer: {}."
        for fixation in cocosearch18_TA_fixations:
            if fixation["fixOnTarget"]:
                subject_answer = "yes"
            else:
                subject_answer = "no"
            task_description = text_template.format(fixation["task"], subject_answer)
            fixation["task_description"] = task_description


        fixations = []
        if datasets is None:
            datasets = self.opt.datasets
        for dataset in datasets:
            if dataset == "AiR-D":
                fixations += air_fixations
            elif dataset == "OSIE":
                fixations += osie_fixations
            elif dataset == "COCO-TP":
                fixations += cocosearch18_TP_fixations
            elif dataset == "COCO-TA":
                fixations += cocosearch18_TA_fixations
            else:
                raise "Invalid Dataset"

        gt_fixation_length = []
        for fixation in fixations:
            gt_fixation_length.append(len(fixation["X"]))
        self.max_gt_label_length = max(gt_fixation_length)

        self.fixations = fixations
        self.explanation_gts = explanation_gts

        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.blip_tokenizer = BertTokenizerFast.from_pretrained("Salesforce/blip-image-captioning-base")


    def __len__(self):
        return len(self.fixations)

    def show_image(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()

    def show_image_and_fixation(self, img, x, y):
        plt.figure()
        plt.imshow(img)
        plt.plot(x, y, 'xb-')
        plt.show()

    def __getitem__(self, idx):
        fixation = self.fixations[idx]
        dataset = fixation["dataset"]

        if dataset == "AiR-D":
            img_name = fixation["image_id"]
            img_path = join(self.opt.dataset_dir, "AiR", "image_features", img_name.replace('jpg', 'pth'))
        elif dataset == "OSIE":
            img_name = fixation["name"]
            img_path = join(self.opt.dataset_dir, "OSIE", "image_features", img_name.replace('jpg', 'pth'))
        elif dataset == "COCO-TP":
            img_name = fixation["name"]
            img_path = join(self.opt.dataset_dir, "COCO", "image_features", img_name.replace('jpg', 'pth'))
        elif dataset == "COCO-TA":
            img_name = fixation["name"]
            img_path = join(self.opt.dataset_dir, "COCO", "image_features", img_name.replace('jpg', 'pth'))
        else:
            raise "Invalid Dataset"

        image_ftrs = torch.load(img_path)
        task = fixation["task_description"]


        origin_size_y, origin_size_x = fixation["height"], fixation["width"]
        self.downscale_x = origin_size_x / self.action_map[1]
        self.downscale_y = origin_size_y / self.action_map[0]

        scanpath = np.zeros((self.opt.max_length, self.action_map[0], self.action_map[1]), dtype=np.float32)
        # the first element denotes the termination action
        target_scanpath = np.zeros((self.opt.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
        duration = np.zeros(self.opt.max_length, dtype=np.float32)
        action_mask = np.zeros(self.opt.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.opt.max_length, dtype=np.float32)

        pos_x = np.array(fixation["X"]).astype(np.float32)
        pos_y = np.array(fixation["Y"]).astype(np.float32)
        duration_raw = np.array(fixation["T"]).astype(np.float32)

        pos_x_discrete = np.zeros(self.opt.max_length, dtype=np.int32) - 1
        pos_y_discrete = np.zeros(self.opt.max_length, dtype=np.int32) - 1
        for index in range(len(pos_x)):
            # only preserve the max_length ground-truth
            if index == self.opt.max_length:
                break
            pos_x_discrete[index] = (pos_x[index] / self.downscale_x).astype(np.int32)
            pos_y_discrete[index] = (pos_y[index] / self.downscale_y).astype(np.int32)
            if pos_x_discrete[index] < 0:
                pos_x_discrete[index] = 0
            if pos_y_discrete[index] < 0:
                pos_y_discrete[index] = 0
            if pos_x_discrete[index] >= self.opt.im_w:
                pos_x_discrete[index] = self.opt.im_w - 1
            if pos_y_discrete[index] >= self.opt.im_h:
                pos_y_discrete[index] = self.opt.im_h - 1
            duration[index] = duration_raw[index] / 1000.0
            action_mask[index] = 1
            duration_mask[index] = 1
        if action_mask.sum() <= self.opt.max_length - 1:
            action_mask[int(action_mask.sum())] = 1

        for index in range(self.opt.max_length):
            if pos_x_discrete[index] == -1 or pos_y_discrete[index] == -1:
                target_scanpath[index, 0] = 1
            else:
                scanpath[index, pos_y_discrete[index], pos_x_discrete[index]] = 1
                if self.opt.blur_sigma:
                    scanpath[index] = filters.gaussian_filter(scanpath[index], self.opt.blur_sigma)
                    scanpath[index] /= scanpath[index].sum()
                target_scanpath[index, 1:] = scanpath[index].reshape(-1)

        # image_size = [fixation["height"], fixation["width"]]
        image_size = [self.opt.height, self.opt.width]

        explanation_mask = np.zeros(self.opt.max_length, np.float32)
        if "explanation" in fixation:
            explanation = fixation["explanation"][:]
            if len(explanation) >= self.opt.max_length:
                explanation_mask += 1
                explanation = explanation[:self.opt.max_length]
            else:
                explanation_mask[: len(explanation)] = 1
                explanation = explanation + ["" for _ in range(self.opt.max_length - len(explanation))]
        else:
            explanation = ["" for _ in range(self.opt.max_length)]
        explanation_similarity_mask = np.ones((self.opt.max_length, self.opt.max_length), np.float32)
        explanation_similarity_mask[explanation_mask == 0] = 0
        explanation_similarity_mask[:, explanation_mask == 0] = 0

        return {
            "image_feature": image_ftrs,
            "task": task,
            "duration": duration,
            "action_mask": action_mask,
            "duration_mask": duration_mask,
            "target_scanpath": target_scanpath,
            "explanation": explanation,
            "explanation_mask": explanation_mask,
            "explanation_similarity_mask": explanation_similarity_mask,
            "fixation_info": fixation,
            "image_size": image_size,
            "idx": idx,
            "dataset": dataset,
            "dataset_idx": self.dataset2idx[dataset]
        }

    def collate_func(self, batch):

        image_feature_batch = []
        task_batch = []
        duration_batch = []
        action_mask_batch = []
        duration_mask_batch = []
        target_scanpath_batch = []
        explanation_batch = []
        explanation_mask_batch = []
        explanation_similarity_mask_batch = []
        fixation_info_batch = []
        image_size_batch = []
        idx_batch = []
        dataset_idx_batch = []

        for sample in batch:
            tmp_image_feature, tmp_task, tmp_duration, tmp_action_mask, tmp_duration_mask, tmp_target_scanpath, \
                tmp_explanation, tmp_explanation_mask, tmp_explanation_similarity_mask, tmp_fixation_info, tmp_image_size, tmp_idx, tmp_dataset_idx = \
                (sample["image_feature"], sample["task"], sample["duration"], sample["action_mask"], sample["duration_mask"], \
                    sample["target_scanpath"], sample["explanation"], sample["explanation_mask"], sample["explanation_similarity_mask"], sample["fixation_info"], sample["image_size"], sample["idx"], sample["dataset_idx"])
            image_feature_batch.append(tmp_image_feature)
            task_batch.append(tmp_task)
            duration_batch.append(tmp_duration)
            action_mask_batch.append(tmp_action_mask)
            duration_mask_batch.append(tmp_duration_mask)
            target_scanpath_batch.append(tmp_target_scanpath)
            explanation_batch.extend(tmp_explanation)
            explanation_mask_batch.append(tmp_explanation_mask)
            explanation_similarity_mask_batch.append(tmp_explanation_similarity_mask)
            fixation_info_batch.append(tmp_fixation_info)
            image_size_batch.append(tmp_image_size)
            idx_batch.append(tmp_idx)
            dataset_idx_batch.append(tmp_dataset_idx)

        batch_num = len(batch)
        gt_label_length = [len(_["fixation_info"]["X"]) for _ in batch]
        # for ground truth label
        gt_fixation = np.zeros((batch_num, self.max_gt_label_length, 3)) - 1
        for idx in range(batch_num):
            downscale_x = fixation_info_batch[idx]["width"] / self.opt.width
            downscale_y = fixation_info_batch[idx]["height"] / self.opt.height
            gt_fixation[idx, :gt_label_length[idx], 0] = (np.array(fixation_info_batch[idx]["X"][: gt_label_length[idx]]) /downscale_x).tolist()
            gt_fixation[idx, :gt_label_length[idx], 1] = (np.array(fixation_info_batch[idx]["Y"][: gt_label_length[idx]]) /downscale_y).tolist()
            gt_fixation[idx, :gt_label_length[idx], 2] = fixation_info_batch[idx]["T"][: gt_label_length[idx]]

        data = dict()
        data["image_feature"] = torch.stack(image_feature_batch)
        data["task"] = task_batch
        data["task_input"] = self.roberta_tokenizer(task_batch, return_tensors="pt", padding=True)
        data["duration"] = np.stack(duration_batch)
        data["action_mask"] = np.stack(action_mask_batch)
        data["duration_mask"] = np.stack(duration_mask_batch)
        data["target_scanpath"] = np.stack(target_scanpath_batch)
        data["explanation"] = self.blip_tokenizer(explanation_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.opt.max_explanation_length)
        data["explanation_mask"] = np.stack(explanation_mask_batch)
        data["explanation_similarity_mask"] = np.stack(explanation_similarity_mask_batch)
        data["explanation_string"] = explanation_batch
        data["fixation_info"] = fixation_info_batch
        data["image_size"] = np.array(image_size_batch)
        data["gt_fixation"] = np.array(gt_fixation)
        data["idx"] = np.array(idx_batch)
        data["dataset_idx"] = np.array(dataset_idx_batch)


        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}  # Turn all ndarray to torch tensor

        # in distribution system, the gather_for_metrics cannot handel dict currently
        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--stimuli_dir', default="/home/AiR/stimuli", help='stimuli folder')
    parser.add_argument('--fixation_dir', default="/home/AiR/processed_data", help='fixation folder')
    parser.add_argument('--feature_dir', default="/home/AiR/processed_feature", help='feature folder')
    parser.add_argument('--dataset_dir', default="/home/", help='feature folder')
    parser.add_argument('--datasets', default=["AiR-D", "OSIE", "COCO-TP", "COCO-TA"], nargs='+', help='used dataset')
    parser.add_argument('--split', default="train", type=str, help='control the random seed')
    parser.add_argument('--tiny', action="store_true", help='use the tiny dataset in debug')

    # image information
    parser.add_argument("--width", type=int, default=512, help="Width of input data")
    parser.add_argument("--height", type=int, default=384, help="Height of input data")
    parser.add_argument('--im_h', default=24, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")

    # fixation
    parser.add_argument("--blur_sigma", type=float, default=None, help="Standard deviation for Gaussian kernel")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")

    parser.add_argument("--max_explanation_length", type=int, default=20, help="Max explanation length")
    parser.add_argument("--min_explanation_length", type=int, default=5, help="MIn explanation length")

    args = parser.parse_args()

    train_dataset = UnifiedScanpath(args.split, args)
    example = train_dataset[0]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_func
    )
    import tqdm
    while True:
        token_len = []
        for batch in tqdm.tqdm(train_loader):
            token_len.append(batch["explanation"]["input_ids"].shape[-1])
            if batch["explanation"]["input_ids"].shape[-1] > 512:
                a = 1
        a = 1
    a = 1