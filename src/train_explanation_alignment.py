import math
import multiprocessing

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import os
import json

import numpy as np
import scipy.stats

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from accelerate.utils import tqdm
from accelerate import DistributedDataParallelKwargs
from transformers import AutoTokenizer, BertTokenizerFast

from lib.dataset.dataset import UnifiedScanpath
from lib.evaluation.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from lib.models.loss import CrossEntropyLoss, MLPLogNormalDistribution, LogAction, LogDuration, AlignmentLoss
from lib.scst.cider.cider import Cider
from lib.scst.ciderR.ciderR import CiderR
from lib.scst.tokenizer import tokenizer
from opts import parse_opt

from lib.utils.checkpointing import CheckpointManager
from lib.utils.recording import RecordManager
from lib.evaluation.evaluator import Evaluator

import lib.models.models
import lib.models.gazeformer_explanation_alignment

from lib.models.sample.sampling import Sampling

os.environ["TOKENIZERS_PARALLELISM"] = "false"


args = parse_opt()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
# These five lines control all the major sources of randomness.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, log_with=LoggerType.ALL, project_dir=args.project_dir, kwargs_handlers=[ddp_kwargs]
        )
        # accelerator = Accelerator(
        #     cpu=args.cpu, mixed_precision=args.mixed_precision, log_with=LoggerType.ALL, project_dir=args.project_dir,
        # )
    else:
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs])
        # accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)


    # setup logger
    if os.path.exists(args.project_dir) and os.path.exists(os.path.join(args.project_dir, "checkpoints/ckpt_current")):
        args.resume_dir = args.project_dir
    if args.resume_dir == "":
        log_dir = args.project_dir
    else:
        log_dir = args.resume_dir
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    hparams_file = os.path.join(log_dir, "hparams.json")
    logger = get_logger(__name__)

    if args.resume_dir == "":
        # write hparams
        with open(hparams_file, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    else:
        # read hparams
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        hparams["resume_dir"] = args.resume_dir
        for k, v in hparams.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)

    if args.start_rl_epoch == -1:
        args.start_rl_epoch = args.epochs

    # We need to initialize the trackers we use, and also store our configuration
    config = {}
    for (key, value) in vars(args).items():
        if isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, bool) or isinstance(value, torch.Tensor):
            config[key] = value
    if args.with_tracking:
        init_kwargs = {
            "wandb": {"resume": args.resume_wandb},
        }
        accelerator.init_trackers(project_name=args.project_name, config=config, init_kwargs=init_kwargs)


    accelerator.print("The args corresponding to training process are: ")
    for (key, value) in vars(args).items():
        accelerator.print("{key:50}: {value:}".format(key=key, value=value))

    # Record manager for writing and loading the best metrics and theirs corresponding epoch
    record_manager = RecordManager(log_dir)
    if args.resume_dir == '':
        record_manager.init_record()
    else:
        record_manager.load()

    start_epoch = record_manager.get_epoch()
    iteration = record_manager.get_iteration()
    best_metric = record_manager.get_best_metric()


    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER, MODEL, OPTIMIZER
    # --------------------------------------------------------------------------------------------

    train_dataset = UnifiedScanpath(split="train", opt=args)
    eval_dataset = UnifiedScanpath(split="validation", opt=args)

    # Instantiate dataloaders.
    train_all_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_func
    )
    train_rl_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch // 2,
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.collate_func
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=4,
        collate_fn=eval_dataset.collate_func,
        drop_last=False
    )


    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    transformer = lib.models.models.Transformer(args=args)
    model = lib.models.gazeformer_explanation_alignment.gazeformer(transformer=transformer, args=args)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    all_params = list(model.parameters())

    # we do not train the embedding since it has an inplace operation
    bert_embedding_params = list(model.roberta.embeddings.parameters())
    bert_embedding_params_ptr = [p.data_ptr() for p in bert_embedding_params]

    bert_params = []
    bert_params_ptr = []
    for p in model.roberta.parameters():
        if p.data_ptr() not in bert_embedding_params_ptr:
            bert_params.append(p)
            bert_params_ptr.append(p.data_ptr())

    other_params = []
    for p in all_params:
        if p.data_ptr() not in bert_params_ptr and p.data_ptr() not in bert_embedding_params_ptr:
            other_params.append(p)

    # froze the embedding
    for p in bert_embedding_params:
        p.requires_grad = False


    optimizer = torch.optim.Adam(params=[
        {'params': other_params, 'lr': args.lr},
        {'params': bert_params, 'lr': args.lr * 0.1},
    ], betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)


    len_train_all_dataloader = len(train_all_dataloader)
    len_train_rl_dataloader = len(train_rl_dataloader)
    def lr_lambda(iteration):
        if iteration <= len_train_all_dataloader * args.warmup_epoch:
            return iteration / (len_train_all_dataloader * args.warmup_epoch)
        elif iteration <= len_train_all_dataloader * args.start_rl_epoch:
            return 1 - (iteration - len_train_all_dataloader * args.warmup_epoch) / \
                (len_train_all_dataloader * (args.start_rl_epoch - args.warmup_epoch))
        else:
            return args.rl_lr_initial_decay * (1 - (iteration - (len_train_all_dataloader * args.start_rl_epoch)) /
                                               (len_train_rl_dataloader * (args.epochs - args.start_rl_epoch)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if iteration != -1:
        lr_scheduler.step(iteration)



    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_all_dataloader, train_rl_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_all_dataloader, train_rl_dataloader, eval_dataloader, lr_scheduler
    )


    # Initialize the Evaluator
    evaluator = Evaluator(args)

    # Construct a cider evaluation object
    tokenizer_pool = multiprocessing.Pool()
    ptb_tokenizer = PTBTokenizer()
    # cider_evaluator = Cider(ptb_tokenizer.tokenize(train_dataset.explanation_gts))
    ciderR_evaluator = CiderR(ptb_tokenizer.tokenize(train_dataset.explanation_gts))

    # blip_tokenizer
    blip_tokenizer = BertTokenizerFast.from_pretrained("Salesforce/blip-image-captioning-base")

    # --------------------------------------------------------------------------------------------
    #  BEFORE TRAINING STARTS
    # --------------------------------------------------------------------------------------------
    # Checkpoint manager to serialize checkpoints periodically while training and keep track of
    # best performing checkpoint.
    if accelerator.is_main_process:
        checkpoint_manager = CheckpointManager(accelerator, checkpoints_dir, mode="max", best_metric=best_metric)

    # Load checkpoint to resume training from there if specified.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    # saved checkpoints if you intend to continue training.
    if args.resume_dir != "":
        resume_from_checkpoint = os.path.join(checkpoints_dir, "ckpt_current")
        accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint, strict=False)

    def train(iteration, epoch):
        # traditional training stage
        train_dataloader = train_all_dataloader

        if epoch < args.start_rl_epoch:
            model.train()
            with tqdm(total=len(train_dataloader)) as pbar:
                for i_batch, batch in enumerate(train_dataloader):
                    batch = {k: v.to(accelerator.device) if not isinstance(v, list) else v for k, v in batch.items()}

                    prediction = model(batch)


                    loss_actions = CrossEntropyLoss(prediction["actions"], batch["target_scanpath"], batch["action_mask"])
                    loss_duration = MLPLogNormalDistribution(prediction["log_normal_mu"], prediction["log_normal_sigma2"],
                                                             batch["duration"], batch["duration_mask"])
                    loss_lm = prediction["dec_outputs"].loss
                    visual_alignment_loss, language_alignment_loss, multimodal_alignment_loss = \
                        AlignmentLoss(prediction["resnet_visual_similarity"],
                        prediction["visual_similarity"], prediction["language_similarity"],
                        prediction["multimodal_similarity"], batch["explanation_similarity_mask"])
                    alignment_loss = visual_alignment_loss + language_alignment_loss + multimodal_alignment_loss
                    loss = loss_actions + args.lambda_duration * loss_duration + args.lambda_lm * loss_lm + args.lambda_alignment * alignment_loss


                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    iteration += 1

                    # Log loss and learning rate to tensorboard.
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "loss": loss.detach().float(),
                                "loss_actions": loss_actions.detach().float(),
                                "loss_duration": loss_duration.detach().float(),
                                "loss_lm": loss_lm.detach().float(),
                                "visual_alignment_loss": visual_alignment_loss.detach().float(),
                                "language_alignment_loss": language_alignment_loss.detach().float(),
                                "multimodal_alignment_loss": multimodal_alignment_loss.detach().float(),
                                "alignment_loss": alignment_loss.detach().float(),
                                "learning_rate": optimizer.param_groups[0]["lr"]
                            },
                            step=iteration,
                        )

                    pbar.update()

        # reinforcement learning stage
        else:
            model.eval()
            # create a ScanMatch object
            with tqdm(total=len(train_rl_dataloader)) as pbar:
                for i_batch, batch in enumerate(train_rl_dataloader):
                    # batch = {k: v.to(accelerator.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    batch = {k: v.to(accelerator.device) if not isinstance(v, list) else v for k, v in batch.items()}

                    metrics_reward_batch = []
                    neg_log_actions_batch = []
                    neg_log_durations_batch = []
                    prediction, scanpath_prediction, generated_ids, generated_idx_logit, dec_output_loss, \
                        sampling_prediction, action_masks, duration_masks = model(batch, args.rl_sample_number, True)

                    gts = {}
                    res = {}
                    for trial in range(args.rl_sample_number):
                        # transform the torch type prediction as json
                        predictions = evaluator.transform(scanpath_prediction[:, trial])
                        targets = evaluator.transform(batch["gt_fixation"])
                        image_size = batch["image_size"]

                        # evaluation on the prediction and ground truth
                        metrics_reward = evaluator.measure_scanmatch(targets, predictions, image_size)

                        metrics_reward = torch.tensor(metrics_reward, dtype=torch.float32).to(
                            batch["image_feature"].get_device())
                        neg_log_actions = - LogAction(sampling_prediction["selected_actions_probs"][:, trial],
                                                      action_masks[:, trial])
                        durations_sample = sampling_prediction["durations"][:, trial].data.clone()
                        neg_log_durations = - LogDuration(durations_sample, prediction["log_normal_mu"],
                                                          prediction["log_normal_sigma2"], duration_masks[:, trial])
                        metrics_reward_batch.append(metrics_reward.unsqueeze(0))
                        neg_log_actions_batch.append(neg_log_actions.unsqueeze(0))
                        neg_log_durations_batch.append(neg_log_durations.unsqueeze(0))


                        # explanation aspect
                        for idx in range(generated_ids.shape[0]):
                            tmp_caption = blip_tokenizer.batch_decode(generated_ids[idx, :, trial].view(-1, generated_ids.shape[-1]), skip_special_tokens=True)

                            gts.setdefault("idx:{}-trial:{}".format(idx, trial), []).append(
                                " ".join(batch["fixation_info"][idx]["explanation"]))
                            res.setdefault("idx:{}-trial:{}".format(idx, trial), []).append(
                                " ".join(tmp_caption).strip())

                    neg_log_actions_tensor = torch.cat(neg_log_actions_batch, dim=0)
                    neg_log_durations_tensor = torch.cat(neg_log_durations_batch, dim=0)
                    # use the mean as reward
                    metrics_reward_tensor = torch.cat(metrics_reward_batch, dim=0)
                    metrics_reward_hmean = scipy.stats.hmean(metrics_reward_tensor.cpu(), axis=-1)
                    metrics_reward_hmean_tensor = torch.tensor(metrics_reward_hmean).to(
                        metrics_reward_tensor.get_device())
                    baseline_reward_hmean_tensor = metrics_reward_hmean_tensor.mean(0, keepdim=True)

                    # caption
                    caps_gen, caps_gt = tokenizer_pool.map(tokenizer.PTBTokenizer().tokenize, [res, gts])
                    # cider_reward = cider_evaluator.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                    cider_reward = ciderR_evaluator.compute_score(caps_gt, caps_gen)[1].astype(np.float32)

                    loss_actions = (neg_log_actions_tensor * (
                            metrics_reward_hmean_tensor - baseline_reward_hmean_tensor)).sum()
                    loss_duration = (neg_log_durations_tensor * (
                            metrics_reward_hmean_tensor - baseline_reward_hmean_tensor)).sum()
                    loss_lm = prediction["dec_outputs"].loss

                    # alignment loss
                    visual_alignment_loss, language_alignment_loss, multimodal_alignment_loss = \
                        AlignmentLoss(prediction["resnet_visual_similarity"],
                                      prediction["visual_similarity"], prediction["language_similarity"],
                                      prediction["multimodal_similarity"], prediction["explanation_similarity_mask"])
                    alignment_loss = visual_alignment_loss + language_alignment_loss + multimodal_alignment_loss

                    loss = loss_actions + loss_duration + args.lambda_lm * loss_lm + args.lambda_alignment * alignment_loss

                    # update the parameters
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    iteration += 1

                    # Log loss and learning rate to tensorboard.
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "loss": loss.detach().float(),
                                "reward_hmean": metrics_reward_hmean.mean(),
                                "cider-R": cider_reward.mean().item(),
                                "loss_lm": loss_lm.detach().float(),
                                "visual_alignment_loss": visual_alignment_loss.detach().float(),
                                "language_alignment_loss": language_alignment_loss.detach().float(),
                                "multimodal_alignment_loss": multimodal_alignment_loss.detach().float(),
                                "alignment_loss": alignment_loss.detach().float(),
                                "learning_rate": optimizer.param_groups[0]["lr"]
                            },
                            step=iteration,
                        )

                    pbar.update()


        return iteration

    def validation(iteration):
        model.eval()
        prediction_scanpaths = []
        gt_scanpaths = []
        image_sizes = []
        evaluation_scores = []
        idx_batch = []
        dataset_idxes = []
        all_generated_ids = []
        with tqdm(total=len(eval_dataloader)) as pbar:
            for i_batch, batch in enumerate(eval_dataloader):
                batch = {k: v.to(accelerator.device) if not isinstance(v, list) else v for k, v in batch.items()}

                with torch.no_grad():
                    prediction, scanpath_prediction, generated_ids, \
                        sampling_prediction, action_masks, duration_masks = model(batch)

                # by default we repeat once for inference
                scanpath_prediction = scanpath_prediction[:, 0]
                generated_ids = generated_ids[:, :, 0]
                if i_batch != len(eval_dataloader) - 1:
                    # not the last iter, do not need to drop, the evaluation can be done in different GPU

                    # transform the torch type prediction as json
                    predictions = evaluator.transform(scanpath_prediction)
                    targets = evaluator.transform(batch["gt_fixation"])

                    # evaluation on the prediction and ground truth
                    scores = evaluator.measure(targets, predictions, batch)

                    # gather for metric
                    scanpath_prediction = accelerator.gather_for_metrics(scanpath_prediction)
                    gt_fixation = accelerator.gather_for_metrics(batch["gt_fixation"])
                    image_size = accelerator.gather_for_metrics(batch["image_size"])
                    scores = accelerator.gather_for_metrics(scores)
                    idx = accelerator.gather_for_metrics(batch["idx"])
                    dataset_idx = accelerator.gather_for_metrics(batch["dataset_idx"])
                    generated_ids = accelerator.gather_for_metrics(generated_ids)

                    if accelerator.is_main_process:
                        prediction_scanpaths.extend(scanpath_prediction)
                        gt_scanpaths.extend(gt_fixation)
                        image_sizes.extend(image_size)
                        evaluation_scores.extend(scores)
                        idx_batch.extend(idx)
                        dataset_idxes.extend(dataset_idx)
                        all_generated_ids.extend(generated_ids)


                else:
                    # the last iter, need to drop, the evaluation cannot be done in different GPU
                    # gather for metric
                    scanpath_prediction = accelerator.gather_for_metrics(scanpath_prediction)
                    gt_fixation = accelerator.gather_for_metrics(batch["gt_fixation"])
                    image_size = accelerator.gather_for_metrics(batch["image_size"])
                    idx = accelerator.gather_for_metrics(batch["idx"])
                    dataset_idx = accelerator.gather_for_metrics(batch["dataset_idx"])
                    generated_ids = accelerator.gather_for_metrics(generated_ids)
                    fixation_info = [eval_dataloader.dataset.fixations[_] for _ in idx]
                    batch["image_size"] = image_size
                    batch["dataset_idx"] = dataset_idx
                    batch["fixation_info"] = fixation_info

                    # transform the torch type prediction as json
                    predictions = evaluator.transform(scanpath_prediction)
                    targets = evaluator.transform(gt_fixation)

                    # evaluation on the prediction and ground truth
                    scores = evaluator.measure(targets, predictions, batch)

                    if accelerator.is_main_process:
                        prediction_scanpaths.extend(scanpath_prediction)
                        gt_scanpaths.extend(gt_fixation)
                        image_sizes.extend(image_size)
                        evaluation_scores.extend(scores)
                        idx_batch.extend(idx)
                        dataset_idxes.extend(dataset_idx)
                        all_generated_ids.extend(generated_ids)

                # accelerator.print(scanpath_prediction.shape)
                # accelerator.print(gt_fixation.shape)
                # accelerator.print(image_size.shape)
                # accelerator.print(len(scores))
                # accelerator.print(len(fixation_info))
                # accelerator.print(len(idx_batch))

                pbar.update()

        # transform the gather scanpath prediction to JSON format file
        if accelerator.is_main_process:
            scanpath_dataset = ["AiR-D", "OSIE", "COCO-TP", "COCO-TA", "COCO-FV"]
            json_prediction_scanpaths = evaluator.transform(prediction_scanpaths)
            idx_list = torch.stack(idx_batch).cpu().numpy().tolist()
            json_gt_scanpaths = eval_dataloader.dataset.fixations
            json_targets = evaluator.transform(gt_scanpaths)

            # collect for saliency evaluation
            prediction_fixation_dict = {}
            gt_fixation_dict = {}
            for iter in range(len(json_gt_scanpaths)):
                cur_dataset = scanpath_dataset[dataset_idxes[iter]]
                if cur_dataset == "AiR-D":
                    key = "{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["question_id"])
                elif cur_dataset == "OSIE":
                    key = "{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["name"])
                elif cur_dataset == "COCO-TP":
                    key = "{}-{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["task"],
                                            json_gt_scanpaths[iter]["name"])
                elif cur_dataset == "COCO-TA":
                    key = "{}-{}-{}".format(cur_dataset, json_gt_scanpaths[iter]["task"],
                                            json_gt_scanpaths[iter]["name"])
                else:
                    raise "Invalid dataset"
                prediction_fixation_dict.setdefault(key, []).append(json_prediction_scanpaths[iter])
                gt_fixation_dict.setdefault(key, []).append(json_targets[iter])

            # collect for caption explanation
            explanation_res = {}
            for iter in range(len(json_gt_scanpaths)):
                cur_dataset = scanpath_dataset[dataset_idxes[iter]]
                if cur_dataset == "AiR-D":
                    key = "{}-{}".format(json_gt_scanpaths[iter]["question_id"],
                                         json_gt_scanpaths[iter]["subject"])
                elif cur_dataset == "OSIE":
                    key = "{}-{}".format(json_gt_scanpaths[iter]["name"][:-4],
                                         json_gt_scanpaths[iter]["subject"])
                elif cur_dataset == "COCO-TP":
                    key = "TP-{}-{}-{}".format(json_gt_scanpaths[iter]["task"],
                                               json_gt_scanpaths[iter]["name"][:-4],
                                               json_gt_scanpaths[iter]["subject"])
                elif cur_dataset == "COCO-TA":
                    key = "TA-{}-{}-{}".format(json_gt_scanpaths[iter]["task"],
                                               json_gt_scanpaths[iter]["name"][:-4],
                                               json_gt_scanpaths[iter]["subject"])
                else:
                    raise "Invalid dataset"

                # explanation_res.setdefault(key, []).append(
                #     {
                #         "caption": " ".join(blip_tokenizer.batch_decode(all_generated_ids[iter], skip_special_tokens=True)).strip(),
                #     }
                # )

                tmp_caption = blip_tokenizer.batch_decode(all_generated_ids[iter], skip_special_tokens=True)
                # tmp_caption = ["there is " + _ if len(_) > 0 else _ for _ in tmp_caption]
                explanation_res.setdefault(key, []).append(
                    {
                        "caption": " ".join(tmp_caption).strip(),
                    }
                )

            all_saliency_evals = evaluator.eval_saliency(prediction_fixation_dict, gt_fixation_dict)

            # evaluation on captioning
            gts = eval_dataloader.dataset.explanation_gts
            res = explanation_res
            evaluator.explanation_evaluation(gts, res)

            for idx in idx_list:
                json_prediction_scanpath = json_prediction_scanpaths[idx]
                json_gt_scanpath = json_gt_scanpaths[idx]
                json_prediction_scanpath["dataset"] = json_gt_scanpath["dataset"]
                json_prediction_scanpath["evaluation_scores"] = evaluation_scores[idx]

                cur_dataset = scanpath_dataset[dataset_idxes[idx]]
                if cur_dataset == "AiR-D":
                    key = "{}-{}".format(json_gt_scanpaths[idx]["question_id"],
                                         json_gt_scanpaths[idx]["subject"])
                elif cur_dataset == "OSIE":
                    key = "{}-{}".format(json_gt_scanpaths[idx]["name"][:-4],
                                         json_gt_scanpaths[idx]["subject"])
                elif cur_dataset == "COCO-TP":
                    key = "TP-{}-{}-{}".format(json_gt_scanpaths[idx]["task"],
                                               json_gt_scanpaths[idx]["name"][:-4],
                                               json_gt_scanpaths[idx]["subject"])
                elif cur_dataset == "COCO-TA":
                    key = "TA-{}-{}-{}".format(json_gt_scanpaths[idx]["task"],
                                               json_gt_scanpaths[idx]["name"][:-4],
                                               json_gt_scanpaths[idx]["subject"])
                else:
                    raise "Invalid dataset"

                json_prediction_scanpath["explanation"] = explanation_res[key]
                json_prediction_scanpath["gt_explanation"] = gts[key]


                # log the caption evaluation
                evaluation_scores[idx]['Bleu_4'] = evaluator.scanpath_eval.evalImgs[idx]['Bleu_4']
                evaluation_scores[idx]['METEOR'] = evaluator.scanpath_eval.evalImgs[idx]['METEOR']
                evaluation_scores[idx]['ROUGE_L'] = evaluator.scanpath_eval.evalImgs[idx]['ROUGE_L']
                evaluation_scores[idx]['CIDEr'] = evaluator.scanpath_eval.evalImgs[idx]['CIDEr']
                evaluation_scores[idx]['CIDEr-R'] = evaluator.scanpath_eval.evalImgs[idx]['CIDEr-R']

            dataset_evaluation_scores_dict = {}
            for iter, evaluation_score in enumerate(evaluation_scores):
                dataset_evaluation_scores_dict.setdefault(scanpath_dataset[dataset_idxes[iter]], []).append(
                    evaluation_score)

            dataset_gather_scores_dict = {}
            for dataset_name, evaluation_score in dataset_evaluation_scores_dict.items():
                if len(evaluation_score) > 0:
                    if dataset_name in ["AiR-D", "OSIE"]:
                        saliency_eval = np.array(
                            [v for k, v in all_saliency_evals.items() if dataset_name in k])
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
                            "Bleu_4": np.array([_["Bleu_4"] for _ in evaluation_score]),
                            "METEOR": np.array([_["METEOR"] for _ in evaluation_score]),
                            "ROUGE_L": np.array([_["ROUGE_L"] for _ in evaluation_score]),
                            "CIDEr": np.array([_["CIDEr"] for _ in evaluation_score]),
                            "CIDEr-R": np.array([_["CIDEr-R"] for _ in evaluation_score]),
                        }
                    else:
                        saliency_eval = np.array(
                            [v for k, v in all_saliency_evals.items() if dataset_name in k])
                        gather_score = {
                            "scanmatch_score": np.array([_["scanmatch_score"] for _ in evaluation_score]),
                            "multimatch_score": np.array([_["multimatch_score"] for _ in evaluation_score]),
                            "sed_score": np.array([_["sed_score"] for _ in evaluation_score]),
                            "stde_score": np.array([_["stde_score"] for _ in evaluation_score]),
                            "SS_score": np.array([_["SS_score"] for _ in evaluation_score]),
                            "SSS_score": np.array([_["SSS_score"] for _ in evaluation_score if not np.isnan(sum(_["SSS_score"]))]),
                            "CC": saliency_eval[:, 0],
                            "AUC": saliency_eval[:, 1],
                            "NSS": saliency_eval[:, 2],
                            "sAUC": saliency_eval[:, 3],
                            "KLD": saliency_eval[:, 4],
                            "SIM": saliency_eval[:, 5],
                            "Bleu_4": np.array([_["Bleu_4"] for _ in evaluation_score]),
                            "METEOR": np.array([_["METEOR"] for _ in evaluation_score]),
                            "ROUGE_L": np.array([_["ROUGE_L"] for _ in evaluation_score]),
                            "CIDEr": np.array([_["CIDEr"] for _ in evaluation_score]),
                            "CIDEr-R": np.array([_["CIDEr-R"] for _ in evaluation_score]),
                        }
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
                "SSS_score": np.array([_["SSS_score"] for _ in evaluation_scores if "SSS_score" in _ and  not np.isnan(sum(_["SSS_score"]))]),
                "CC": saliency_eval[:, 0],
                "AUC": saliency_eval[:, 1],
                "NSS": saliency_eval[:, 2],
                "sAUC": saliency_eval[:, 3],
                "KLD": saliency_eval[:, 4],
                "SIM": saliency_eval[:, 5],
                "Bleu_4": np.array([_["Bleu_4"] for _ in evaluation_scores]),
                "METEOR": np.array([_["METEOR"] for _ in evaluation_scores]),
                "ROUGE_L": np.array([_["ROUGE_L"] for _ in evaluation_scores]),
                "CIDEr": np.array([_["CIDEr"] for _ in evaluation_scores]),
                "CIDEr-R": np.array([_["CIDEr-R"] for _ in evaluation_scores]),
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

            # log all evaluation metrics to tensorboard.
            if args.with_tracking:
                accelerator.log(
                    dataset_cur_metrics_dict,
                    step=iteration,
                )

            predict_results = json_prediction_scanpaths
            save_path = os.path.join(args.project_dir, "validation")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, "metric.json"), "w") as f:
                json.dump(dataset_cur_metrics_dict, f, indent=2)
            with open(os.path.join(save_path, "predictions.json"), "w") as f:
                json.dump(predict_results, f, indent=2)

            return cur_metrics
        else:
            return None

    for epoch in range(start_epoch + 1, args.epochs):
        accelerator.print("-" * 50)
        accelerator.print("Running the {epoch:2d}-th epoch...".format(epoch=epoch))

        iteration = train(iteration, epoch)
        if (epoch < args.start_rl_epoch and (epoch + 1) % args.checkpoint_every == 0) or \
                (epoch >= args.start_rl_epoch and (epoch + 1) % args.checkpoint_every_rl == 0):
            accelerator.print("Evaluating the {epoch:2d}-th epoch...".format(epoch=epoch))

            cur_metrics = validation(iteration)

            # save
            if accelerator.is_main_process:
                cur_metric = scipy.stats.hmean([cur_metrics["metrics/SM without Dur"], cur_metrics["metrics/SM with Dur"]])
                checkpoint_manager.step(accelerator, float(cur_metric))
                best_metric = checkpoint_manager.get_best_metric()
                record_manager.save(epoch, iteration, best_metric)

                # Log loss and learning rate to tensorboard.
                if args.with_tracking:
                    accelerator.log(
                        {
                            "metrics/cur_metric": cur_metric,
                            "epoch": epoch
                        },
                        step=iteration,
                    )

        else:
            # save current epoch
            if accelerator.is_main_process:
                checkpoint_manager.step(accelerator, float(np.nan))
                best_metric = checkpoint_manager.get_best_metric()
                record_manager.save(epoch, iteration, best_metric)

        # check  whether to save the final supervised training file
        if args.supervised_save and epoch == args.start_rl_epoch - 1:
            # Serialize best performing checkpoint observed so far.
            output_dir = os.path.join(checkpoints_dir, f"ckpt_supervised_end")
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

if __name__ == "__main__":
    main()
