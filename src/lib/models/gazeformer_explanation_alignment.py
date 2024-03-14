import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel

from lib.models.models import ResNetCOCO
from lib.models.positional_encodings import PositionEmbeddingSine2d
import math
from typing import Optional, List

from lib.models.sample.sampling import Sampling
from transformers import AutoTokenizer, RobertaModel, BertModel, AutoModelForSequenceClassification, BertTokenizerFast, \
    BlipProcessor, BlipForConditionalGeneration

eps = 1e-16


class CrossAttentionPredictor(nn.Module):
    def __init__(self, nhead=8, dropout=0.4, d_model=512):
        super(CrossAttentionPredictor, self).__init__()
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self._reset_parameters(self.self_attn)
        self._reset_parameters(self.multihead_attn)

    def _reset_parameters(self, mod):
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward(self, tgt, memory, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                querypos_embed: Optional[Tensor] = None,
                patchpos_embed: Optional[Tensor] = None):
        q = k = v = self.with_pos_embed(tgt, querypos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        att = self.multihead_attn(query=self.with_pos_embed(tgt, querypos_embed),
                                  key=patchpos_embed(memory),
                                  value=memory, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[1]
        att_logit = torch.log(att + eps)

        return att_logit


class gazeformer(nn.Module):
    def __init__(self, transformer, args):
        super(gazeformer, self).__init__()
        self.args = args

        spatial_dim = (args.im_h, args.im_w)
        dropout = args.cls_dropout
        max_len = args.max_length
        patch_size = args.patch_size

        self.spatial_dim = spatial_dim
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        # fixation embeddings
        self.querypos_embed = nn.Embedding(max_len, self.hidden_dim)
        # 2D patch positional encoding
        self.patchpos_embed = PositionEmbeddingSine2d(spatial_dim, hidden_dim=self.hidden_dim, normalize=True)

        # classify fixation, or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_dim, 1)
        # Gaussian parameters for x,y,t
        self.generator_t_mu = nn.Linear(self.hidden_dim, 1)
        self.generator_t_logvar = nn.Linear(self.hidden_dim, 1)

        self.max_len = max_len

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.LogSoftmax(dim=-1)
        # projection for first fixation encoding
        self.attention_map_predictor = CrossAttentionPredictor(self.args.nhead, dropout, self.args.hidden_dim)

        # sampling modules
        self.sampling = Sampling(args=args)

        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # BlipTextLMHeadModel https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/blip/modeling_blip_text.py#L807
        # https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb
        self.blip_model = BlipTextLMHeadModel.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_tokenizer = BertTokenizerFast.from_pretrained("Salesforce/blip-image-captioning-base")

        self.termination_feature = nn.Parameter(torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32))
        self.image_emb = nn.Parameter(torch.zeros((1, self.hidden_dim), dtype=torch.float32))
        self.fixation_emb = nn.Parameter(torch.zeros((1, self.hidden_dim), dtype=torch.float32))
        self.task_emb = nn.Parameter(torch.zeros((1, self.args.lm_hidden_dim), dtype=torch.float32))
        self.hidden_state_transform = nn.Linear(self.hidden_dim, self.args.lm_hidden_dim)

        # alignment layer
        self.visual_projection = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        self.language_projection = nn.Sequential(nn.Linear(self.args.lm_hidden_dim, self.hidden_dim), nn.ReLU())

        self.init_weights()


    def init_weights(self):
        for modules in [self.querypos_embed.modules(), self.token_predictor.modules(),
                        self.generator_t_mu.modules(), self.generator_t_logvar.modules()]:
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=1)

    def get_fixation_map(self, y_mu, y_logvar, x_mu, x_logvar):
        y_grid = self.y_grid.unsqueeze(0).unsqueeze(0)
        x_grid = self.x_grid.unsqueeze(0).unsqueeze(0)
        y_mu, y_logvar, x_mu, x_logvar = y_mu.unsqueeze(-1), y_logvar.unsqueeze(-1), x_mu.unsqueeze(
            -1), x_logvar.unsqueeze(-1)
        y_std = torch.exp(0.5 * y_logvar)
        x_std = torch.exp(0.5 * x_logvar)

        exp_term = (y_grid - y_mu) ** 2 / (y_std ** 2 + eps) + (x_grid - x_mu) ** 2 / (x_std ** 2 + eps)
        fixation_map = 1 / (2 * math.pi) / (y_std + eps) / (x_std + eps) * torch.exp(-0.5 * exp_term)
        fixation_map = fixation_map.view(fixation_map.shape[0], fixation_map.shape[1], -1)
        fixation_map = fixation_map / (fixation_map.sum(-1, keepdim=True) + eps)
        return fixation_map

    # reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: List[Tensor or List], sample_num: int=1, scst: bool=False, detail: bool=False):
        if self.training:
            prediction = self.training_process(batch)
        else:
            prediction = self.inference(batch, sample_num, scst, detail)
        # prediction = self.training_process(batch)

        return prediction

    def training_process(self, batch):
        src = batch["image_feature"]

        outputs = self.roberta(**batch["task_input"])
        task = outputs.pooler_output

        tgt_input = src.new_zeros(
            (self.max_len, src.size(0), self.hidden_dim))  # Notice that this where we convert target input to zeros
        # a  = src.detach().cpu().numpy()
        # tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:,1], :])
        memory, memory_task, outs = self.transformer(src=src, tgt=tgt_input, tgt_mask=None, tgt_key_padding_mask=None,
                                             task=task,  querypos_embed=self.querypos_embed.weight.unsqueeze(1),
                                             patchpos_embed=self.patchpos_embed)

        outs = self.dropout(outs)
        # get Gaussian parameters for (t)
        t_log_normal_mu, t_log_normal_sigma2 = self.generator_t_mu(outs), torch.exp(self.generator_t_logvar(outs))
        action_map = self.attention_map_predictor(outs, memory_task, tgt_mask=None, memory_mask=None,
                                                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                                  querypos_embed=self.querypos_embed.weight.unsqueeze(1),
                                                  patchpos_embed=self.patchpos_embed)

        token_prediction = self.token_predictor(outs).permute(1, 0, 2)

        action_map_prob = F.softmax(action_map, -1)
        token_prediction_prob = torch.sigmoid(token_prediction)

        z = torch.cat([token_prediction_prob, action_map_prob * (1 - token_prediction_prob)], dim=-1)
        z = torch.log(z + eps)

        aggr_action_map = action_map
        aggr_z = z

        if self.training == False:
            aggr_z = F.softmax(aggr_z, -1)

        # visual feature gt alignment from resnet pretrain feature
        visual_feature_resnet = torch.cat([src.new_zeros((src.shape[0], 1, src.shape[-1])), src], dim=1).permute(1, 0, 2)
        target_scanpath_index = \
            torch.argmax(batch["target_scanpath"], dim=-1, keepdim=True).repeat(1, 1,visual_feature_resnet.shape[-1]).unsqueeze(0)
        fixated_visual_feature_resnet = torch.gather(
            visual_feature_resnet.unsqueeze(2).repeat(1, 1, target_scanpath_index.shape[2], 1), dim=0, index=target_scanpath_index).squeeze(0)
        fixated_visual_feature_resnet = fixated_visual_feature_resnet / (
                    torch.norm(fixated_visual_feature_resnet, p=2, dim=2, keepdim=True) + eps)
        resnet_visual_similarity = torch.matmul(fixated_visual_feature_resnet, fixated_visual_feature_resnet.transpose(1, 2))

        # cat the memory feature
        cat_memory = torch.cat([self.termination_feature.repeat(1, memory.shape[1], 1), memory], dim=0)

        # for explanation
        target_scanpath_index = torch.argmax(batch["target_scanpath"], dim=-1, keepdim=True).repeat(1, 1, cat_memory.shape[-1]).unsqueeze(0)
        aggr_fixation_feature = torch.gather(cat_memory.unsqueeze(2).repeat(1, 1, target_scanpath_index.shape[2], 1), dim=0, index=target_scanpath_index).squeeze(0)

        # visual project feature for alignment
        aggr_fixation_projection_feature = self.visual_projection(aggr_fixation_feature)
        aggr_fixation_projection_feature = aggr_fixation_projection_feature / (torch.norm(aggr_fixation_projection_feature, p=2, dim=2, keepdim=True) + eps)
        visual_similarity = torch.matmul(aggr_fixation_projection_feature, aggr_fixation_projection_feature.transpose(1, 2))

        # encoder_hidden_states = self.hidden_state_transform(
        #     torch.cat([self.image_emb + memory, self.fixation_emb + aggr_fixation_feature], dim=0).permute(1, 0, 2))

        encoder_hidden_state_list = []
        for idx in range(aggr_fixation_feature.shape[0]):
            transform_feature = self.hidden_state_transform(torch.stack([self.fixation_emb + aggr_fixation_feature[idx]])).permute(1, 0, 2)
            transform_feature = torch.cat([transform_feature, (self.task_emb + task[idx].unsqueeze(0)).unsqueeze(1).repeat(transform_feature.shape[0], 1, 1)], dim=1)
            encoder_hidden_state_list.append(transform_feature)
        encoder_hidden_states = torch.stack(encoder_hidden_state_list)
        encoder_hidden_states = encoder_hidden_states.view(-1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1])

        # [batch, C, Feature]
        explanation = batch["explanation"]
        dec_input = {
            "input_ids": explanation.input_ids,
            "labels": explanation.input_ids,
            "attention_mask": explanation.attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "output_hidden_states": True
            # "reduction": "none"
        }
        dec_outputs = self.blip_model(**dec_input)

        # Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer)
        cls_language_feature = dec_outputs.hidden_states[-1][:, 0]

        # language project feature for alignment
        cls_language_feature_projection = self.language_projection(cls_language_feature)
        cls_language_feature_projection = cls_language_feature_projection.view(src.shape[0], -1, cls_language_feature_projection.shape[-1])
        cls_language_feature_projection = cls_language_feature_projection / (
                    torch.norm(cls_language_feature_projection, p=2, dim=2, keepdim=True) + eps)
        language_similarity = torch.matmul(cls_language_feature_projection, cls_language_feature_projection.transpose(1, 2))

        multimodal_similarity = torch.matmul(aggr_fixation_projection_feature, cls_language_feature_projection.transpose(1, 2))


        prediction = {}
        # [N, T, A] A = H * W + 1
        prediction['actions'] = aggr_z
        # [N, T]
        prediction['log_normal_mu'] = t_log_normal_mu.permute(1, 0, 2).squeeze(-1)
        # [N, T]
        prediction['log_normal_sigma2'] = t_log_normal_sigma2.permute(1, 0, 2).squeeze(-1)
        # [N, T, H, W]
        prediction["action_map"] = aggr_action_map.view(-1, self.max_len, self.spatial_dim[0], self.spatial_dim[1])
        # CausalLMOutputWithCrossAttention
        prediction["dec_outputs"] = dec_outputs
        # [N, T, T]
        prediction["resnet_visual_similarity"] = resnet_visual_similarity
        # [N, T, T]
        prediction["visual_similarity"] = visual_similarity
        # [N, T, T]
        prediction["language_similarity"] = language_similarity
        # [N, T, T]
        prediction["multimodal_similarity"] = multimodal_similarity

        return prediction

    def inference(self, batch, sample_num, scst, detail=False):
        src = batch["image_feature"]

        outputs = self.roberta(**batch["task_input"])
        task = outputs.pooler_output

        tgt_input = src.new_zeros(
            (self.max_len, src.size(0), self.hidden_dim))  # Notice that this where we convert target input to zeros
        # a  = src.detach().cpu().numpy()
        # tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:,1], :])
        memory, memory_task, outs = self.transformer(src=src, tgt=tgt_input, tgt_mask=None, tgt_key_padding_mask=None,
                                             task=task,
                                             querypos_embed=self.querypos_embed.weight.unsqueeze(1),
                                             patchpos_embed=self.patchpos_embed)

        outs = self.dropout(outs)
        # get Gaussian parameters for (t)
        t_log_normal_mu, t_log_normal_sigma2 = self.generator_t_mu(outs), torch.exp(self.generator_t_logvar(outs))
        action_map = self.attention_map_predictor(outs, memory_task, tgt_mask=None, memory_mask=None,
                                                  tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                                  querypos_embed=self.querypos_embed.weight.unsqueeze(1),
                                                  patchpos_embed=self.patchpos_embed)

        token_prediction = self.token_predictor(outs).permute(1, 0, 2)

        action_map_prob = F.softmax(action_map, -1)
        token_prediction_prob = torch.sigmoid(token_prediction)

        z = torch.cat([token_prediction_prob, action_map_prob * (1 - token_prediction_prob)], dim=-1)
        z = torch.log(z + eps)

        aggr_action_map = action_map
        aggr_z = z

        if self.training == False:
            aggr_z = F.softmax(aggr_z, -1)

        # cat the memory feature
        cat_memory = torch.cat([self.termination_feature.repeat(1, memory.shape[1], 1), memory], dim=0)

        # for explanation
        target_scanpath_index = torch.argmax(batch["target_scanpath"], dim=-1, keepdim=True).repeat(1, 1, cat_memory.shape[-1]).unsqueeze( 0)
        aggr_fixation_feature = torch.gather(cat_memory.unsqueeze(2).repeat(1, 1, target_scanpath_index.shape[2], 1),
                                             dim=0, index=target_scanpath_index).squeeze(0)

        # encoder_hidden_states = self.hidden_state_transform(
        #     torch.cat([self.image_emb + memory, self.fixation_emb + aggr_fixation_feature], dim=0).permute(1, 0, 2))

        encoder_hidden_state_list = []
        for idx in range(aggr_fixation_feature.shape[0]):
            transform_feature = self.hidden_state_transform(
                torch.stack([self.fixation_emb + aggr_fixation_feature[idx]])).permute(1, 0, 2)
            transform_feature = torch.cat([transform_feature,
                                           (self.task_emb + task[idx].unsqueeze(0)).unsqueeze(1).repeat(transform_feature.shape[0], 1, 1)], dim=1)
            encoder_hidden_state_list.append(transform_feature)
        encoder_hidden_states = torch.stack(encoder_hidden_state_list)
        encoder_hidden_states = encoder_hidden_states.view(-1, encoder_hidden_states.shape[-2],
                                                           encoder_hidden_states.shape[-1])

        # [batch, C, Feature]
        explanation = batch["explanation"]
        dec_input = {
            "input_ids": explanation.input_ids,
            "labels": explanation.input_ids,
            "attention_mask": explanation.attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "output_hidden_states": True
            # "reduction": "none"
        }
        lm_dec_outputs = self.blip_model(**dec_input)


        # predict the scanpath first
        prediction = {}
        # [N, T, A] A = H * W + 1
        prediction['all_actions_prob'] = aggr_z
        # [N, T]
        prediction['log_normal_mu'] = t_log_normal_mu.permute(1, 0, 2).squeeze(-1)
        # [N, T]
        prediction['log_normal_sigma2'] = t_log_normal_sigma2.permute(1, 0, 2).squeeze(-1)
        # [N, T, H, W]
        prediction["action_map"] = aggr_action_map.view(-1, self.max_len, self.spatial_dim[0], self.spatial_dim[1])
        # CausalLMOutputWithCrossAttention
        prediction["dec_outputs"] = lm_dec_outputs

        # sampling results
        scanpath_prediction_list = []
        scanpath_length_list = []
        durations_list = []
        selected_actions_probs_list = []
        selected_actions_list = []
        action_masks_list = []
        duration_masks_list = []
        generated_ids_list = []
        dec_output_loss_list = []
        generated_idx_logit_list = []
        # alignment lst
        resnet_visual_similarity_list = []
        visual_similarity_list = []
        language_similarity_list = []
        multimodal_similarity_list = []
        explanation_similarity_mask_list = []
        for idx in range(sample_num):
            sampling_prediction = self.sampling.random_sample(prediction)
            scanpath_prediction, action_masks, duration_masks = self.sampling.generate_scanpath(
                src, sampling_prediction)
            scanpath_prediction_list.append(scanpath_prediction)
            scanpath_length_list.append(sampling_prediction["scanpath_length"])
            durations_list.append(sampling_prediction["durations"])
            selected_actions_probs_list.append(sampling_prediction["selected_actions_probs"])
            selected_actions_list.append(sampling_prediction["selected_actions"])
            action_masks_list.append(action_masks)
            duration_masks_list.append(duration_masks)

            explanation_similarity_mask = duration_masks.new_ones(duration_masks.shape[0], duration_masks.shape[1], duration_masks.shape[1])
            for index in range(duration_masks.shape[0]):
                explanation_similarity_mask[index, duration_masks[index]==0] = 0
                explanation_similarity_mask[index, :, duration_masks[index] == 0] = 0
            explanation_similarity_mask_list.append(explanation_similarity_mask)

            # visual feature gt alignment from resnet pretrain feature
            visual_feature_resnet = torch.cat([src.new_zeros((src.shape[0], 1, src.shape[-1])), src], dim=1).permute(1, 0, 2)
            selected_action_index = (sampling_prediction["selected_actions"] * duration_masks). \
                unsqueeze(-1).repeat(1, 1, visual_feature_resnet.shape[-1]).unsqueeze(0).long()
            fixated_visual_feature_resnet = torch.gather(
                visual_feature_resnet.unsqueeze(2).repeat(1, 1, selected_action_index.shape[2], 1), dim=0,
                index=selected_action_index).squeeze(0)
            fixated_visual_feature_resnet = fixated_visual_feature_resnet / (
                    torch.norm(fixated_visual_feature_resnet, p=2, dim=2, keepdim=True) + eps)
            resnet_visual_similarity = torch.matmul(fixated_visual_feature_resnet, fixated_visual_feature_resnet.transpose(1, 2))
            resnet_visual_similarity_list.append(resnet_visual_similarity)

            # captioning generation
            # cat the memory feature
            cat_memory = torch.cat([self.termination_feature.repeat(1, memory.shape[1], 1), memory], dim=0)

            # for explanation
            selected_action_index = (sampling_prediction["selected_actions"] * duration_masks).\
                unsqueeze(-1).repeat(1, 1, cat_memory.shape[-1]).unsqueeze(0).long()
            aggr_fixation_feature = torch.gather(
                cat_memory.unsqueeze(2).repeat(1, 1, selected_action_index.shape[2], 1),
                dim=0, index=selected_action_index).squeeze(0)

            # visual project feature for alignment
            aggr_fixation_projection_feature = self.visual_projection(aggr_fixation_feature)
            aggr_fixation_projection_feature = aggr_fixation_projection_feature / (
                        torch.norm(aggr_fixation_projection_feature, p=2, dim=2, keepdim=True) + eps)
            visual_similarity = torch.matmul(aggr_fixation_projection_feature,
                                             aggr_fixation_projection_feature.transpose(1, 2))
            visual_similarity_list.append(visual_similarity)

            # for encoder hidden
            encoder_hidden_state_list = []
            for idx in range(aggr_fixation_feature.shape[0]):
                transform_feature = self.hidden_state_transform(
                    torch.stack([self.fixation_emb + aggr_fixation_feature[idx]])).permute(1, 0, 2)
                transform_feature = torch.cat([transform_feature,
                                               (self.task_emb + task[idx].unsqueeze(0)).unsqueeze(1).repeat(
                                                   transform_feature.shape[0], 1, 1)], dim=1)
                encoder_hidden_state_list.append(transform_feature)
            encoder_hidden_states = torch.stack(encoder_hidden_state_list)
            encoder_hidden_states = encoder_hidden_states.view(-1, encoder_hidden_states.shape[-2],
                                                               encoder_hidden_states.shape[-1])

            # [batch, C, Feature]
            dec_input = {
                "encoder_hidden_states": encoder_hidden_states,
            }
            # generation_kwargs = {
            #     "max_length": self.opt.max_length,
            #     "num_beams": self.opt.num_beams,
            #     "num_return_sequences": self.opt.num_return_sequences,
            #     "early_stopping": True,
            #     "pad_token_id": self.tokenizer.eos_token_id
            # }
            if scst:
                generation_kwargs = {
                    "do_sample": True,
                    # "top_k": 50,
                    "top_p": 0.95,
                    "max_length": self.args.max_generation_length,
                    "num_return_sequences": 1,
                }
            else:
                decoder_input_ids = self.blip_tokenizer(["[CLS] there" for _ in range(encoder_hidden_states.shape[0])], return_tensors="pt", add_special_tokens=False)
                decoder_input_ids.to(src.device)
                dec_input = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "input_ids": decoder_input_ids.input_ids,
                    "attention_mask": decoder_input_ids.attention_mask,
                }
                generation_kwargs = {
                    "max_length": self.args.max_generation_length,
                    "num_beams": self.args.num_explanation_beams,
                    "num_return_sequences": 1,
                    "early_stopping": True,
                }
            kwargs = {**dec_input, **generation_kwargs}
            generated_ids = self.blip_model.generate(**kwargs)


            if scst or detail:
                # with grad version, we use the generated ids to get the output logit and keep the gradient track
                dec_input = {
                    "input_ids": generated_ids,
                    "labels": generated_ids,
                    "attention_mask": generated_ids != 0,
                    "encoder_hidden_states": encoder_hidden_states,
                    "output_hidden_states": True,
                    "reduction": "none"
                }
                dec_outputs = self.blip_model(**dec_input)
                dec_output_loss_list.append(dec_outputs.loss.view(src.shape[0], -1))

                logsoftmax_logits = F.log_softmax(dec_outputs.logits, -1)
                generated_idx_logit = torch.gather(logsoftmax_logits, index=generated_ids.unsqueeze(-1), dim=-1).squeeze(-1)
                generated_idx_logit_list.append(generated_idx_logit.view(src.shape[0], self.args.max_length, -1))

                # Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer)
                cls_language_feature = dec_outputs.hidden_states[-1][:, 0]

                # language project feature for alignment
                cls_language_feature_projection = self.language_projection(cls_language_feature)
                cls_language_feature_projection = cls_language_feature_projection.view(src.shape[0], -1,
                                                                                       cls_language_feature_projection.shape[-1])
                cls_language_feature_projection = cls_language_feature_projection / (
                        torch.norm(cls_language_feature_projection, p=2, dim=2, keepdim=True) + eps)
                language_similarity = torch.matmul(cls_language_feature_projection,
                                                   cls_language_feature_projection.transpose(1, 2))

                multimodal_similarity = torch.matmul(aggr_fixation_projection_feature,
                                                     cls_language_feature_projection.transpose(1, 2))

                language_similarity_list.append(language_similarity)
                multimodal_similarity_list.append(multimodal_similarity)

            generated_ids = generated_ids * duration_masks.view(-1, 1).long()
            generated_ids_list.append(generated_ids.view(src.shape[0], self.args.max_length, -1))

        scanpath_prediction = torch.stack(scanpath_prediction_list, dim=1)
        sampling_prediction["scanpath_length"] = torch.stack(scanpath_length_list, dim=1)
        sampling_prediction["durations"] = torch.stack(durations_list, dim=1)
        sampling_prediction["selected_actions_probs"] = torch.stack(selected_actions_probs_list, dim=1)
        sampling_prediction["selected_actions"] = torch.stack(selected_actions_list, dim=1)
        action_masks = torch.stack(action_masks_list, dim=1)
        duration_masks = torch.stack(duration_masks_list, dim=1)
        generated_ids = torch.stack(generated_ids_list, dim=2)
        resnet_visual_similarity = torch.stack(resnet_visual_similarity_list, dim=1)
        visual_similarity = torch.stack(visual_similarity_list, dim=1)
        explanation_similarity_mask = torch.stack(explanation_similarity_mask_list, dim=1)

        # [N, R, H, W]
        prediction["resnet_visual_similarity"] = resnet_visual_similarity
        # [N, R, H, W]
        prediction["visual_similarity"] = visual_similarity
        # [N, R, H, W]
        prediction["explanation_similarity_mask"] = explanation_similarity_mask
        if scst or detail:
            dec_output_loss = torch.stack(dec_output_loss_list, dim=2)
            language_similarity = torch.stack(language_similarity_list, dim=1)
            multimodal_similarity = torch.stack(multimodal_similarity_list, dim=1)
            generated_idx_logit = torch.stack(generated_idx_logit_list, dim=2)
            # [N, R, H, W]
            prediction["language_similarity"] = language_similarity
            # [N, R, H, W]
            prediction["multimodal_similarity"] = multimodal_similarity
            return prediction, scanpath_prediction, generated_ids, generated_idx_logit, dec_output_loss, sampling_prediction, action_masks, duration_masks

        else:
            return prediction, scanpath_prediction, generated_ids, sampling_prediction, action_masks, duration_masks