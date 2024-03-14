import torch
import numpy as np

class Sampling():
    def __init__(self, args):
        super(Sampling, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.min_length = args.min_length
        self.map_width = args.im_w
        self.map_height = args.im_h
        self.width = args.width
        self.height = args.height
        self.x_granularity = float(self.width / self.map_width)
        self.y_granularity = float(self.height / self.map_height)

    def random_sample(self, model_prediction):
        log_normal_mu = model_prediction["log_normal_mu"]
        log_normal_sigma2 = model_prediction["log_normal_sigma2"]
        all_actions_prob = model_prediction["all_actions_prob"]

        # sampling stage
        batch = all_actions_prob.shape[0]
        probs = all_actions_prob.data.clone()
        probs[:, :self.min_length, 0] = 0
        dist = torch.distributions.categorical.Categorical(probs=probs)
        selected_specific_actions = dist.sample()
        selected_actions_probs = \
            torch.gather(all_actions_prob, dim=2, index=selected_specific_actions.unsqueeze(-1)).squeeze(-1)

        random_rand = torch.randn(log_normal_mu.shape).to(log_normal_mu.get_device())
        duration_samples = torch.exp(random_rand * log_normal_sigma2 + log_normal_mu)

        scanpath_length = all_actions_prob.new_zeros(batch)
        for index in range(self.max_length):
            scanpath_length[torch.logical_and(
                scanpath_length == 0, selected_specific_actions[:, index] == 0)] = index
        scanpath_length[scanpath_length == 0] = self.max_length
        scanpath_length = scanpath_length.unsqueeze(-1)

        prediction = {}
        # [N, 1]
        prediction["scanpath_length"] = scanpath_length
        # [N, T]
        prediction["durations"] = duration_samples
        # [N, T]
        prediction["selected_actions_probs"] = selected_actions_probs
        # [N, T]
        prediction["selected_actions"] = selected_specific_actions

        return prediction

    def generate_scanpath(self, images, sampling_prediction):
        prob_sample_actions = sampling_prediction["selected_actions_probs"]
        durations = sampling_prediction["durations"]
        sample_actions  = sampling_prediction["selected_actions"]
        # computer the logprob for action and duration
        action_masks = images.new_zeros(prob_sample_actions.shape)
        duration_masks = images.new_zeros(prob_sample_actions.shape)
        scanpath_predictions = images.new_zeros((images.shape[0], self.max_length, 3)) - 1
        t = durations.data.clone()
        N = images.shape[0]
        for index in range(N):
            sample_action = sample_actions[index].cpu().numpy()
            drts = t[index].cpu().numpy()
            for order in range(sample_action.shape[0]):
                if sample_action[order] == 0:
                    action_masks[index, order] = 1
                    break
                else:
                    image_index = sample_action[order] - 1
                    map_pos_x = image_index % self.map_width
                    map_pos_y = image_index // self.map_width
                    pos_x = map_pos_x * self.x_granularity + self.x_granularity / 2
                    pos_y = map_pos_y * self.y_granularity + self.y_granularity / 2
                    drt = drts[order] * 1000
                    action_masks[index, order] = 1
                    duration_masks[index, order] = 1
                    scanpath_predictions[index, order, 0] = pos_x
                    scanpath_predictions[index, order, 1] = pos_y
                    scanpath_predictions[index, order, 2] = drt


        return scanpath_predictions, action_masks, duration_masks
