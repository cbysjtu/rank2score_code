import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_mask(mos, dataset_name):
    return mos.ne(0) if dataset_name == "T23D-CompBench" else torch.ones_like(mos, dtype=torch.bool)


class MSE_Learning(nn.Module):
    """Stage-II MOS regression loss (Eq. 19)."""

    def __init__(self, device=None):
        super().__init__()

    def forward(self, quality_score, mos, feature_fusion=None, dataset_name="T23D-CompBench", **_):
        valid = _valid_mask(mos, dataset_name)
        if not valid.any():
            return quality_score.sum() * 0.0
        return F.mse_loss(quality_score[valid], mos[valid])


class Rank_Learning(nn.Module):
    """Stage-I objective L_rank + lambda * L_cons (paper Eqs. 9-11)."""

    def __init__(self, device=None, margin=0.5, temperature=2.0, contrastive_weight=1.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

    def forward(self, quality_score, mos, feature_fusion, dataset_name="T23D-CompBench",
                score_threshold=0.0, consistency_threshold=0.0, **_):
        rank_loss = self.compute_rank(
            quality_score, mos, dataset_name, score_threshold, consistency_threshold
        )
        contrastive_loss = self.compute_contrastive(mos, feature_fusion, dataset_name)
        return rank_loss + self.contrastive_weight * contrastive_loss

    def _pair_masks(self, mos, dataset_name, score_threshold, consistency_threshold):
        valid = _valid_mask(mos, dataset_name)
        diff = mos[:, None, :] - mos[None, :, :]
        pair_valid = valid[:, None, :] & valid[None, :, :]
        signs = torch.sign(diff) * pair_valid
        non_ties = signs.ne(0).sum(dim=-1)
        consistency = signs.sum(dim=-1).abs() / non_ties.clamp_min(1)
        curriculum = consistency.ge(consistency_threshold)
        if score_threshold > 0:
            curriculum &= (diff.abs().ge(score_threshold) & pair_valid).any(dim=-1)
        upper = torch.triu(torch.ones_like(curriculum, dtype=torch.bool), diagonal=1)
        return diff, pair_valid, curriculum & upper

    def compute_rank(self, quality_score, mos, dataset_name,
                     score_threshold=0.0, consistency_threshold=0.0):
        diff_gt, valid_dims, selected_pairs = self._pair_masks(
            mos, dataset_name, score_threshold, consistency_threshold
        )
        pred_diff = quality_score[:, None, :] - quality_score[None, :, :]
        eta = torch.sign(diff_gt)
        per_dimension = F.relu(-eta * pred_diff + self.margin)
        usable = valid_dims & eta.ne(0) & selected_pairs.unsqueeze(-1)
        if not usable.any():
            return quality_score.sum() * 0.0
        return per_dimension[usable].mean()

    def compute_contrastive(self, mos, feature_fusion, dataset_name):
        """Ordered label-distance contrastive regression with negative L2 similarity."""
        valid = _valid_mask(mos, dataset_name)
        _, num_dimensions = mos.shape
        losses = []
        for d in range(num_dimensions):
            indices = torch.where(valid[:, d])[0]
            if indices.numel() < 3:
                continue
            labels = mos[indices, d]
            features = feature_fusion[indices, d]
            label_dist = (labels[:, None] - labels[None, :]).abs()
            similarity = -torch.cdist(features, features, p=2)
            n = indices.numel()
            for anchor in range(n):
                non_anchor = torch.arange(n, device=mos.device) != anchor
                for positive in torch.where(non_anchor)[0]:
                    negatives = label_dist[anchor].ge(label_dist[anchor, positive]) & non_anchor
                    candidates = torch.where(negatives)[0]
                    if candidates.numel() == 0:
                        continue
                    log_num = similarity[anchor, positive] / self.temperature
                    log_den = torch.logsumexp(
                        similarity[anchor, candidates] / self.temperature, dim=0
                    )
                    losses.append(-(log_num - log_den))
        if not losses:
            return feature_fusion.sum() * 0.0
        return torch.stack(losses).mean()
