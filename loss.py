import torch
from torch import nn
import math

class Sample_Consistency(nn.Module):
    def __init__(self, batch_size=256, temperature=1):
        super(Sample_Consistency, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class Community_Versatility(nn.Module):
    def __init__(self, batch_size=256, temperature=1, gamma=0.7):
        super(Community_Versatility, self).__init__()
        self.batch_size = batch_size*2
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.gamma = gamma
        self.margin = gamma**2

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        scores = torch.matmul(z_i, z_j.T) / self.temperature

        diagonal = scores.diag().view(scores.size(0), 1)
        diagonal[diagonal >= self.gamma] = self.gamma
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
        loss = cost_s_max.mean() + cost_im_max.mean()
        return loss
    
def Community_Commonality(c_i, c_j, com_lambda=0.02):
    p_i = c_i.sum(0).view(-1)
    p_i /= p_i.sum()
    p_j = c_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i+ 1e-8)).sum()
    ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j+ 1e-8)).sum()
    ne_loss = (ne_i + ne_j)*1/2

    ne_i_onehot = -torch.sum(c_i * torch.log(c_i + 1e-8), dim=1).mean()
    ne_j_onehot = -torch.sum(c_j * torch.log(c_j + 1e-8), dim=1).mean()
    onehot_loss = (ne_i_onehot + ne_j_onehot) * com_lambda / 2
    return ne_loss + onehot_loss