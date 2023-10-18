"""
@Date  : 2023/2/1
@Time  : 17:54
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel


class BertSoftmax(torch.nn.Module):
    def __init__(self, device: torch.device, backbone: str, hidden_dim: int, num_classes: int):
        super(BertSoftmax, self).__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.linear = torch.nn.Linear(hidden_dim, num_classes)
        self.to(device)

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor, seq_labels: torch.tensor):
        featuers = self.backbone(input_ids, attention_mask).last_hidden_state
        logits = self.linear(featuers)

        if not self.training:
            return logits.argmax(dim=-1)
        label_mask = seq_labels != -100
        loss = F.cross_entropy(logits[label_mask], seq_labels[label_mask])
        return loss
