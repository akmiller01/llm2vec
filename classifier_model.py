import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class PreEmbeddedSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config['num_labels']
        self.problem_type = None
        self.weights = config.get('weights', None)

        self.pre_classifier = nn.Linear(config['dim'], config['dim'])
        self.classifier = nn.Linear(config['dim'], config['num_labels'])
        self.dropout = nn.Dropout(config['seq_classif_dropout'])

    def forward(self, idx, targets=None):
        pooled_output = idx  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if targets is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (targets.dtype == torch.long or targets.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), targets.squeeze())
                else:
                    loss = loss_fct(logits, targets)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=self.weights)
                loss = loss_fct(logits.view(-1, self.num_labels), targets.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(weight=self.weights)
                loss = loss_fct(logits, targets)

        return logits, loss

