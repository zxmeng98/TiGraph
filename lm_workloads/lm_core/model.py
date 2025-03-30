import time
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from lm_workloads.lm_core.utils import init_random_state


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink='', gnn_input_dim=128, emb=None, pred=None):
        super().__init__(model.config)
        self.bert_encoder = model
        self.emb, self.pred = emb, pred
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        self.gnn_input_dim = gnn_input_dim
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.feat_to_gnn_input_layer = nn.Linear(
            model.config.hidden_size, int(self.gnn_input_dim), bias=cla_bias)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)   

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None,
                node_id=None,
                exit_layer=None,
                ):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True,
                                    exit_layer=exit_layer,
                                    )
        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy() # batch_nodes are not in order
        emb_save = outputs['hidden_states'][-1]
        cls_token_emb_save = emb_save.permute(1, 0, 2)[0].detach()
        # outputs[0]=last hidden state
        emb = self.dropout(emb_save)
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        # cls_token_emb_save = self.feat_to_gnn_input_layer(cls_token_emb_save)
        self.emb[batch_nodes] = cls_token_emb_save.detach().cpu().numpy().astype(np.float16)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        self.pred[batch_nodes] = logits.detach().cpu().numpy().astype(np.float16)

        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        cls_token_emb_save = self.bert_classifier.feat_to_gnn_input_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        # t0 = time.time()
        batch_nodes = node_id.cpu().numpy() # batch_nodes are in order from samll to large
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)
        # print(f"Time to save to disk: {time.time() - t0}")

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
