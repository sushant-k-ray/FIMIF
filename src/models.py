import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from configs import cfg

class FIM(nn.Module):
    def __init__(self, dim_a, dim_b, epsilon, device):
        super().__init__()
        self.epsilon = epsilon
        
        self.residual = nn.Linear(dim_a, dim_b).to(device)
        
        self.w1 = nn.Linear(dim_a, dim_b).to(device)
        self.w2 = nn.Linear(dim_b, dim_b, bias=False).to(device)
        torch.nn.init.eye_(self.w2.weight)
        self.w3 = nn.Linear(dim_b, dim_b).to(device)
        
        self.relu = nn.ReLU()
        
        self.w4 = nn.Linear(dim_a, dim_b).to(device)
        self.w5 = nn.Linear(dim_b, dim_b, bias=False).to(device)
        torch.nn.init.eye_(self.w5.weight)
        self.w6 = nn.Linear(dim_b, dim_b).to(device)

    def mult_layer(self, x, A, W, B):
        return B(torch.exp(W(torch.log(self.relu(A(x)) + self.epsilon))))

    def forward(self, x):
        return self.residual(x) +\
               self.mult_layer(x, self.w1, self.w2, self.w3) *\
               self.mult_layer(x, self.w4, self.w5, self.w6)

class resModule(nn.Module):
    def __init__(self, dim_a, dim_b, device):
        super().__init__()
        self.w1 = nn.Linear(dim_a, dim_b).to(device)
        self.w2 = nn.Linear(dim_a, dim_b).to(device)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.w1(x)) + self.w2(x)

    def getl1(self):
        return self.w1.weight.abs().sum() + self.w2.weight.abs().sum()

class prideMMClassifier(pl.LightningModule):
    def __init__(self, cfg, loss_weights = None):
        super().__init__()
        self.lr = cfg.lr
        self.l1reg = cfg.l1reg
        self.l2reg = cfg.l2reg
        
        self.acc = torchmetrics.Accuracy(task='multiclass',\
                                         num_classes = cfg.num_classes)

        self.auroc = torchmetrics.AUROC(task='multiclass',\
                                        num_classes = cfg.num_classes)
                                        
        self.f1 = torchmetrics.F1Score(task='multiclass',\
                                       num_classes = cfg.num_classes,\
                                       average='macro')

        self.dropout = nn.Dropout(p=0.2)
        self.residual = resModule(cfg.embed_dim, cfg.hidden_dim, cfg.device)
        self.mult = FIM(cfg.hidden_dim, cfg.hidden_dim, cfg.device)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim, device=cfg.device)
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes)\
                            .to(cfg.device)
        self.loss = torch.nn.CrossEntropyLoss(weight = loss_weights,\
                                              reduction='mean').to(cfg.device)
        
    def forward(self, batch):
        data = self.dropout(batch)
        data = self.residual(data)
        data = self.layer_norm(self.mult(data))
        logits = self.classifier(data)
        return logits

    def common_step(self, batch):
        data, labels = batch
        labels = labels.long().reshape(-1)
        logits = self.forward(data)
        
        output = {}
        output['loss'] = self.loss(logits, labels) +\
                         self.l1reg * self.residual.getl1()

        preds = F.softmax(logits, dim = 1)
        output['accuracy'] = self.acc(preds, labels)
        output['auroc'] = self.auroc(preds, labels)
        output['f1'] = self.f1(preds, labels)

        return output
    
    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)

        total_loss = output['loss']

        self.log('train/total_loss', total_loss)
        self.log('train/loss', output['loss'])
        self.log('train/accuracy', output['accuracy'])
        self.log(f'train/auroc', output['auroc'],\
                 on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)
        total_loss = output['loss']

        self.log(f'val/total_loss', total_loss)
        self.log(f'val/loss', output['loss'])
        self.log(f'val/accuracy', output['accuracy'],\
                 on_step=False, on_epoch=True, prog_bar=True)
                 
        self.log(f'val/auroc', output['auroc'],\
                 on_step=False, on_epoch=True, prog_bar=True)

        self.log(f'val/f1', output['f1'],\
                 on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log(f'test/accuracy', output['accuracy'])
        self.log(f'test/auroc', output['auroc'])
        self.log(f'test/f1', output['f1'])

        return output

    def on_train_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()
        
    def on_validation_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()

    def on_test_epoch_end(self):
        self.acc.reset()
        self.auroc.reset()
        self.f1.reset()
        
    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters()
                              if p.requires_grad]}
        ]

        optimizer = torch.optim.AdamW(param_dicts,\
                                      lr=self.lr,\
                                      weight_decay=self.l2reg)
        return optimizer
