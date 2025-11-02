import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from loss import dice_bnce_loss
from metrics import dice_score, iou_score, precision_score, recall_score
from models.adc_mambanet import ADC_MambaNet
from dataloaders import ISICLoader
import os


class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = dice_bnce_loss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss = self.criterion(y_pred, y_true)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        precision = precision_score(y_pred, y_true)
        recall = recall_score(y_pred, y_true)
        return loss, dice, iou, precision, recall

    def training_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall = self._step(batch)
        metrics = {
            "loss": loss,
            "train_dice": dice,
            "train_iou": iou,
            "train_precision": precision,
            "train_recall": recall,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall = self._step(batch)
        metrics = {
            "val_loss": loss,
            "val_dice": dice,
            "val_iou": iou,
            "val_precision": precision,
            "val_recall": recall,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice, iou, precision, recall = self._step(batch)
        metrics = {
            "test_loss": loss,
            "test_dice": dice,
            "test_iou": iou,
            "test_precision": precision,
            "test_recall": recall,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                               factor=0.7, patience=5, verbose=True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers
    


model = ADC_MambaNet.cuda()
DATA_PATH = ''

train_dataset = ISICLoader(type='train', data_path=DATA_PATH, transform=True)
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True)

val_dataset = ISICLoader(type='test', data_path=DATA_PATH, transform=False)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

# Training config
os.makedirs('/content/weights', exist_ok = True)
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint('/content/weights', filename="ckpt{val_dice:0.4f}",
                                                            monitor="val_dice", mode = "max", save_top_k =1,
                                                            verbose=True, save_weights_only=True,
                                                            auto_insert_metric_name=False)
progress_bar = pl.callbacks.TQDMProgressBar()
PARAMS = {"benchmark": True, "enable_progress_bar" : True,"logger":True,
          "callbacks" : [check_point, progress_bar],
          "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":100,
          "precision":16,
          }
trainer = pl.Trainer(**PARAMS)
segmentor = Segmentor(model=model)


trainer.fit(segmentor, train_loader, val_loader)