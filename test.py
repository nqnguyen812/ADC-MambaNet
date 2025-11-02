import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.adc_mambanet import ADC_MambaNet
from loss import DiceLoss
from metrics import dice_score, iou_score, precision_score, recall_score
from dataloaders import ISICLoader


class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = DiceLoss()  

    def forward(self, x):
        return self.model(x)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                               factor=0.5, patience=5, verbose=True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers
    

model = ADC_MambaNet.cuda()
model = model.eval()

CHECKPOINT_PATH = ''
DATA_PATH = ''
test_dataset = ISICLoader(type='test', data_path=DATA_PATH, transform=False)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)
# Prediction
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)