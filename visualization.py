import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloaders import ISICLoader
from models.adc_mambanet import ADC_MambaNet

def visualize_prediction_VT(model, dataset):
  model.eval()
  plt.figure(figsize=(6, 2*5), layout='compressed')
  x1, y1 = dataset[20]
  x2, y2 = dataset[21]
  x3, y3 = dataset[22]
  x4, y4 = dataset[23]
  x5, y5 = dataset[24]
  x = torch.cat((x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0),x4.unsqueeze(0),x5.unsqueeze(0))).cuda()
  y = torch.cat((y1.unsqueeze(0),y2.unsqueeze(0),y3.unsqueeze(0),y4.unsqueeze(0),y5.unsqueeze(0))).cuda()


  y_pred = model(x).data.squeeze()
  y_pred[y_pred>0.5] = 1
  y_pred[y_pred<=0.5] = 0
  y_pred = y_pred.cpu().numpy()
  for i in range(5):
            xa = x[i]
            xa = xa.permute(1, 2, 0).cpu().numpy()
            ya = y[i]
            ya = ya.permute(1, 2, 0).cpu().numpy()
            plt.subplot(5, 3, 3*i + 1)
            plt.title(f"Image {i+1}")
            plt.imshow(xa)
            plt.axis('off')

            plt.subplot(5, 3, 3*i + 2)
            plt.title(f"Ground truth {i+1}")
            plt.imshow(ya, cmap='gray')
            plt.axis('off')

            plt.subplot(5, 3, 3*i + 3)
            plt.title(f"SK {i+1}")
            plt.imshow(y_pred[i], cmap='gray')
            plt.axis('off')

device = 'cuda'
model = ADC_MambaNet()
model = model.to(device)
DATA_PATH = ''
test_dataset = ISICLoader(type='test', data_path=DATA_PATH, transform=False)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)
visualize_prediction_VT(test_dataset,idx = [])