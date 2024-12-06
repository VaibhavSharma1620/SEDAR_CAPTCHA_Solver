import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class CAptcha_CNN(nn.Module):
    def __init__(self,input_dim,hidden_dim_1,hidden_dim_2,hidden_dim_3, output_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim_1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(hidden_dim_2, hidden_dim_3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(hidden_dim_3 * 3 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
        )

        self.softmax_layers = nn.ModuleList([nn.Linear(512, 62) for _ in range(output_dim)])

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        outs = []
        for layer in self.softmax_layers:
            outs.append(F.softmax(layer(x),dim=-1))

        return torch.stack(outs, dim=1)

class Captcha_inf():
    def __init__(self,device="cpu"):
        self.device=device
        self.NUM_OF_LETTERS=5
        self.alphabets="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.model =CAptcha_CNN(1, 32, 48, 64, self.NUM_OF_LETTERS)
        self.model.load_state_dict(torch.load("model/model_smooth.pth"))
        self.model.to(self.device)

    def Captcha_sol(self,IMG_PATH):
        img=cv2.imread(IMG_PATH)
        img=img[471:521,580:781]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (int(201/2), int(50/2)), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img=np.expand_dims(np.transpose(img, axes=( 2,0,1)),axis=0)
        a=self.model(torch.tensor(img,dtype=torch.float).to(self.device)/255)[0]
        a=np.array(a.cpu().detach())
        out=""
        for i in range(5):
            out+=self.alphabets[np.array(a[i]).argmax()]
        return out
    

####### SAMPLE INFERENCE
# INF=Captcha_inf()
# INF.Captcha_sol("ss2/1000.png")
