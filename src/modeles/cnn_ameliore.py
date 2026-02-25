import torch.nn as nn
from src.config import NOMBRE_CANAUX

class CNNAmeliore(nn.Module):

    def __init__(self):
        super().__init__()
        self.couches_convolution = nn.Sequential(nn.Conv2d(NOMBRE_CANAUX, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pooling_global = nn.AdaptiveAvgPool2d(1)
        self.couches_classification = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, tenseur_entree):
        caracteristiques = self.couches_convolution(tenseur_entree)
        caracteristiques = self.pooling_global(caracteristiques)
        sortie = self.couches_classification(caracteristiques)
        return sortie
