import torch
import torch.nn as nn
from src.config import NOMBRE_CANAUX, TAILLE_IMAGE

class CNNSimple(nn.Module):

    def __init__(self):
        super().__init__()
        self.couches_convolution = nn.Sequential(nn.Conv2d(NOMBRE_CANAUX, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        with torch.no_grad():
            dummy_input = torch.zeros(1, NOMBRE_CANAUX, TAILLE_IMAGE, TAILLE_IMAGE)
            dummy_output = self.couches_convolution(dummy_input)
            self.taille_flatten = dummy_output.view(1, -1).size(1)
        self.couches_classification = nn.Sequential(nn.Flatten(), nn.Linear(self.taille_flatten, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, tenseur_entree):
        caracteristiques = self.couches_convolution(tenseur_entree)
        sortie = self.couches_classification(caracteristiques)
        return sortie
