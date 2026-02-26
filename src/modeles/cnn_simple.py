import torch
import torch.nn as nn
from src.config import NOMBRE_CANAUX, TAILLE_IMAGE


class CNNSimple(nn.Module):

    def __init__(self):
        super().__init__()

        self.couches_convolution = nn.Sequential(
            nn.Conv2d(NOMBRE_CANAUX, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, NOMBRE_CANAUX, TAILLE_IMAGE, TAILLE_IMAGE)
            dummy_output = self.couches_convolution(dummy_input)
            self.taille_flatten = dummy_output.view(1, -1).size(1)

        self.features_partagees = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.taille_flatten, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.tete_classification = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

        self.tete_regression = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tenseur_entree):
        caracteristiques = self.couches_convolution(tenseur_entree)
        features = self.features_partagees(caracteristiques)
        sortie_classification = self.tete_classification(features)
        sortie_regression = self.tete_regression(features).squeeze(-1)
        return sortie_classification, sortie_regression
