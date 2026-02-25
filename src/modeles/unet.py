import torch
import torch.nn as nn
from src.config import NOMBRE_CANAUX, NOMBRE_CLASSES

class BlocConvolution(nn.Module):

    def __init__(self, canaux_entree, canaux_sortie):
        super().__init__()
        self.bloc = nn.Sequential(nn.Conv2d(canaux_entree, canaux_sortie, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(canaux_sortie), nn.ReLU(inplace=True), nn.Conv2d(canaux_sortie, canaux_sortie, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(canaux_sortie), nn.ReLU(inplace=True))

    def forward(self, tenseur_entree):
        return self.bloc(tenseur_entree)

class Encodeur(nn.Module):

    def __init__(self, canaux_entree, liste_filtres):
        super().__init__()
        self.niveaux = nn.ModuleList()
        self.pools = nn.ModuleList()
        for filtres in liste_filtres:
            self.niveaux.append(BlocConvolution(canaux_entree, filtres))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            canaux_entree = filtres

    def forward(self, tenseur_entree):
        sorties_intermediaires = []
        for niveau, pool in zip(self.niveaux, self.pools):
            tenseur_entree = niveau(tenseur_entree)
            sorties_intermediaires.append(tenseur_entree)
            tenseur_entree = pool(tenseur_entree)
        return (tenseur_entree, sorties_intermediaires)

class Decodeur(nn.Module):

    def __init__(self, liste_filtres):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.niveaux = nn.ModuleList()
        for i in range(len(liste_filtres) - 1):
            filtres_entree = liste_filtres[i]
            filtres_sortie = liste_filtres[i + 1]
            self.upconvs.append(nn.ConvTranspose2d(filtres_entree, filtres_sortie, kernel_size=2, stride=2))
            self.niveaux.append(BlocConvolution(filtres_sortie * 2, filtres_sortie))

    def forward(self, tenseur_entree, sorties_intermediaires):
        for i, (upconv, niveau) in enumerate(zip(self.upconvs, self.niveaux)):
            tenseur_entree = upconv(tenseur_entree)
            skip = sorties_intermediaires[-(i + 1)]
            if tenseur_entree.shape != skip.shape:
                tenseur_entree = nn.functional.interpolate(tenseur_entree, size=skip.shape[2:], mode='bilinear', align_corners=True)
            tenseur_entree = torch.cat([skip, tenseur_entree], dim=1)
            tenseur_entree = niveau(tenseur_entree)
        return tenseur_entree

class UNetClassification(nn.Module):

    def __init__(self):
        super().__init__()
        liste_filtres = [32, 64, 128, 256]
        self.encodeur = Encodeur(NOMBRE_CANAUX, liste_filtres)
        self.bottleneck = BlocConvolution(liste_filtres[-1], liste_filtres[-1] * 2)
        filtres_decodeur = [liste_filtres[-1] * 2] + liste_filtres[::-1]
        self.decodeur = Decodeur(filtres_decodeur)
        self.pooling_global = nn.AdaptiveAvgPool2d(1)
        self.classificateur = nn.Sequential(nn.Flatten(), nn.Linear(liste_filtres[0], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, NOMBRE_CLASSES))

    def forward(self, tenseur_entree):
        caracteristiques, skips = self.encodeur(tenseur_entree)
        caracteristiques = self.bottleneck(caracteristiques)
        caracteristiques = self.decodeur(caracteristiques, skips)
        caracteristiques = self.pooling_global(caracteristiques)
        sortie = self.classificateur(caracteristiques)
        return sortie
