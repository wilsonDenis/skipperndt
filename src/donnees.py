import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from src.config import TAILLE_IMAGE, TAILLE_LOT, DOSSIER_DONNEES, DOSSIER_RESULTATS, NOMS_CLASSES
transformation_entrainement = transforms.Compose([transforms.ToTensor(), transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15)])
transformation_test = transforms.Compose([transforms.ToTensor(), transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE))])

class DatasetDroneNpz(Dataset):

    def __init__(self, chemins_fichiers, etiquettes, transform=None):
        self.chemins_fichiers = chemins_fichiers
        self.etiquettes = etiquettes
        self.transform = transform

    def __len__(self):
        return len(self.chemins_fichiers)

    def __getitem__(self, idx):
        chemin = self.chemins_fichiers[idx]
        avec_fichiers = np.load(chemin)
        if 'data' in avec_fichiers.files:
            matrice = avec_fichiers['data']
        else:
            matrice = avec_fichiers[avec_fichiers.files[0]]
        if matrice.ndim == 2:
            matrice = np.expand_dims(matrice, axis=-1)
        matrice = matrice.astype(np.float32)
        mat_min, mat_max = (matrice.min(), matrice.max())
        if mat_max > mat_min:
            matrice = (matrice - mat_min) / (mat_max - mat_min)
        if self.transform:
            try:
                matrice = self.transform(matrice)
            except Exception as e:
                matrice_tensor = torch.from_numpy(matrice).permute(2, 0, 1)
                matrice = transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE))(matrice_tensor)
        etiquette = torch.tensor(self.etiquettes[idx], dtype=torch.long)
        return (matrice, etiquette)

def charger_donnees():
    chemins_tous = []
    etiquettes_toutes = []
    print(f'Recherche de fichiers dans {DOSSIER_DONNEES}...')
    for idx_classe, nom_classe in enumerate(NOMS_CLASSES):
        dossier_classe = os.path.join(DOSSIER_DONNEES, nom_classe)
        fichiers_classe = glob.glob(os.path.join(dossier_classe, '*.npz'))
        chemins_tous.extend(fichiers_classe)
        etiquettes_toutes.extend([idx_classe] * len(fichiers_classe))
        print(f"   - Classe '{nom_classe}' : {len(fichiers_classe)} fichiers trouvés")
    if not chemins_tous:
        print('AUCUN FICHIER TROUVÉ ! Avez-vous exécuté le script de nettoyage ?')
        return (None, None, None, None, None)
    if len(chemins_tous) < 5:
        print(" Attention: Dataset très petit. Le test et la validation utiliseront les mêmes données d'entraînement pour éviter les erreurs de split.")
        X_train, y_train = (chemins_tous, etiquettes_toutes)
        X_val, y_val = (chemins_tous, etiquettes_toutes)
        X_test, y_test = (chemins_tous, etiquettes_toutes)
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(chemins_tous, etiquettes_toutes, test_size=0.15, stratify=etiquettes_toutes, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)
    print(f'\n Répartition du dataset :')
    print(f'   Entraînement : {len(X_train)} fichiers')
    print(f'   Validation   : {len(X_val)} fichiers')
    print(f'   Test         : {len(X_test)} fichiers')
    dataset_train = DatasetDroneNpz(X_train, y_train, transform=transformation_entrainement)
    dataset_val = DatasetDroneNpz(X_val, y_val, transform=transformation_test)
    dataset_test = DatasetDroneNpz(X_test, y_test, transform=transformation_test)
    chargeur_train = DataLoader(dataset_train, batch_size=TAILLE_LOT, shuffle=True)
    chargeur_val = DataLoader(dataset_val, batch_size=TAILLE_LOT, shuffle=False)
    chargeur_test = DataLoader(dataset_test, batch_size=TAILLE_LOT, shuffle=False)
    return (chargeur_train, chargeur_val, chargeur_test, dataset_train, dataset_test)

def afficher_echantillons(donnees, nombre=9):
    figure, axes = plt.subplots(3, 3, figsize=(8, 8))
    figure.suptitle('Échantillons de radiographies', fontsize=14, fontweight='bold')
    for indice, axe in enumerate(axes.flat):
        if indice >= nombre or indice >= len(donnees):
            break
        image, etiquette = donnees[indice]
        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()
        axe.imshow(image, cmap='gray')
        axe.set_title(NOMS_CLASSES[int(etiquette)], fontsize=10)
        axe.axis('off')
    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/echantillons.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('---Échantillons sauvegardés dans resultats/echantillons.png')

def afficher_distribution(dataset_entrainement, dataset_test):
    etiquettes_train = np.array(dataset_entrainement.etiquettes)
    etiquettes_test = np.array(dataset_test.etiquettes)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure.suptitle('Distribution des classes', fontsize=14, fontweight='bold')
    for indice, (etiquettes, titre) in enumerate([(etiquettes_train, 'Entraînement'), (etiquettes_test, 'Test')]):
        valeurs, comptages = np.unique(etiquettes, return_counts=True)
        couleurs = ['#2ecc71', '#e74c3c']
        barres = axes[indice].bar([NOMS_CLASSES[int(v)] for v in valeurs], comptages, color=couleurs, edgecolor='black', alpha=0.8)
        axes[indice].set_title(titre, fontsize=12)
        axes[indice].set_ylabel("Nombre d'images")
        for barre, comptage in zip(barres, comptages):
            axes[indice].text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 50, str(comptage), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/distribution_classes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('💾 Distribution sauvegardée dans resultats/distribution_classes.png')
