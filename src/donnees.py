import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from src.config import (
    TAILLE_IMAGE, TAILLE_LOT, DOSSIER_DONNEES, DOSSIER_DONNEES_REELLES,
    FICHIER_CSV, DOSSIER_RESULTATS, NOMS_CLASSES
)

transformation_entrainement = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
])

transformation_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
])


class DatasetMultiTask(Dataset):

    def __init__(self, chemins_fichiers, etiquettes, largeurs, transform=None):
        self.chemins_fichiers = chemins_fichiers
        self.etiquettes = etiquettes
        self.largeurs = largeurs
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
        matrice = np.nan_to_num(matrice, nan=0.0)

        mat_min, mat_max = matrice.min(), matrice.max()
        if mat_max > mat_min:
            matrice = (matrice - mat_min) / (mat_max - mat_min)

        if self.transform:
            try:
                matrice = self.transform(matrice)
            except Exception:
                matrice_tensor = torch.from_numpy(matrice).permute(2, 0, 1)
                matrice = transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE))(matrice_tensor)

        etiquette = torch.tensor(self.etiquettes[idx], dtype=torch.long)
        largeur = torch.tensor(self.largeurs[idx], dtype=torch.float32)

        return matrice, etiquette, largeur


def _resoudre_chemin_fichier(nom_fichier):
    if nom_fichier.startswith('real_data'):
        chemin = os.path.join(DOSSIER_DONNEES_REELLES, nom_fichier)
        if os.path.exists(chemin):
            return chemin
        return None

    chemin_avec = os.path.join(DOSSIER_DONNEES, 'avec_fourreau', nom_fichier)
    if os.path.exists(chemin_avec):
        return chemin_avec

    chemin_sans = os.path.join(DOSSIER_DONNEES, 'sans_fourreau', nom_fichier)
    if os.path.exists(chemin_sans):
        return chemin_sans

    return None


def charger_donnees():
    print(f'Lecture du CSV : {FICHIER_CSV}')
    df = pd.read_csv(FICHIER_CSV, sep=';', keep_default_na=False)

    chemins = []
    etiquettes = []
    largeurs = []
    est_reel = []
    fichiers_manquants = 0

    for _, ligne in df.iterrows():
        nom_fichier = ligne['field_file']
        label = int(ligne['label'])

        width = ligne['width_m']
        if isinstance(width, str) and width == 'N/A':
            width_val = 0.0
        else:
            try:
                width_val = float(width)
            except (ValueError, TypeError):
                width_val = 0.0

        chemin = _resoudre_chemin_fichier(nom_fichier)
        if chemin is None:
            fichiers_manquants += 1
            continue

        chemins.append(chemin)
        etiquettes.append(label)
        largeurs.append(width_val)
        est_reel.append(nom_fichier.startswith('real_data'))

    print(f'  Fichiers trouves : {len(chemins)} / {len(df)}')
    if fichiers_manquants > 0:
        print(f'  Fichiers manquants : {fichiers_manquants}')

    chemins = np.array(chemins)
    etiquettes = np.array(etiquettes)
    largeurs = np.array(largeurs)
    est_reel = np.array(est_reel)

    idx_synthetiques = np.where(~est_reel)[0]
    idx_reels = np.where(est_reel)[0]

    nb_avec = (etiquettes[idx_synthetiques] == 1).sum()
    nb_sans = (etiquettes[idx_synthetiques] == 0).sum()
    print(f'\n  Donnees synthetiques : {len(idx_synthetiques)}')
    print(f'    avec_fourreau : {nb_avec}')
    print(f'    sans_fourreau : {nb_sans}')

    nb_reel_avec = (etiquettes[idx_reels] == 1).sum()
    nb_reel_sans = (etiquettes[idx_reels] == 0).sum()
    print(f'  Donnees reelles : {len(idx_reels)}')
    print(f'    avec_fourreau : {nb_reel_avec}')
    print(f'    sans_fourreau : {nb_reel_sans}')

    if len(idx_synthetiques) == 0:
        print('ERREUR : Aucune donnee synthetique trouvee.')
        return None

    etiq_synth = etiquettes[idx_synthetiques]
    idx_temp, idx_test_synth, _, _ = train_test_split(
        idx_synthetiques, etiq_synth,
        test_size=0.15, stratify=etiq_synth, random_state=42
    )
    etiq_temp = etiquettes[idx_temp]
    idx_train_synth, idx_val, _, _ = train_test_split(
        idx_temp, etiq_temp,
        test_size=0.1765, stratify=etiq_temp, random_state=42
    )

    if len(idx_reels) >= 4:
        etiq_reel = etiquettes[idx_reels]
        idx_train_reel, idx_test_reel, _, _ = train_test_split(
            idx_reels, etiq_reel,
            test_size=0.30, stratify=etiq_reel, random_state=42
        )
    else:
        idx_train_reel = np.array([], dtype=int)
        idx_test_reel = idx_reels

    facteur_surechantillonnage = 20
    idx_train_reel_augmente = np.tile(idx_train_reel, facteur_surechantillonnage)

    idx_train = np.concatenate([idx_train_synth, idx_train_reel_augmente])

    print(f'\n  Repartition :')
    print(f'    Entrainement (synth)  : {len(idx_train_synth)}')
    print(f'    Entrainement (reel)   : {len(idx_train_reel)} x{facteur_surechantillonnage} = {len(idx_train_reel_augmente)}')
    print(f'    Entrainement (total)  : {len(idx_train)}')
    print(f'    Validation (synth)    : {len(idx_val)}')
    print(f'    Test (synth)          : {len(idx_test_synth)}')
    print(f'    Test (reel)           : {len(idx_test_reel)}')

    dataset_train = DatasetMultiTask(
        chemins[idx_train].tolist(),
        etiquettes[idx_train].tolist(),
        largeurs[idx_train].tolist(),
        transform=transformation_entrainement
    )
    dataset_val = DatasetMultiTask(
        chemins[idx_val].tolist(),
        etiquettes[idx_val].tolist(),
        largeurs[idx_val].tolist(),
        transform=transformation_test
    )
    dataset_test_synth = DatasetMultiTask(
        chemins[idx_test_synth].tolist(),
        etiquettes[idx_test_synth].tolist(),
        largeurs[idx_test_synth].tolist(),
        transform=transformation_test
    )

    if len(idx_test_reel) > 0:
        dataset_test_reel = DatasetMultiTask(
            chemins[idx_test_reel].tolist(),
            etiquettes[idx_test_reel].tolist(),
            largeurs[idx_test_reel].tolist(),
            transform=transformation_test
        )
    else:
        dataset_test_reel = None

    chargeur_train = DataLoader(dataset_train, batch_size=TAILLE_LOT, shuffle=True)
    chargeur_val = DataLoader(dataset_val, batch_size=TAILLE_LOT, shuffle=False)
    chargeur_test_synth = DataLoader(dataset_test_synth, batch_size=TAILLE_LOT, shuffle=False)
    chargeur_test_reel = None
    if dataset_test_reel is not None:
        chargeur_test_reel = DataLoader(dataset_test_reel, batch_size=TAILLE_LOT, shuffle=False)

    return (
        chargeur_train, chargeur_val,
        chargeur_test_synth, chargeur_test_reel,
        dataset_train, dataset_test_synth, dataset_test_reel
    )


def afficher_distribution(dataset_train, dataset_test_synth, dataset_test_reel=None):
    etiq_train = np.array(dataset_train.etiquettes)
    etiq_test = np.array(dataset_test_synth.etiquettes)

    nb_plots = 3 if dataset_test_reel is not None else 2
    figure, axes = plt.subplots(1, nb_plots, figsize=(5 * nb_plots, 5))
    figure.suptitle('Distribution des classes', fontsize=14, fontweight='bold')

    ensembles = [
        (etiq_train, 'Entrainement (synth)'),
        (etiq_test, 'Test (synth)'),
    ]
    if dataset_test_reel is not None:
        etiq_reel = np.array(dataset_test_reel.etiquettes)
        ensembles.append((etiq_reel, 'Test (reel)'))

    couleurs = ['#2ecc71', '#e74c3c']
    for i, (etiquettes, titre) in enumerate(ensembles):
        valeurs, comptages = np.unique(etiquettes, return_counts=True)
        noms = [NOMS_CLASSES[int(v)] for v in valeurs]
        barres = axes[i].bar(noms, comptages, color=couleurs[:len(valeurs)],
                             edgecolor='black', alpha=0.8)
        axes[i].set_title(titre, fontsize=12)
        axes[i].set_ylabel("Nombre d'echantillons")
        for barre, comptage in zip(barres, comptages):
            axes[i].text(barre.get_x() + barre.get_width() / 2,
                         barre.get_height() + max(comptages) * 0.02,
                         str(comptage), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/distribution_classes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Distribution sauvegardee dans resultats/distribution_classes.png')
