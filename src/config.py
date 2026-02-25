import torch
import os
if torch.cuda.is_available():
    APPAREIL = torch.device('cuda')
elif torch.backends.mps.is_available():
    APPAREIL = torch.device('mps')
else:
    APPAREIL = torch.device('cpu')
print(f' Appareil utilisé : {APPAREIL}')
RACINE_PROJET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_DONNEES = os.path.join(RACINE_PROJET, 'data', 'nettoye')
DOSSIER_RESULTATS = os.path.join(RACINE_PROJET, 'resultats')
os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
os.makedirs(DOSSIER_DONNEES, exist_ok=True)
TAILLE_IMAGE = 224
NOMBRE_CANAUX = 4
NOMBRE_CLASSES = 2
TAILLE_LOT = 64
NOMBRE_EPOQUES = 25
TAUX_APPRENTISSAGE = 0.001
PATIENCE = 5
NOMS_CLASSES = ['sans_fourreau', 'avec_fourreau']
