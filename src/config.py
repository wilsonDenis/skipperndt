import torch
import os

if torch.cuda.is_available():
    APPAREIL = torch.device('cuda')
elif torch.backends.mps.is_available():
    APPAREIL = torch.device('mps')
else:
    APPAREIL = torch.device('cpu')

print(f'  Appareil utilise : {APPAREIL}')

RACINE_PROJET = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_DONNEES = os.path.join(RACINE_PROJET, 'data', 'nettoye')
DOSSIER_DONNEES_REELLES = os.path.join(RACINE_PROJET, 'real_data')
FICHIER_CSV = os.path.join(DOSSIER_DONNEES_REELLES, 'pipe_presence_width_detection_label.csv')
DOSSIER_RESULTATS = os.path.join(RACINE_PROJET, 'resultats')

os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
os.makedirs(DOSSIER_DONNEES, exist_ok=True)

TAILLE_IMAGE = 224
NOMBRE_CANAUX = 4
NOMBRE_CLASSES = 2
TAILLE_LOT = 64
NOMBRE_EPOQUES = 35
TAUX_APPRENTISSAGE = 0.001
PATIENCE = 7
LAMBDA_REGRESSION = 0.01
POIDS_CLASSES = [1.5, 1.0]

NOMS_CLASSES = ['sans_fourreau', 'avec_fourreau']
