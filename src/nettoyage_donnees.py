import os
import glob
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
DOSSIER_SOURCE = os.getenv('DRONE_DATA_BRUT', '../data/brut')
DOSSIER_DESTINATION = os.getenv('DRONE_DATA_NETTOYE', '../data/nettoye')
EXTENSION_FICHIER = os.getenv('DRONE_DATA_EXT', '*.npz')

def remplacer_nan_par_zero(matrice):
    if not np.isnan(matrice).any():
        return matrice
    resultat = np.nan_to_num(matrice, nan=0.0)
    return resultat

def traiter_image(chemin_fichier, chemin_sauvegarde):
    try:
        if chemin_fichier.endswith('.npz'):
            avec_fichiers = np.load(chemin_fichier)
            cles = avec_fichiers.files
            if not cles:
                print(f' Archive vide : {chemin_fichier}')
                return False
            donnees = avec_fichiers[cles[0]]
        else:
            print(f'Format non supporté: {chemin_fichier}')
            return False
        nan_avant = np.isnan(donnees).sum()
        masque_valide = ~np.isnan(donnees)
        if masque_valide.ndim > 2:
            masque_valide_2d = np.any(masque_valide, axis=-1)
        else:
            masque_valide_2d = masque_valide
        masque_cv = (masque_valide_2d * 255).astype(np.uint8)
        contours, _ = cv2.findContours(masque_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f' Aucun contour valide trouvé dans {chemin_fichier}. Ignoré.')
            return False
        x_min, y_min = (float('inf'), float('inf'))
        x_max, y_max = (0, 0)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        if x_min == float('inf'):
            print(f'Image vide ou invalide {chemin_fichier}.')
            return False
        if donnees.ndim > 2:
            donnees_recadrees = donnees[y_min:y_max, x_min:x_max, :]
        else:
            donnees_recadrees = donnees[y_min:y_max, x_min:x_max]
        donnees_propres = remplacer_nan_par_zero(donnees_recadrees)
        nan_apres = np.isnan(donnees_propres).sum()
        print(f' {os.path.basename(chemin_fichier)} : NaN {nan_avant} → {nan_apres} | Taille {donnees.shape} → {donnees_propres.shape}')
        os.makedirs(os.path.dirname(chemin_sauvegarde), exist_ok=True)
        if chemin_sauvegarde.endswith('.npz'):
            np.savez_compressed(chemin_sauvegarde, data=donnees_propres)
        return True
    except Exception as e:
        print(f'Erreur lors du traitement de {chemin_fichier}: {str(e)}')
        return False

def processus_complet():
    chemin_source = Path(DOSSIER_SOURCE)
    fichiers = list(chemin_source.rglob(EXTENSION_FICHIER))
    if not fichiers:
        print(f'Aucun fichier valide {EXTENSION_FICHIER} trouvé dans {DOSSIER_SOURCE}')
        return
    print(f'🚀  Début du traitement : {len(fichiers)} fichiers trouvés.')
    succes = 0
    for fichier in tqdm(fichiers, desc='Nettoyage des données'):
        chemin_relatif = fichier.relative_to(chemin_source)
        chemin_sauvegarde = Path(DOSSIER_DESTINATION) / chemin_relatif
        if traiter_image(str(fichier), str(chemin_sauvegarde)):
            succes += 1
    print(f'✅ Traitement terminé. {succes}/{len(fichiers)} fichiers traités avec succès.')
    print(f'📁 Données sauvegardées dans : {DOSSIER_DESTINATION}')
if __name__ == '__main__':
    processus_complet()
