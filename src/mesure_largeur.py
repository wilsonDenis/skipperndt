import numpy as np
from scipy import ndimage

RESOLUTION_METRES_PAR_PIXEL = 0.20


def mesurer_largeur_physique(chemin_fichier):
    data = np.load(chemin_fichier)
    key = 'data' if 'data' in data.files else data.files[0]
    mat = data[key].astype(np.float64)
    mat = np.nan_to_num(mat, nan=0.0)

    if mat.ndim == 2:
        norme = np.abs(mat)
    else:
        norme = np.sqrt(np.sum(mat ** 2, axis=2))

    if norme.max() == 0:
        return 0.0

    norme = norme / norme.max()

    centre_y, centre_x = ndimage.center_of_mass(norme)
    centre_y, centre_x = int(centre_y), int(centre_x)

    marge = 2
    y_min = max(0, centre_y - marge)
    y_max = min(norme.shape[0], centre_y + marge + 1)
    x_min = max(0, centre_x - marge)
    x_max = min(norme.shape[1], centre_x + marge + 1)

    profil_horizontal = norme[y_min:y_max, :].mean(axis=0)
    profil_vertical = norme[:, x_min:x_max].mean(axis=1)

    seuil = 0.3

    largeur_h = _largeur_a_seuil(profil_horizontal, seuil)
    largeur_v = _largeur_a_seuil(profil_vertical, seuil)

    largeur_pixels = min(largeur_h, largeur_v)
    largeur_metres = largeur_pixels * RESOLUTION_METRES_PAR_PIXEL

    return largeur_metres


def _largeur_a_seuil(profil, seuil_relatif):
    pic = profil.max()
    if pic == 0:
        return 0
    threshold = pic * seuil_relatif
    au_dessus = profil >= threshold
    if au_dessus.sum() == 0:
        return 0
    indices = np.where(au_dessus)[0]
    return indices[-1] - indices[0] + 1
