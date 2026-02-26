import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from src.config import APPAREIL, NOMS_CLASSES, DOSSIER_RESULTATS


@torch.no_grad()
def evaluer_modele(modele, chargeur_test, nom_ensemble='Test'):
    modele.eval()
    toutes_predictions = []
    toutes_etiquettes = []
    toutes_largeurs_pred = []
    toutes_largeurs_vraies = []
    bonnes_predictions = 0
    total = 0

    for images, etiquettes, largeurs in chargeur_test:
        images = images.to(APPAREIL)
        etiquettes = etiquettes.squeeze().long().to(APPAREIL)
        largeurs = largeurs.to(APPAREIL)

        sorties_classif, sorties_reg = modele(images)
        _, predictions = torch.max(sorties_classif, 1)

        toutes_predictions.extend(predictions.cpu().numpy())
        toutes_etiquettes.extend(etiquettes.cpu().numpy())
        toutes_largeurs_pred.extend(sorties_reg.cpu().numpy())
        toutes_largeurs_vraies.extend(largeurs.cpu().numpy())

        bonnes_predictions += (predictions == etiquettes).sum().item()
        total += images.size(0)

    precision = bonnes_predictions / total

    predictions_arr = np.array(toutes_predictions)
    etiquettes_arr = np.array(toutes_etiquettes)
    largeurs_pred_arr = np.array(toutes_largeurs_pred)
    largeurs_vraies_arr = np.array(toutes_largeurs_vraies)

    masque_pipe = etiquettes_arr == 1
    mae = 0.0
    rmse = 0.0
    if masque_pipe.sum() > 0:
        erreurs = np.abs(largeurs_pred_arr[masque_pipe] - largeurs_vraies_arr[masque_pipe])
        mae = erreurs.mean()
        rmse = np.sqrt(((largeurs_pred_arr[masque_pipe] - largeurs_vraies_arr[masque_pipe]) ** 2).mean())

    print(f'\n  [{nom_ensemble}] Classification :')
    print(f'    Precision : {precision:.4f} ({precision * 100:.2f}%)')
    if masque_pipe.sum() > 0:
        print(f'  [{nom_ensemble}] Regression (echantillons avec pipe) :')
        print(f'    MAE  : {mae:.2f} m')
        print(f'    RMSE : {rmse:.2f} m')

    return {
        'predictions': predictions_arr,
        'etiquettes': etiquettes_arr,
        'largeurs_pred': largeurs_pred_arr,
        'largeurs_vraies': largeurs_vraies_arr,
        'precision': precision,
        'mae': mae,
        'rmse': rmse,
    }


def afficher_courbes_apprentissage(historique):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    figure.suptitle("Courbes d'apprentissage", fontsize=14, fontweight='bold')
    epoques = range(1, len(historique['perte_entrainement']) + 1)

    axes[0].plot(epoques, historique['perte_entrainement'], 'b-o', label='Train', markersize=4)
    axes[0].plot(epoques, historique['perte_validation'], 'r-o', label='Val', markersize=4)
    axes[0].set_title('Perte totale')
    axes[0].set_xlabel('Epoque')
    axes[0].set_ylabel('Perte')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoques, historique['precision_entrainement'], 'b-o', label='Train', markersize=4)
    axes[1].plot(epoques, historique['precision_validation'], 'r-o', label='Val', markersize=4)
    axes[1].set_title('Precision (Classification)')
    axes[1].set_xlabel('Epoque')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epoques, historique['mae_entrainement'], 'b-o', label='Train', markersize=4)
    axes[2].plot(epoques, historique['mae_validation'], 'r-o', label='Val', markersize=4)
    axes[2].set_title('MAE (Regression, metres)')
    axes[2].set_xlabel('Epoque')
    axes[2].set_ylabel('MAE (m)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/courbes_apprentissage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Courbes sauvegardees dans resultats/courbes_apprentissage.png')


def afficher_matrice_confusion(resultats, suffixe=''):
    matrice = confusion_matrix(resultats['etiquettes'], resultats['predictions'])
    figure, axe = plt.subplots(figsize=(8, 6))
    affichage = ConfusionMatrixDisplay(matrice, display_labels=NOMS_CLASSES)
    affichage.plot(ax=axe, cmap='Blues', values_format='d')
    axe.set_title(f'Matrice de Confusion{suffixe}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    nom_fichier = f'matrice_confusion{suffixe.replace(" ", "_").lower()}.png'
    plt.savefig(f'{DOSSIER_RESULTATS}/{nom_fichier}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Matrice sauvegardee dans resultats/{nom_fichier}')

    print(f'\n  Rapport de classification{suffixe} :')
    print(classification_report(
        resultats['etiquettes'], resultats['predictions'],
        target_names=NOMS_CLASSES
    ))


def afficher_resultats_regression(resultats, suffixe=''):
    masque_pipe = resultats['etiquettes'] == 1
    if masque_pipe.sum() == 0:
        print('Pas d echantillons avec pipe pour la regression.')
        return

    vraies = resultats['largeurs_vraies'][masque_pipe]
    predites = resultats['largeurs_pred'][masque_pipe]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    figure.suptitle(f'Regression - Prediction de largeur{suffixe}',
                     fontsize=14, fontweight='bold')

    axes[0].scatter(vraies, predites, alpha=0.5, s=20, color='#3498db')
    lim_min = min(vraies.min(), predites.min()) - 5
    lim_max = max(vraies.max(), predites.max()) + 5
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Prediction parfaite')
    axes[0].set_xlabel('Largeur reelle (m)')
    axes[0].set_ylabel('Largeur predite (m)')
    axes[0].set_title('Predictions vs Valeurs reelles')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    erreurs = predites - vraies
    axes[1].hist(erreurs, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='black', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Erreur (m)')
    axes[1].set_ylabel('Nombre d echantillons')
    axes[1].set_title(f'Distribution des erreurs (MAE={np.abs(erreurs).mean():.2f}m)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    nom_fichier = f'regression{suffixe.replace(" ", "_").lower()}.png'
    plt.savefig(f'{DOSSIER_RESULTATS}/{nom_fichier}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Resultats regression sauvegardes dans resultats/{nom_fichier}')


def afficher_exemples_predictions(modele, chargeur_test, nombre=12):
    modele.eval()
    images_batch, etiquettes_batch, largeurs_batch = next(iter(chargeur_test))
    images_batch = images_batch.to(APPAREIL)

    with torch.no_grad():
        sorties_classif, sorties_reg = modele(images_batch)
        _, predictions = torch.max(sorties_classif, 1)

    lignes = min(nombre, len(images_batch))
    colonnes = 4
    rangees = (lignes + colonnes - 1) // colonnes

    figure, axes = plt.subplots(rangees, colonnes, figsize=(14, 3.5 * rangees))
    figure.suptitle('Exemples de predictions (Classification + Regression)',
                     fontsize=14, fontweight='bold')

    for indice, axe in enumerate(axes.flat):
        if indice >= lignes:
            axe.axis('off')
            continue

        image = images_batch[indice].cpu().numpy()
        if image.ndim == 3 and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)

        vraie_classe = int(etiquettes_batch[indice])
        classe_predite = int(predictions[indice].cpu())
        largeur_vraie = float(largeurs_batch[indice])
        largeur_predite = float(sorties_reg[indice].cpu())
        est_correct = vraie_classe == classe_predite

        axe.imshow(image, cmap='gray' if image.ndim == 2 else None)
        couleur = 'green' if est_correct else 'red'

        titre = f'Vrai: {NOMS_CLASSES[vraie_classe]}\nPred: {NOMS_CLASSES[classe_predite]}'
        if vraie_classe == 1:
            titre += f'\nL: {largeur_vraie:.1f}m -> {largeur_predite:.1f}m'

        axe.set_title(titre, color=couleur, fontsize=8)
        axe.axis('off')

    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/exemples_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Exemples sauvegardes dans resultats/exemples_predictions.png')
