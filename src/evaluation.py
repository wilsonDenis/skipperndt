import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from src.config import APPAREIL, NOMS_CLASSES, DOSSIER_RESULTATS

@torch.no_grad()
def evaluer_modele(modele, chargeur_test):
    modele.eval()
    toutes_predictions = []
    toutes_etiquettes = []
    bonnes_predictions = 0
    total = 0
    for images, etiquettes in chargeur_test:
        images = images.to(APPAREIL)
        etiquettes = etiquettes.squeeze().long().to(APPAREIL)
        sorties = modele(images)
        _, predictions = torch.max(sorties, 1)
        toutes_predictions.extend(predictions.cpu().numpy())
        toutes_etiquettes.extend(etiquettes.cpu().numpy())
        bonnes_predictions += (predictions == etiquettes).sum().item()
        total += images.size(0)
    precision = bonnes_predictions / total
    print(f'\n🎯 Précision sur le test : {precision:.4f} ({precision * 100:.2f}%)')
    return (np.array(toutes_predictions), np.array(toutes_etiquettes), precision)

def afficher_courbes_apprentissage(historique):
    figure, (axe_perte, axe_precision) = plt.subplots(1, 2, figsize=(14, 5))
    figure.suptitle("Courbes d'apprentissage", fontsize=14, fontweight='bold')
    epoques = range(1, len(historique['perte_entrainement']) + 1)
    axe_perte.plot(epoques, historique['perte_entrainement'], 'b-o', label='Entraînement', markersize=4)
    axe_perte.plot(epoques, historique['perte_validation'], 'r-o', label='Validation', markersize=4)
    axe_perte.set_title('Perte (Loss)')
    axe_perte.set_xlabel('Époque')
    axe_perte.set_ylabel('Perte')
    axe_perte.legend()
    axe_perte.grid(True, alpha=0.3)
    axe_precision.plot(epoques, historique['precision_entrainement'], 'b-o', label='Entraînement', markersize=4)
    axe_precision.plot(epoques, historique['precision_validation'], 'r-o', label='Validation', markersize=4)
    axe_precision.set_title('Précision (Accuracy)')
    axe_precision.set_xlabel('Époque')
    axe_precision.set_ylabel('Précision')
    axe_precision.legend()
    axe_precision.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/courbes_apprentissage.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('💾 Courbes sauvegardées dans resultats/courbes_apprentissage.png')

def afficher_matrice_confusion(predictions, etiquettes):
    matrice = confusion_matrix(etiquettes, predictions)
    figure, axe = plt.subplots(figsize=(8, 6))
    affichage = ConfusionMatrixDisplay(matrice, display_labels=NOMS_CLASSES)
    affichage.plot(ax=axe, cmap='Blues', values_format='d')
    axe.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/matrice_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('💾 Matrice sauvegardée dans resultats/matrice_confusion.png')
    print('\n📋 Rapport de classification :')
    print(classification_report(etiquettes, predictions, target_names=NOMS_CLASSES))

def afficher_exemples_predictions(modele, chargeur_test, nombre=12):
    modele.eval()
    images_batch, etiquettes_batch = next(iter(chargeur_test))
    images_batch = images_batch.to(APPAREIL)
    with torch.no_grad():
        sorties = modele(images_batch)
        _, predictions = torch.max(sorties, 1)
    lignes = min(nombre, len(images_batch))
    colonnes = 4
    rangees = (lignes + colonnes - 1) // colonnes
    figure, axes = plt.subplots(rangees, colonnes, figsize=(12, 3 * rangees))
    figure.suptitle('Exemples de prédictions', fontsize=14, fontweight='bold')
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
        est_correct = vraie_classe == classe_predite
        axe.imshow(image, cmap='gray' if image.ndim == 2 else None)
        couleur = 'green' if est_correct else 'red'
        axe.set_title(f'Vrai: {NOMS_CLASSES[vraie_classe]}\nPréd: {NOMS_CLASSES[classe_predite]}', color=couleur, fontsize=9)
        axe.axis('off')
    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/exemples_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('💾 Exemples sauvegardés dans resultats/exemples_predictions.png')

def comparer_modeles(resultats_modeles):
    noms = list(resultats_modeles.keys())
    precisions = [r['precision'] * 100 for r in resultats_modeles.values()]
    parametres = [r['parametres'] for r in resultats_modeles.values()]
    figure, (axe_prec, axe_param) = plt.subplots(1, 2, figsize=(14, 5))
    figure.suptitle('Comparaison des modèles', fontsize=14, fontweight='bold')
    couleurs = ['#3498db', '#e74c3c', '#2ecc71']
    barres = axe_prec.bar(noms, precisions, color=couleurs[:len(noms)], edgecolor='black', alpha=0.8)
    axe_prec.set_ylabel('Précision (%)')
    axe_prec.set_title('Précision sur le test')
    axe_prec.set_ylim(0, 100)
    for barre, prec in zip(barres, precisions):
        axe_prec.text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 1, f'{prec:.1f}%', ha='center', fontweight='bold')
    barres = axe_param.bar(noms, parametres, color=couleurs[:len(noms)], edgecolor='black', alpha=0.8)
    axe_param.set_ylabel('Nombre de paramètres')
    axe_param.set_title('Taille du modèle')
    for barre, param in zip(barres, parametres):
        axe_param.text(barre.get_x() + barre.get_width() / 2, barre.get_height() + max(parametres) * 0.02, f'{param:,}', ha='center', fontweight='bold', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{DOSSIER_RESULTATS}/comparaison_modeles.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('💾 Comparaison sauvegardée dans resultats/comparaison_modeles.png')
