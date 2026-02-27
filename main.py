import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.donnees import charger_donnees, afficher_distribution
from src.modeles import creer_modele
from src.entrainement import Entraineur
from src.evaluation import (
    evaluer_modele,
    afficher_courbes_apprentissage,
    afficher_matrice_confusion,
    afficher_resultats_regression,
    afficher_exemples_predictions,
)
from src.mesure_largeur import mesurer_largeur_physique


def evaluer_mesure_physique(dataset_test_reel):
    chemins = dataset_test_reel.chemins_fichiers
    etiquettes = dataset_test_reel.etiquettes
    largeurs_vraies = dataset_test_reel.largeurs

    erreurs = []
    print(f'\n  Mesure physique sur {len(chemins)} echantillons reels :')

    for i in range(len(chemins)):
        if etiquettes[i] != 1:
            continue

        largeur_reelle = largeurs_vraies[i]
        largeur_mesuree = mesurer_largeur_physique(chemins[i])
        erreur = abs(largeur_mesuree - largeur_reelle)
        erreurs.append(erreur)

    if len(erreurs) > 0:
        mae = np.mean(erreurs)
        rmse = np.sqrt(np.mean([e ** 2 for e in erreurs]))
        print(f'    Echantillons avec pipe : {len(erreurs)}')
        print(f'    MAE  : {mae:.2f} m')
        print(f'    RMSE : {rmse:.2f} m')
        return {'mae': mae, 'rmse': rmse}

    print('    Aucun echantillon avec pipe.')
    return {'mae': 0.0, 'rmse': 0.0}


def pipeline():
    print('\n' + '=' * 60)
    print('  PIPELINE MULTI-TASK : CLASSIFICATION + REGRESSION')
    print('=' * 60)

    print('\n  Chargement des donnees...')
    resultat = charger_donnees()

    if resultat is None:
        print('Pas de donnees trouvees. Arret.')
        return

    (
        chargeur_train, chargeur_val,
        chargeur_test_synth, chargeur_test_reel,
        dataset_train, dataset_test_synth, dataset_test_reel
    ) = resultat

    afficher_distribution(dataset_train, dataset_test_synth, dataset_test_reel)

    modele = creer_modele('cnn_simple')
    nombre_params = sum(p.numel() for p in modele.parameters())

    entraineur = Entraineur(modele, chargeur_train, chargeur_val)
    historique = entraineur.entrainer()

    afficher_courbes_apprentissage(historique)

    print('\n' + '=' * 60)
    print('  EVALUATION SUR DONNEES SYNTHETIQUES')
    print('=' * 60)
    resultats_synth = evaluer_modele(modele, chargeur_test_synth, nom_ensemble='Test Synthetique')
    afficher_matrice_confusion(resultats_synth, suffixe=' (Synthetique)')
    afficher_resultats_regression(resultats_synth, suffixe=' (Synthetique)')

    resultats_reel_physique = None
    if chargeur_test_reel is not None:
        print('\n' + '=' * 60)
        print('  EVALUATION SUR DONNEES REELLES')
        print('=' * 60)
        resultats_reel = evaluer_modele(modele, chargeur_test_reel, nom_ensemble='Test Reel')
        afficher_matrice_confusion(resultats_reel, suffixe=' (Reel)')
        afficher_resultats_regression(resultats_reel, suffixe=' (Reel)')

        print('\n' + '=' * 60)
        print('  MESURE PHYSIQUE DE LARGEUR (1 pixel = 20cm)')
        print('=' * 60)
        from src.mesure_largeur import mesurer_largeur_physique
        resultats_reel_physique = evaluer_mesure_physique(dataset_test_reel)

    print('\n' + '=' * 60)
    print('  RESUME FINAL')
    print('=' * 60)
    print(f'  Modele : CNN Simple Multi-Task ({nombre_params:,} parametres)')
    print(f'  Classification (synth) : {resultats_synth["precision"] * 100:.2f}%')
    print(f'  Regression MAE (synth) : {resultats_synth["mae"]:.2f} m')
    if chargeur_test_reel is not None:
        print(f'  Classification (reel)  : {resultats_reel["precision"] * 100:.2f}%')
        print(f'  Regression MAE CNN (reel)  : {resultats_reel["mae"]:.2f} m')
    if resultats_reel_physique is not None:
        print(f'  Regression MAE Phys (reel) : {resultats_reel_physique["mae"]:.2f} m')
    print('=' * 60)

    afficher_exemples_predictions(modele, chargeur_test_synth, nombre=8)


if __name__ == '__main__':
    pipeline()
