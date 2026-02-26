import os
import sys

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

    if chargeur_test_reel is not None:
        print('\n' + '=' * 60)
        print('  EVALUATION SUR DONNEES REELLES')
        print('=' * 60)
        resultats_reel = evaluer_modele(modele, chargeur_test_reel, nom_ensemble='Test Reel')
        afficher_matrice_confusion(resultats_reel, suffixe=' (Reel)')
        afficher_resultats_regression(resultats_reel, suffixe=' (Reel)')

    print('\n' + '=' * 60)
    print('  RESUME FINAL')
    print('=' * 60)
    print(f'  Modele : CNN Simple Multi-Task ({nombre_params:,} parametres)')
    print(f'  Classification (synth) : {resultats_synth["precision"] * 100:.2f}%')
    print(f'  Regression MAE (synth) : {resultats_synth["mae"]:.2f} m')
    if chargeur_test_reel is not None:
        print(f'  Classification (reel)  : {resultats_reel["precision"] * 100:.2f}%')
        print(f'  Regression MAE (reel)  : {resultats_reel["mae"]:.2f} m')
    print('=' * 60)

    afficher_exemples_predictions(modele, chargeur_test_synth, nombre=8)


if __name__ == '__main__':
    pipeline()
