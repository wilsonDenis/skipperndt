import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.donnees import charger_donnees, afficher_distribution
from src.modeles import creer_modele
from src.modeles.utilitaires import compter_parametres
from src.entrainement import Entraineur
from src.evaluation import (
    evaluer_modele,
    afficher_courbes_apprentissage,
    afficher_matrice_confusion,
    afficher_exemples_predictions,
    comparer_modeles,
)


def entrainer_un_modele(nom_modele, chargeur_train, chargeur_val, chargeur_test):
    """
    Entraîne un seul modèle et retourne ses résultats.
    """
    print(f"\n{'='*60}")
    print(f"  MODÈLE : {nom_modele.upper()}")
    print(f"{'='*60}")

    
    modele = creer_modele(nom_modele)
    nombre_params = sum(p.numel() for p in modele.parameters())

   
    entraineur = Entraineur(modele, chargeur_train, chargeur_val)
    historique = entraineur.entrainer()

    
    afficher_courbes_apprentissage(historique)

    
    predictions, vraies_etiquettes, precision = evaluer_modele(modele, chargeur_test)
    afficher_matrice_confusion(predictions, vraies_etiquettes)

    return {
        "precision": precision,
        "parametres": nombre_params,
        "historique": historique,
        "modele": modele,
    }


def pipeline_comparatif():
    """
    Entraîne les 3 modèles (CNN Simple, CNN Amélioré, U-Net)
    sur les mêmes données et les compare.
    """
    print("\n" + "=" * 30)
    print("  ENTRAÎNEMENT COMPARATIF DE TOUS LES MODÈLES")
    print("" * 30)


    print("\n Chargement des données...")
    (
        chargeur_train,
        chargeur_val,
        chargeur_test,
        dataset_train,
        dataset_test,
    ) = charger_donnees()

    if chargeur_train is None:
        print("Pas de données trouvées. Arrêt.")
        return

    afficher_distribution(dataset_train, dataset_test)

  
    modeles_a_tester = ["cnn_simple", "cnn_ameliore", "unet"]
    resultats = {}

    for nom in modeles_a_tester:
        resultats[nom] = entrainer_un_modele(
            nom, chargeur_train, chargeur_val, chargeur_test
        )

    print("\n" + "=" * 60)
    print("   COMPARAISON FINALE")
    print("=" * 60)

    for nom, res in resultats.items():
        print(f"  {nom:20s} | Précision: {res['precision']*100:.2f}% | Params: {res['parametres']:>10,}")

        
    resultats_comparaison = {
        nom: {"precision": res["precision"], "parametres": res["parametres"]}
        for nom, res in resultats.items()
    }
    comparer_modeles(resultats_comparaison)

    
    meilleur_nom = max(resultats, key=lambda n: resultats[n]["precision"])
    print(f"\n🏆 Meilleur modèle : {meilleur_nom} ({resultats[meilleur_nom]['precision']*100:.2f}%)")
    afficher_exemples_predictions(resultats[meilleur_nom]["modele"], chargeur_test, nombre=8)


if __name__ == "__main__":
    pipeline_comparatif()

