from src.modeles.cnn_simple import CNNSimple
from src.modeles.cnn_ameliore import CNNAmeliore
from src.modeles.unet import UNetClassification

def compter_parametres(modele):
    total = sum((p.numel() for p in modele.parameters()))
    entrainables = sum((p.numel() for p in modele.parameters() if p.requires_grad))
    print(f' Paramètres totaux    : {total:,}')
    print(f'   Paramètres entraînables : {entrainables:,}')
    return total

def creer_modele(nom_modele):
    modeles_disponibles = {'cnn_simple': CNNSimple, 'cnn_ameliore': CNNAmeliore, 'unet': UNetClassification}
    if nom_modele not in modeles_disponibles:
        raise ValueError(f"Modèle '{nom_modele}' inconnu. Choix possibles : {list(modeles_disponibles.keys())}")
    modele = modeles_disponibles[nom_modele]()
    print(f'\n🏗️  Modèle créé : {nom_modele}')
    compter_parametres(modele)
    return modele
