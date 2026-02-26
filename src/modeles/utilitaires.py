from src.modeles.cnn_simple import CNNSimple


def compter_parametres(modele):
    total = sum(p.numel() for p in modele.parameters())
    entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    print(f'  Parametres totaux       : {total:,}')
    print(f'  Parametres entrainables : {entrainables:,}')
    return total


def creer_modele(nom_modele='cnn_simple'):
    modeles_disponibles = {'cnn_simple': CNNSimple}
    if nom_modele not in modeles_disponibles:
        raise ValueError(f"Modele '{nom_modele}' inconnu. Choix : {list(modeles_disponibles.keys())}")
    modele = modeles_disponibles[nom_modele]()
    print(f'\n  Modele cree : {nom_modele}')
    compter_parametres(modele)
    return modele
