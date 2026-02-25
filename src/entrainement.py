import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from src.config import APPAREIL, NOMBRE_EPOQUES, TAUX_APPRENTISSAGE, PATIENCE, DOSSIER_RESULTATS

class Entraineur:

    def __init__(self, modele, chargeur_entrainement, chargeur_validation):
        self.modele = modele.to(APPAREIL)
        self.chargeur_entrainement = chargeur_entrainement
        self.chargeur_validation = chargeur_validation
        self.fonction_perte = nn.CrossEntropyLoss()
        self.optimiseur = torch.optim.Adam(self.modele.parameters(), lr=TAUX_APPRENTISSAGE)
        self.planificateur = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiseur, mode='min', factor=0.5, patience=3)
        self.historique = {'perte_entrainement': [], 'perte_validation': [], 'precision_entrainement': [], 'precision_validation': []}

    def _entrainer_une_epoque(self):
        self.modele.train()
        perte_totale = 0
        bonnes_predictions = 0
        total_echantillons = 0
        for images, etiquettes in self.chargeur_entrainement:
            images = images.to(APPAREIL)
            etiquettes = etiquettes.squeeze().long().to(APPAREIL)
            self.optimiseur.zero_grad()
            sorties = self.modele(images)
            perte = self.fonction_perte(sorties, etiquettes)
            perte.backward()
            self.optimiseur.step()
            perte_totale += perte.item() * images.size(0)
            _, predictions = torch.max(sorties, 1)
            bonnes_predictions += (predictions == etiquettes).sum().item()
            total_echantillons += images.size(0)
        perte_moyenne = perte_totale / total_echantillons
        precision = bonnes_predictions / total_echantillons
        return (perte_moyenne, precision)

    @torch.no_grad()
    def _valider(self):
        self.modele.eval()
        perte_totale = 0
        bonnes_predictions = 0
        total_echantillons = 0
        for images, etiquettes in self.chargeur_validation:
            images = images.to(APPAREIL)
            etiquettes = etiquettes.squeeze().long().to(APPAREIL)
            sorties = self.modele(images)
            perte = self.fonction_perte(sorties, etiquettes)
            perte_totale += perte.item() * images.size(0)
            _, predictions = torch.max(sorties, 1)
            bonnes_predictions += (predictions == etiquettes).sum().item()
            total_echantillons += images.size(0)
        perte_moyenne = perte_totale / total_echantillons
        precision = bonnes_predictions / total_echantillons
        return (perte_moyenne, precision)

    def entrainer(self):
        meilleure_perte_validation = float('inf')
        compteur_patience = 0
        meilleur_modele = None
        print(f"\n🚀 Début de l'entraînement ({NOMBRE_EPOQUES} époques max)")
        print(f'   Early stopping : patience = {PATIENCE}')
        print('-' * 60)
        for epoque in range(NOMBRE_EPOQUES):
            perte_train, precision_train = self._entrainer_une_epoque()
            perte_val, precision_val = self._valider()
            self.historique['perte_entrainement'].append(perte_train)
            self.historique['perte_validation'].append(perte_val)
            self.historique['precision_entrainement'].append(precision_train)
            self.historique['precision_validation'].append(precision_val)
            self.planificateur.step(perte_val)
            print(f'Époque [{epoque + 1:2d}/{NOMBRE_EPOQUES}] | Perte Train: {perte_train:.4f} | Préc Train: {precision_train:.4f} | Perte Val: {perte_val:.4f} | Préc Val: {precision_val:.4f}')
            if perte_val < meilleure_perte_validation:
                meilleure_perte_validation = perte_val
                compteur_patience = 0
                meilleur_modele = copy.deepcopy(self.modele.state_dict())
                print(f'   ✅ Meilleur modèle sauvegardé (perte val: {perte_val:.4f})')
            else:
                compteur_patience += 1
                if compteur_patience >= PATIENCE:
                    print(f"\n⏹️  Early stopping à l'époque {epoque + 1}")
                    break
        if meilleur_modele is not None:
            self.modele.load_state_dict(meilleur_modele)
            chemin_sauvegarde = f'{DOSSIER_RESULTATS}/meilleur_modele.pth'
            torch.save(meilleur_modele, chemin_sauvegarde)
            print(f'\n💾 Meilleur modèle sauvegardé : {chemin_sauvegarde}')
        return self.historique
