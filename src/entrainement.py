import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from src.config import (
    APPAREIL, NOMBRE_EPOQUES, TAUX_APPRENTISSAGE, PATIENCE,
    DOSSIER_RESULTATS, LAMBDA_REGRESSION, POIDS_CLASSES
)


class Entraineur:

    def __init__(self, modele, chargeur_entrainement, chargeur_validation):
        self.modele = modele.to(APPAREIL)
        self.chargeur_entrainement = chargeur_entrainement
        self.chargeur_validation = chargeur_validation

        poids = torch.tensor(POIDS_CLASSES, dtype=torch.float32).to(APPAREIL)
        self.perte_classification = nn.CrossEntropyLoss(weight=poids)
        self.perte_regression = nn.MSELoss()

        self.optimiseur = torch.optim.Adam(
            self.modele.parameters(), lr=TAUX_APPRENTISSAGE
        )
        self.planificateur = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiseur, mode='min', factor=0.5, patience=3
        )
        self.historique = {
            'perte_entrainement': [],
            'perte_validation': [],
            'precision_entrainement': [],
            'precision_validation': [],
            'mae_entrainement': [],
            'mae_validation': [],
        }

    def _calculer_perte(self, sorties_classif, sorties_reg, etiquettes, largeurs):
        perte_cls = self.perte_classification(sorties_classif, etiquettes)

        masque_pipe = (etiquettes == 1)
        if masque_pipe.sum() > 0:
            perte_reg = self.perte_regression(
                sorties_reg[masque_pipe],
                largeurs[masque_pipe]
            )
            mae = (sorties_reg[masque_pipe] - largeurs[masque_pipe]).abs().mean().item()
        else:
            perte_reg = torch.tensor(0.0, device=APPAREIL)
            mae = 0.0

        perte_totale = perte_cls + LAMBDA_REGRESSION * perte_reg
        return perte_totale, perte_cls.item(), perte_reg.item(), mae

    def _entrainer_une_epoque(self):
        self.modele.train()
        perte_totale = 0
        bonnes_predictions = 0
        total_echantillons = 0
        mae_total = 0
        nb_echantillons_reg = 0

        for images, etiquettes, largeurs in self.chargeur_entrainement:
            images = images.to(APPAREIL)
            etiquettes = etiquettes.squeeze().long().to(APPAREIL)
            largeurs = largeurs.to(APPAREIL)

            self.optimiseur.zero_grad()
            sorties_classif, sorties_reg = self.modele(images)
            perte, _, _, mae_batch = self._calculer_perte(
                sorties_classif, sorties_reg, etiquettes, largeurs
            )
            perte.backward()
            self.optimiseur.step()

            perte_totale += perte.item() * images.size(0)
            _, predictions = torch.max(sorties_classif, 1)
            bonnes_predictions += (predictions == etiquettes).sum().item()
            total_echantillons += images.size(0)

            nb_pipe = (etiquettes == 1).sum().item()
            if nb_pipe > 0:
                mae_total += mae_batch * nb_pipe
                nb_echantillons_reg += nb_pipe

        perte_moyenne = perte_totale / total_echantillons
        precision = bonnes_predictions / total_echantillons
        mae_moyenne = mae_total / max(nb_echantillons_reg, 1)
        return perte_moyenne, precision, mae_moyenne

    @torch.no_grad()
    def _valider(self):
        self.modele.eval()
        perte_totale = 0
        bonnes_predictions = 0
        total_echantillons = 0
        mae_total = 0
        nb_echantillons_reg = 0

        for images, etiquettes, largeurs in self.chargeur_validation:
            images = images.to(APPAREIL)
            etiquettes = etiquettes.squeeze().long().to(APPAREIL)
            largeurs = largeurs.to(APPAREIL)

            sorties_classif, sorties_reg = self.modele(images)
            perte, _, _, mae_batch = self._calculer_perte(
                sorties_classif, sorties_reg, etiquettes, largeurs
            )

            perte_totale += perte.item() * images.size(0)
            _, predictions = torch.max(sorties_classif, 1)
            bonnes_predictions += (predictions == etiquettes).sum().item()
            total_echantillons += images.size(0)

            nb_pipe = (etiquettes == 1).sum().item()
            if nb_pipe > 0:
                mae_total += mae_batch * nb_pipe
                nb_echantillons_reg += nb_pipe

        perte_moyenne = perte_totale / total_echantillons
        precision = bonnes_predictions / total_echantillons
        mae_moyenne = mae_total / max(nb_echantillons_reg, 1)
        return perte_moyenne, precision, mae_moyenne

    def entrainer(self):
        meilleure_perte_validation = float('inf')
        compteur_patience = 0
        meilleur_modele = None

        print(f"\n  Debut de l'entrainement ({NOMBRE_EPOQUES} epoques max)")
        print(f'  Early stopping : patience = {PATIENCE}')
        print(f'  Lambda regression : {LAMBDA_REGRESSION}')
        print('-' * 80)

        for epoque in range(NOMBRE_EPOQUES):
            perte_train, prec_train, mae_train = self._entrainer_une_epoque()
            perte_val, prec_val, mae_val = self._valider()

            self.historique['perte_entrainement'].append(perte_train)
            self.historique['perte_validation'].append(perte_val)
            self.historique['precision_entrainement'].append(prec_train)
            self.historique['precision_validation'].append(prec_val)
            self.historique['mae_entrainement'].append(mae_train)
            self.historique['mae_validation'].append(mae_val)

            self.planificateur.step(perte_val)

            print(
                f'Epoque [{epoque + 1:2d}/{NOMBRE_EPOQUES}] | '
                f'Perte: {perte_train:.4f}/{perte_val:.4f} | '
                f'Prec: {prec_train:.4f}/{prec_val:.4f} | '
                f'MAE: {mae_train:.2f}/{mae_val:.2f}m'
            )

            if perte_val < meilleure_perte_validation:
                meilleure_perte_validation = perte_val
                compteur_patience = 0
                meilleur_modele = copy.deepcopy(self.modele.state_dict())
                print(f'   >> Meilleur modele sauvegarde (perte val: {perte_val:.4f})')
            else:
                compteur_patience += 1
                if compteur_patience >= PATIENCE:
                    print(f"\n  Early stopping a l'epoque {epoque + 1}")
                    break

        if meilleur_modele is not None:
            self.modele.load_state_dict(meilleur_modele)
            chemin_sauvegarde = f'{DOSSIER_RESULTATS}/meilleur_modele.pth'
            torch.save(meilleur_modele, chemin_sauvegarde)
            print(f'\n  Meilleur modele sauvegarde : {chemin_sauvegarde}')

        return self.historique
