# SkipperNDT -- Detection de Fourreaux Souterrains par IA

Detection de fourreaux (conduites souterraines) a partir de mesures de champ magnetique captees par drone, en utilisant un CNN multi-task (classification + regression) et une mesure physique de largeur basee sur les cartes magnetiques.

## Resultats

| Metrique | Donnees Synthetiques | Donnees Reelles |
|----------|:--------------------:|:---------------:|
| **Classification** | 98.59% | **100.00%** |
| **Regression CNN** | 12.11 m | 14.91 m |
| **Mesure Physique** | — | **3.02 m** |

Le modele atteint une classification parfaite (31/31) sur les donnees reelles. La mesure physique (1 pixel = 20cm) est **5x plus precise** que la regression CNN pour estimer la largeur.

---

## Architecture

**CNN Simple Multi-Task** (25.8M parametres) pour la classification :
- 3 blocs convolutifs (32 → 64 → 128 filtres) + BatchNorm + ReLU + MaxPool
- Couches partagees (Flatten → Linear 256 → Dropout 0.5)
- **Tete de classification** : Linear 256→64→2 (avec/sans fourreau)
- **Tete de regression** : Linear 256→64→1 (largeur en metres)

**Mesure physique** pour la largeur :
- Calcul de la norme des 4 canaux magnetiques
- Localisation du centre du fourreau (center_of_mass)
- Mesure a 30% du pic d intensite sur les cross-sections
- Conversion pixels → metres (1 pixel = 20cm)

---

## Donnees Requises

> **Les donnees ne sont PAS incluses dans le depot GitHub** car elles sont trop volumineuses. Vous devez les obtenir separement et les placer dans les dossiers suivants avant de lancer le programme.

### Structure attendue

```
skipperndt/
├── data/
│   └── nettoye/
│       ├── avec_fourreau/          # 1700 fichiers .npz synthetiques
│       └── sans_fourreau/          # 1133 fichiers .npz synthetiques
├── real_data/
│   ├── real_data_00000.npz ...     # 51 fichiers reels avec fourreau
│   ├── real_data_no_pipe_00000.npz ...  # 51 fichiers reels sans fourreau
│   └── pipe_presence_width_detection_label.csv  # Labels (obligatoire)
```

### Comment obtenir les donnees

1. **Donnees synthetiques** : decompresser `Training_database_float16.zip` dans `data/nettoye/`
2. **Donnees reelles** : placer les fichiers `.npz` et le CSV dans `real_data/`

---

## Installation

```bash
# Cloner le depot
git clone https://github.com/wilsonDenis/skipperndt.git
cd skipperndt

# Installer les dependances
pip install -r requirements.txt
```

### Dependances

| Librairie | Usage |
|-----------|-------|
| torch | Reseau de neurones (PyTorch) |
| torchvision | Transformations d images |
| numpy | Calcul numerique |
| pandas | Lecture du CSV de labels |
| matplotlib | Graphiques |
| scikit-learn | Metriques d evaluation |
| opencv-python | Traitement d images |
| scipy | Mesure physique de largeur (center_of_mass) |
| tqdm | Barres de progression |

---

## Utilisation

```bash
# Lancer l entrainement + evaluation
python main.py
```

Le programme :
1. Charge les donnees depuis le CSV
2. Entraine le modele CNN Multi-Task (35 epoques max)
3. Evalue la classification et regression CNN sur synth + reel
4. Mesure la largeur physique (pixels x 20cm) sur les donnees reelles
5. Sauvegarde les resultats dans `resultats/`

---

## Structure du Projet

```
skipperndt/
├── main.py                    # Point d entree principal
├── requirements.txt           # Dependances Python
├── src/
│   ├── config.py              # Hyperparametres et configuration
│   ├── donnees.py             # Chargement, augmentation, surechantillonnage
│   ├── entrainement.py        # Boucle d entrainement multi-task
│   ├── evaluation.py          # Metriques, graphiques, matrices de confusion
│   ├── mesure_largeur.py      # Mesure physique de largeur (pixels x 20cm)
│   └── modeles/
│       ├── cnn_simple.py      # Architecture CNN Multi-Task
│       └── utilitaires.py     # Fonctions utilitaires
├── docs/                      # Documentation de reference (non versionnee)
└── resultats/                 # Genere automatiquement
    ├── meilleur_modele.pth
    ├── courbes_apprentissage.png
    ├── matrice_confusion_*.png
    ├── regression_*.png
    └── exemples_predictions.png
```

---

## Hyperparametres

| Parametre | Valeur | Description |
|-----------|:------:|-------------|
| TAILLE_IMAGE | 224 | Redimensionnement des images |
| NOMBRE_CANAUX | 4 | Canaux de mesure magnetique |
| TAILLE_LOT | 64 | Batch size |
| NOMBRE_EPOQUES | 35 | Maximum d epoques |
| TAUX_APPRENTISSAGE | 0.001 | Learning rate (Adam) |
| PATIENCE | 7 | Early stopping |
| LAMBDA_REGRESSION | 0.01 | Poids de la regression |
| POIDS_CLASSES | [1.5, 1.0] | Compensation du desequilibre |

---

## Techniques Cles

- **Multi-Task Learning** : classification + regression avec features partagees
- **Mesure physique hybride** : CNN pour la classification, calcul pixels x 20cm pour la largeur
- **Surechantillonnage x20** des donnees reelles (71 → 1420 dans le train)
- **Data Augmentation** : flips horizontaux/verticaux, rotation ±15°
- **Early Stopping** (patience=7) + **ReduceLROnPlateau** (factor=0.5)
- **Poids de classes** [1.5, 1.0] pour compenser le desequilibre
