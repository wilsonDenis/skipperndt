# Rapport Technique -- Detection de Fourreaux Souterrains par CNN Multi-Task

## Objectif du Projet

Le projet SkipperNDT vise a detecter la presence de fourreaux (conduites souterraines) a partir de mesures de champ magnetique captees par un drone. Le drone survole le sol et enregistre des cartes d'intensite magnetique sous forme de tableaux 2D a 4 canaux. Chaque fichier `.npz` contient une image de champ magnetique.

Le modele doit repondre a deux questions :
- **Tache 1 -- Classification** : Y a-t-il un fourreau sous le sol ? (Oui ou Non)
- **Tache 2 -- Regression** : Si oui, quelle est la largeur physique du fourreau en metres ?


---


## Les Donnees

### Donnees synthetiques (generees par simulation)

Le dataset principal contient **2833 echantillons synthetiques** generes par Skipper NDT. Ces donnees simulent des mesures de champ magnetique avec et sans fourreau.

- 1700 echantillons **avec fourreau** (label=1), avec une largeur connue (`width_m` de 2.01 a 154.84 metres)
- 1133 echantillons **sans fourreau** (label=0)

Chaque fichier `.npz` contient un tableau numpy de forme `(hauteur, largeur, 4)` -- c'est une image a 4 canaux, pas une image RGB classique. Les dimensions varient d'un echantillon a l'autre (de 150 a 4000 pixels de cote).

Particularites importantes :
- Beaucoup de valeurs **NaN** (Not a Number) : les zones eloignees des fourreaux n'emettent pas de signal magnetique, elles sont donc vides. C'est un comportement physique normal.
- Les dimensions sont **variables** : chaque image a une taille differente. Il faut toutes les redimensionner a la meme taille avant de les donner au modele.


### Donnees reelles (mesurees sur le terrain)

En plus des donnees synthetiques, nous avons **102 echantillons reels** mesures par des drones sur de vrais terrains :
- 51 echantillons **avec fourreau** (avec largeur connue)
- 51 echantillons **sans fourreau**

Ces donnees reelles sont cruciales : elles permettent de verifier si le modele entraine sur des donnees synthetiques fonctionne aussi dans le monde reel.


### Le fichier CSV

Toutes les annotations sont centralisees dans un seul fichier CSV (`pipe_presence_width_detection_label.csv`) avec les colonnes :

| Colonne | Type | Description |
|---------|------|-------------|
| `field_file` | texte | Nom du fichier `.npz` |
| `label` | 0 ou 1 | 0 = pas de fourreau, 1 = fourreau present |
| `width_m` | nombre | Largeur physique en metres (uniquement si label=1) |
| `coverage_type` | texte | Type de couverture (auxiliaire) |
| `shape` | texte | Forme du fourreau (auxiliaire) |
| `noisy` | vrai/faux | Si l'echantillon contient du bruit (auxiliaire) |


---


## Architecture du Modele : CNN Simple Multi-Task

### Pourquoi "Multi-Task" ?

Plutot que d'avoir deux modeles separes (un pour la classification, un pour la regression), on utilise un seul modele qui fait les deux taches en meme temps. C'est ce qu'on appelle le **multi-task learning**.

L'idee est simple : les couches convolutives (qui analysent l'image) apprennent a extraire des caracteristiques utiles pour LES DEUX taches. Puis, deux "tetes" separees utilisent ces caracteristiques pour donner chacune leur reponse.

Avantage : le modele apprend mieux car la regression l'aide a mieux comprendre la classification, et vice versa.


### Structure du modele (fichier `src/modeles/cnn_simple.py`)

Le modele est compose de trois parties :

**Partie 1 -- Extracteur de caracteristiques (couches convolutives)**

Ce sont les couches qui "regardent" l'image et en extraient des informations utiles (formes, contours, intensites).

```python
self.couches_convolution = nn.Sequential(
    # Bloc 1 : 4 canaux d'entree -> 32 filtres
    nn.Conv2d(4, 32, kernel_size=3, padding=1),   # Detecte des motifs simples
    nn.BatchNorm2d(32),                            # Stabilise l'apprentissage
    nn.ReLU(),                                     # Activation non-lineaire
    nn.MaxPool2d(2),                               # Reduit la taille par 2

    # Bloc 2 : 32 filtres -> 64 filtres
    nn.Conv2d(32, 64, kernel_size=3, padding=1),   # Detecte des motifs plus complexes
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    # Bloc 3 : 64 filtres -> 128 filtres
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Detecte des motifs abstraits
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
)
```

Explication de chaque element :
- `Conv2d` : Applique des filtres de convolution (comme des "loupes" qui scannent l'image pour detecter des motifs)
- `BatchNorm2d` : Normalise les valeurs entre chaque couche pour que le modele apprenne plus vite et de maniere plus stable
- `ReLU` : Fonction d'activation. Elle remplace les valeurs negatives par zero. Sans elle, le modele ne pourrait apprendre que des relations lineaires (des lignes droites)
- `MaxPool2d(2)` : Divise la taille de l'image par 2 en ne gardant que la valeur maximale dans chaque carre de 2x2 pixels. Cela force le modele a se concentrer sur l'essentiel

Apres les 3 blocs, une image de 224x224 pixels devient un tableau de 28x28x128 = 100,352 valeurs.

**Partie 2 -- Couches partagees (features_partagees)**

Les 100,352 valeurs sont aplaties en un seul vecteur, puis compressees en 256 valeurs essentielles.

```python
self.features_partagees = nn.Sequential(
    nn.Flatten(),                          # 28x28x128 -> 100,352 valeurs
    nn.Linear(100352, 256),                # Compresse en 256 valeurs
    nn.ReLU(),                             # Activation
    nn.Dropout(0.5),                       # Desactive 50% des neurones aleatoirement
)
```

Le `Dropout(0.5)` est une technique de regularisation : pendant l'entrainement, il eteint aleatoirement la moitie des neurones a chaque passage. Cela empeche le modele de trop s'appuyer sur certains neurones specifiques et le force a generaliser.


**Partie 3 -- Les deux tetes de sortie**

A partir des 256 valeurs partagees, deux branches separees donnent les reponses finales :

```python
# Tete 1 : Classification (fourreau ou pas ?)
self.tete_classification = nn.Sequential(
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2),       # 2 sorties : [score_sans_fourreau, score_avec_fourreau]
)

# Tete 2 : Regression (quelle largeur ?)
self.tete_regression = nn.Sequential(
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 1),       # 1 sortie : la largeur predite en metres
)
```

La tete de classification sort 2 scores : le plus grand des deux determine la classe predite. La tete de regression sort directement un nombre (la largeur en metres).

Nombre total de parametres : **25,817,443** (~25.8 millions de valeurs que le modele ajuste pendant l'entrainement).


---


## Preparation des Donnees (fichier `src/donnees.py`)

### Etape 1 : Lecture du CSV

Le programme lit le fichier CSV pour avoir la liste de tous les echantillons avec leurs labels et largeurs.

```python
df = pd.read_csv(FICHIER_CSV, sep=';', keep_default_na=False)
```

Le parametre `keep_default_na=False` est important : sans lui, pandas convertit automatiquement le texte "N/A" en NaN (une valeur speciale "pas un nombre"). Or dans notre CSV, les echantillons sans fourreau ont "N/A" comme largeur, et on veut les garder comme texte pour les convertir en 0.0 nous-memes.


### Etape 2 : Chargement des fichiers .npz

Pour chaque echantillon, le programme :
1. Charge le fichier `.npz`
2. Remplace les NaN par des zeros (`np.nan_to_num(matrice, nan=0.0)`)
3. Normalise les valeurs entre 0 et 1 (`(valeur - min) / (max - min)`)
4. Redimensionne l'image a 224x224 pixels

```python
matrice = np.nan_to_num(matrice, nan=0.0)     # Remplace NaN par 0
mat_min, mat_max = matrice.min(), matrice.max()
if mat_max > mat_min:
    matrice = (matrice - mat_min) / (mat_max - mat_min)  # Normalise entre 0 et 1
```


### Etape 3 : Separation des donnees

Les donnees sont separees en plusieurs groupes :

| Groupe | Source | Nombre | Utilisation |
|--------|--------|--------|-------------|
| **Train (synth)** | Synthetiques | 1982 | Apprendre au modele |
| **Train (reel x20)** | Reelles | 71 x 20 = 1420 | Apprendre les patterns reels |
| **Train total** | Mix | 3402 | Total des donnees d'apprentissage |
| **Validation** | Synthetiques | 426 | Verifier pendant l'entrainement |
| **Test synth** | Synthetiques | 425 | Evaluation finale (synth) |
| **Test reel** | Reelles | 31 | Evaluation finale (reel) |


### Etape 4 : Augmentation des donnees

L'augmentation de donnees consiste a appliquer des transformations aleatoires aux images pendant l'entrainement. A chaque passage, la meme image est legerement differente, ce qui aide le modele a generaliser.

```python
transformation_entrainement = transforms.Compose([
    transforms.ToTensor(),                 # Convertit en tenseur PyTorch
    transforms.Resize((224, 224)),         # Redimensionne
    transforms.RandomHorizontalFlip(),     # Retournement horizontal aleatoire
    transforms.RandomVerticalFlip(),       # Retournement vertical aleatoire
    transforms.RandomRotation(15),         # Rotation aleatoire de -15 a +15 degres
])
```


### Etape 5 : Surechantillonnage des donnees reelles (x20)

C'est l'une des ameliorations les plus importantes. Le probleme : nous avons 1982 echantillons synthetiques mais seulement 71 echantillons reels dans le train. Le modele voit donc tres peu de donnees reelles et n'apprend pas a les reconnaitre.

La solution : on repete les donnees reelles **20 fois** dans le jeu d'entrainement. Combine avec l'augmentation de donnees (flips, rotations), chaque copie est un peu differente. Le modele voit donc 1420 versions de donnees reelles au lieu de 71.

```python
facteur_surechantillonnage = 20
idx_train_reel_augmente = np.tile(idx_train_reel, facteur_surechantillonnage)
idx_train = np.concatenate([idx_train_synth, idx_train_reel_augmente])
```

Impact sur les resultats :
- **Avant surechantillonnage** : 51.61% de precision sur les donnees reelles
- **Avec surechantillonnage x10** : 96.77% de precision sur les donnees reelles


---


## Entrainement (fichier `src/entrainement.py`)

### La fonction de perte (ce que le modele essaie de minimiser)

Le modele est entraine en minimisant une **perte combinee** qui prend en compte les deux taches :

```
perte_totale = perte_classification + 0.01 * perte_regression
```

**Perte de classification** : `CrossEntropyLoss`

C'est la fonction standard pour les problemes de classification. Elle mesure a quel point les predictions du modele sont eloignees des vrais labels. Plus la perte est basse, mieux le modele predit.

On lui donne des **poids de classes** `[1.5, 1.0]` pour compenser le desequilibre : il y a plus d'echantillons "avec fourreau" (60%) que "sans fourreau" (40%). Le poids 1.5 sur la classe "sans fourreau" dit au modele : "fais plus attention a cette classe, chaque erreur compte davantage".

```python
poids = torch.tensor([1.5, 1.0])
perte_classification = nn.CrossEntropyLoss(weight=poids)
```

**Perte de regression** : `MSELoss` (Mean Squared Error)

Elle mesure l'ecart au carre entre la largeur predite et la largeur reelle. Elle n'est calculee que sur les echantillons avec fourreau (label=1), car les echantillons sans fourreau n'ont pas de largeur a predire.

```python
masque_pipe = (etiquettes == 1)
if masque_pipe.sum() > 0:
    perte_reg = MSELoss(prediction[masque_pipe], largeur_reelle[masque_pipe])
```

**Pourquoi le coefficient 0.01 ?**

La perte de regression (MSELoss) produit des valeurs beaucoup plus grandes que la perte de classification. Par exemple, si la largeur predite est 50m et la reelle est 30m, MSE = (50-30)^2 = 400. Alors que la classification donne des pertes autour de 0.1-1.0. Le coefficient 0.01 (appele `LAMBDA_REGRESSION`) equilibre les deux pour que ni l'une ni l'autre ne domine.


### Optimiseur : Adam

L'optimiseur est l'algorithme qui ajuste les parametres du modele pour reduire la perte. Adam est l'un des plus utilises car il ajuste automatiquement la vitesse d'apprentissage pour chaque parametre.

```python
optimiseur = torch.optim.Adam(modele.parameters(), lr=0.001)
```

Le `lr=0.001` (learning rate, taux d'apprentissage) controle la taille des pas : trop grand et le modele "saute" au-dessus de la solution ; trop petit et il met trop longtemps a converger.


### Planificateur : ReduceLROnPlateau

Si la perte de validation ne diminue plus pendant 3 epoques, le learning rate est divise par 2. Cela permet au modele de faire des ajustements plus fins quand il commence a stagner.

```python
planificateur = ReduceLROnPlateau(optimiseur, mode='min', factor=0.5, patience=3)
```


### Early Stopping (arret anticipe)

Pour eviter le **surapprentissage** (le modele apprend par coeur les donnees d'entrainement au lieu de generaliser), on surveille la perte de validation. Si elle ne diminue plus pendant **7 epoques consecutives**, l'entrainement s'arrete automatiquement et on garde le modele qui avait la meilleure perte de validation.

```python
if perte_val < meilleure_perte_validation:
    meilleure_perte_validation = perte_val
    compteur_patience = 0
    sauvegarder_modele()
else:
    compteur_patience += 1
    if compteur_patience >= 7:
        arreter_entrainement()
```


### Deroulement d'une epoque

Une **epoque** = un passage complet sur toutes les donnees d'entrainement.

1. **Phase d'entrainement** : le modele voit chaque batch de 64 images, fait ses predictions, calcule l'erreur, et ajuste ses parametres
2. **Phase de validation** : le modele voit les donnees de validation SANS ajuster ses parametres. Cela mesure sa capacite a generaliser sur des donnees qu'il n'a pas vues pendant l'entrainement


---


## Evaluation (fichier `src/evaluation.py`)

### Metriques de classification

| Metrique | Definition | Interpretation |
|----------|-----------|---------------|
| **Precision** | Nombre de bonnes reponses / Total | Si le modele dit "fourreau" 100 fois et a raison 95 fois, precision = 0.95 |
| **Recall** | Vrais positifs / Total des positifs reels | Si 100 fourreaux existent et le modele en detecte 90, recall = 0.90 |
| **F1-Score** | Moyenne harmonique de Precision et Recall | Combine les deux metriques en une seule |

### Metriques de regression

| Metrique | Definition | Interpretation |
|----------|-----------|---------------|
| **MAE** | Erreur absolue moyenne | En moyenne, de combien de metres la prediction est-elle fausse |
| **RMSE** | Racine de l'erreur quadratique moyenne | Comme MAE mais penalise davantage les grosses erreurs |


---


## Historique des Resultats et Ameliorations

### Version 1 : Entrainement sur donnees synthetiques uniquement

| Metrique | Synthetique | Reel |
|----------|:-----------:|:----:|
| Classification | 99.76% | 50.00% |
| Regression MAE | 12.62 m | 24.44 m |

**Probleme** : Le modele avait 99.76% sur le synthetique mais seulement **50% sur le reel**. Il classait TOUS les echantillons reels comme "avec fourreau". Il ne savait absolument pas reconnaitre les donnees reelles car il n'en avait jamais vu pendant l'entrainement.


### Version 2 : Ajout des donnees reelles dans le train (70% reel en train, 30% en test)

| Metrique | Synthetique | Reel |
|----------|:-----------:|:----:|
| Classification | 99.76% | 51.61% |
| Regression MAE | 12.58 m | 13.03 m |

**Legers progres** : la regression est passee de 24.44m a 13.03m (bonne amelioration). Mais la classification est restee a ~51% car les 71 echantillons reels etaient noyes parmi 1982 synthetiques (3.5% du total).


### Version 3 : Surechantillonnage x10 des donnees reelles

| Metrique | Synthetique | Reel |
|----------|:-----------:|:----:|
| Classification | 99.06% | **96.77%** |
| Regression MAE | 11.97 m | **14.49 m** |

**Amelioration massive** : la classification reelle est passee de 51.61% a **96.77%**. Le surechantillonnage a permis au modele de voir suffisamment de donnees reelles pour apprendre leurs specificites. Seule 1 erreur sur 31 echantillons reels.


### Comparaison avec le U-Net

Nous avons aussi teste un modele U-Net Multi-Task (architecture avec encodeur-decodeur et skip connections) sur les memes donnees. Resultat :

| Modele | Params | Classif Synth | MAE Synth | Classif Reel | MAE Reel |
|--------|:------:|:-----:|:---:|:-----:|:---:|
| **CNN Simple** | 25.8M | 99.06% | 11.97 m | **96.77%** | **14.49 m** |
| U-Net | 7.8M | 99.53% | 16.75 m | 74.19% | 14.95 m |

Le CNN Simple a surpasse le U-Net sur les donnees reelles (+22 points en classification). Le U-Net avait tendance a voir des fourreaux partout (recall sans_fourreau = 0.50 seulement). Le CNN Simple a donc ete conserve comme modele final.


### Version 4 (finale) : Surechantillonnage x20 + 35 epoques + patience 7

Optimisations :
- Surechantillonnage passe de x10 a **x20** (1420 echantillons reels dans le train au lieu de 710)
- Epoques max passees de 25 a **35** (plus de temps pour apprendre)
- Patience passee de 5 a **7** (laisse le modele plus de chances avant d'arreter)

| Metrique | Synthetique | Reel |
|----------|:-----------:|:----:|
| Classification | 96.94% | **100.00%** |
| Regression MAE | 12.68 m | **14.13 m** |

```
               precision    recall  f1-score   support

sans_fourreau       1.00      1.00      1.00        16
avec_fourreau       1.00      1.00      1.00        15

     accuracy                           1.00        31
```

**Resultat parfait sur les donnees reelles** : 31/31 echantillons correctement classes. Zero erreur.


### Resume de l'evolution

| Version | Changement cle | Classif Reel | MAE Reel |
|:---:|---|:---:|:---:|
| V1 | Synth seul | 50.00% | 24.44 m |
| V2 | +reel dans train (71 samples) | 51.61% | 13.03 m |
| V3 | +surechantillonnage x10 | 96.77% | 14.49 m |
| V4 | +surechantillonnage x20 + 35 epoques | 100.00% | 14.13 m (CNN) |
| **V5** | **+mesure physique (pixels x 20cm)** | **100.00%** | **3.02 m** |


### Version 5 (finale) : Approche hybride CNN + mesure physique

Le constat avec la V4 : la classification est parfaite (100%), mais la regression CNN se trompe de ~14m en moyenne. Le CNN ne "mesure" pas, il "devine" la largeur a partir de patterns visuels. En redimensionnant toutes les images a 224x224, on perd l'information d'echelle.

**La solution** : garder le CNN pour la classification (100%) et remplacer la regression CNN par un calcul physique base sur les cartes magnetiques. Chaque pixel represente 20cm de terrain.

**L'algorithme** (fichier `src/mesure_largeur.py`) :

1. Calculer la norme des 4 canaux magnetiques : `norme = sqrt(c0^2 + c1^2 + c2^2 + c3^2)`
2. Normaliser entre 0 et 1
3. Trouver le centre du fourreau avec `scipy.ndimage.center_of_mass`
4. Prendre un cross-section de 5 pixels de large au centre, dans les deux directions (horizontal et vertical)
5. Mesurer la largeur a 30% du pic d'intensite (variante du FWHM)
6. Prendre la plus petite dimension (le fourreau est lineaire : long dans un sens, etroit dans l'autre)
7. Convertir en metres : `largeur_pixels x 0.20m`

**Resultats finaux** :

| Metrique | Synthetique | Reel |
|----------|:-----------:|:----:|
| Classification | 98.59% | **100.00%** |
| Regression CNN | 12.11 m | 14.91 m |
| **Mesure Physique** | -- | **3.02 m** |

La mesure physique est **5x plus precise** que la regression CNN (3.02m vs 14.91m de MAE).


---


## Configuration Finale (fichier `src/config.py`)

| Parametre | Valeur | Explication |
|-----------|--------|-------------|
| `TAILLE_IMAGE` | 224 | Toutes les images sont redimensionnees en 224x224 pixels |
| `NOMBRE_CANAUX` | 4 | Chaque image a 4 canaux (pas 3 comme RGB) |
| `NOMBRE_CLASSES` | 2 | Deux classes : sans_fourreau (0) et avec_fourreau (1) |
| `TAILLE_LOT` | 64 | Le modele traite 64 images a la fois (batch size) |
| `NOMBRE_EPOQUES` | 35 | Maximum 35 passages sur les donnees |
| `TAUX_APPRENTISSAGE` | 0.001 | Vitesse d'ajustement des parametres |
| `PATIENCE` | 7 | Arret si pas d'amelioration pendant 7 epoques |
| `LAMBDA_REGRESSION` | 0.01 | Poids de la regression dans la perte totale |
| `POIDS_CLASSES` | [1.5, 1.0] | Poids pour compenser le desequilibre des classes |


---


## Structure des Fichiers du Projet

```
skipperndt/
    main.py                    <- Point d'entree : lance le pipeline complet
    requirements.txt           <- Dependances Python
    data/
        nettoye/
            avec_fourreau/     <- 1700 fichiers .npz synthetiques (label=1)
            sans_fourreau/     <- 1133 fichiers .npz synthetiques (label=0)
    real_data/
        real_data_00000.npz    <- 51 fichiers reels avec fourreau
        real_data_no_pipe_00000.npz  <- 51 fichiers reels sans fourreau
        pipe_presence_width_detection_label.csv  <- Labels pour tout
    src/
        config.py              <- Tous les hyperparametres
        donnees.py             <- Chargement, preparation, augmentation des donnees
        entrainement.py        <- Boucle d'entrainement multi-task
        evaluation.py          <- Metriques, graphiques, matrices de confusion
        modeles/
            cnn_simple.py      <- Architecture du CNN Multi-Task
            utilitaires.py     <- Fonctions de creation et comptage de modeles
    resultats/                 <- Graphiques et modele sauvegarde genereres automatiquement
        meilleur_modele.pth
        courbes_apprentissage.png
        matrice_confusion_*.png
        regression_*.png
        distribution_classes.png
        exemples_predictions.png
    docs/
        documentation.md       <- Documentation detaillee du pipeline
        rapport.md             <- Ce fichier
```


---


## Dependances

```
torch            <- Framework de deep learning (PyTorch)
torchvision      <- Transformations d'images pour PyTorch
numpy            <- Calcul numerique (manipulation de tableaux)
pandas           <- Lecture et traitement du fichier CSV
matplotlib       <- Generation de graphiques
scikit-learn     <- Metriques d'evaluation et separation des donnees
opencv-python    <- Traitement d'images (redimensionnement, contours)
tqdm             <- Barres de progression
```

Pour installer : `pip install -r requirements.txt`

Pour lancer l'entrainement : `python main.py`
