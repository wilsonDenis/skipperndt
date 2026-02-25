# Documentation Technique du Projet SkipperDT

## Objectif du projet

Ce projet a pour but de detecter la presence de fourreaux (gaines protectrices de canalisations enterrees) a partir de donnees captees par des drones. Le programme utilise des reseaux de neurones convolutifs (CNN) pour classifier automatiquement les donnees en deux categories : "sans_fourreau" (classe 0) et "avec_fourreau" (classe 1).

Trois architectures de modeles differentes sont entrainees et comparees afin de determiner laquelle offre la meilleure precision pour cette tache de classification binaire.

---

## Structure du projet

```
skipperdt/
├── main.py                         --> Point d'entree principal du programme
├── src/
│   ├── __init__.py                 --> Fichier d init du module Python src
│   ├── config.py                   --> Configuration centrale (hyperparametres, chemins, constantes)
│   ├── donnees.py                  --> Chargement, decoupage et visualisation des donnees
│   ├── entrainement.py             --> Classe Entraineur qui gere l apprentissage des modeles
│   ├── evaluation.py               --> Evaluation des modeles (metriques, graphiques, comparaisons)
│   ├── nettoyage_donnees.py        --> Nettoyage et pre-traitement des donnees brutes
│   └── modeles/
│       ├── __init__.py             --> Expose les classes de modeles et fonctions utilitaires
│       ├── cnn_simple.py           --> Architecture CNN Simple (baseline)
│       ├── cnn_ameliore.py         --> Architecture CNN Ameliore (avec Global Average Pooling)
│       ├── unet.py                 --> Architecture U-Net adaptee pour la classification
│       └── utilitaires.py          --> Fonctions pour creer un modele et compter ses parametres
├── data/
│   ├── brut/                       --> Donnees brutes issues des capteurs du drone
│   └── nettoye/                    --> Donnees nettoyees, pretes pour l entrainement
│       ├── sans_fourreau/          --> Fichiers .npz de la classe "sans fourreau"
│       └── avec_fourreau/          --> Fichiers .npz de la classe "avec fourreau"
├── resultats/                      --> Dossier de sortie pour les graphiques et modeles sauvegardes
└── docs/
    └── documentation.md            --> Ce fichier
```

---

## Etape 0 : Configuration du projet (config.py)

Le fichier `config.py` est le fichier de configuration central. Il est importe par tous les autres modules. Il definit :

### Detection de l appareil de calcul

Le programme detecte automatiquement le meilleur processeur disponible sur la machine :

- Si une carte graphique NVIDIA est presente (CUDA), elle est utilisee car elle permet de paralleliser massivement les calculs matriciels, ce qui accelere l entrainement de maniere significative.
- Si la machine est un Mac avec une puce Apple Silicon (M1, M2, etc.), le backend MPS est utilise pour profiter de l acceleration GPU integree.
- Sinon, le processeur central (CPU) est utilise. C est le plus lent pour l entrainement, mais il fonctionne sur toutes les machines.

### Les chemins

- `RACINE_PROJET` : chemin absolu vers la racine du projet, calcule automatiquement a partir de la position du fichier `config.py`.
- `DOSSIER_DONNEES` : chemin vers le dossier `data/nettoye/` qui contient les donnees nettoyees.
- `DOSSIER_RESULTATS` : chemin vers le dossier `resultats/` ou seront sauvegardes les graphiques et les modeles.

Les deux dossiers sont automatiquement crees s ils n existent pas.

### Les hyperparametres

| Parametre           | Valeur  | Signification                                                                 |
|---------------------|---------|-------------------------------------------------------------------------------|
| TAILLE_IMAGE        | 224     | Les images (matrices de capteurs) sont redimensionnees en 224x224 pixels.     |
| NOMBRE_CANAUX       | 4       | Chaque echantillon a 4 canaux (4 types de mesures differentes du drone).      |
| NOMBRE_CLASSES      | 2       | Classification binaire : "sans_fourreau" (0) ou "avec_fourreau" (1).         |
| TAILLE_LOT          | 64      | Le nombre d echantillons traites en une seule passe (batch size).             |
| NOMBRE_EPOQUES      | 25      | Le nombre maximum de cycles complets sur les donnees d entrainement.          |
| TAUX_APPRENTISSAGE  | 0.001   | La vitesse a laquelle le modele ajuste ses poids a chaque iteration.          |
| PATIENCE            | 5       | Si le modele ne s ameliore pas pendant 5 epoques, l entrainement s arrete.    |

### Les noms de classes

`NOMS_CLASSES = ["sans_fourreau", "avec_fourreau"]` : cette liste associe l index de chaque classe a son nom. L index 0 correspond a "sans_fourreau" et l index 1 a "avec_fourreau".

---

## Etape 1 : Nettoyage des donnees brutes (nettoyage_donnees.py)

Avant de pouvoir entrainer les modeles, il faut nettoyer les donnees brutes. Les fichiers bruts sont des fichiers `.npz` (un format compresse de NumPy) qui contiennent des matrices de mesures captees par les drones. Ces donnees contiennent souvent des valeurs manquantes (NaN) et des zones inutiles.

### Ce que fait le script de nettoyage

La fonction `processus_complet()` parcourt tous les fichiers `.npz` dans le dossier `data/brut/` et, pour chacun :

1. **Chargement** : le fichier `.npz` est ouvert et la matrice de donnees est extraite.

2. **Detection des zones valides** : le programme cree un masque binaire qui identifie quels pixels contiennent des donnees reelles (non-NaN). Il utilise la fonction `findContours` de la librairie OpenCV (cv2) pour detecter les contours de ces zones valides.

3. **Recadrage** : une boite englobante (bounding box) est calculee autour de toutes les zones valides. La matrice est ensuite decoupee pour ne garder que cette zone utile, supprimant ainsi les bords vides ou remplis de NaN.

4. **Remplacement des NaN** : toutes les valeurs NaN restantes a l interieur de la zone recadree sont remplacees par 0.0 grace a la fonction `remplacer_nan_par_zero()`.

5. **Sauvegarde** : la matrice nettoyee est sauvegardee dans le dossier `data/nettoye/` en conservant la meme arborescence de sous-dossiers (les classes `sans_fourreau` et `avec_fourreau`). Le format `.npz` compresse est utilise pour economiser de l espace disque.

### Comment executer le nettoyage

```bash
python src/nettoyage_donnees.py
```

Les chemins source et destination peuvent etre configures via des variables d environnement :
- `DRONE_DATA_BRUT` (par defaut : `../data/brut`)
- `DRONE_DATA_NETTOYE` (par defaut : `../data/nettoye`)

---

## Etape 2 : Chargement et preparation des donnees (donnees.py)

Une fois les donnees nettoyees, le fichier `donnees.py` se charge de les preparer pour l entrainement.

### La classe DatasetDroneNpz

Cette classe herite de `torch.utils.data.Dataset`. Elle permet a PyTorch de charger les donnees de maniere efficace pendant l entrainement. Pour chaque echantillon, voici ce qui se passe lorsque le programme demande une donnee (methode `__getitem__`) :

1. **Chargement du fichier .npz** : la matrice est extraite du fichier compresse.

2. **Gestion des dimensions** : si la matrice est en 2D (un seul canal), une dimension est automatiquement ajoutee pour obtenir un format 3D (Hauteur x Largeur x Canaux).

3. **Conversion en float32** : les donnees sont converties en nombres a virgule flottante de 32 bits, le format standard pour l entrainement de reseaux de neurones.

4. **Normalisation** : les valeurs de la matrice sont mises a l echelle entre 0 et 1 en appliquant la formule `(valeur - minimum) / (maximum - minimum)`. Cette etape est essentielle car les reseaux de neurones convergent mieux avec des valeurs normalisees.

5. **Transformations** : selon que l echantillon appartient a l ensemble d entrainement ou de test, des transformations differentes sont appliquees :

   - **Transformations d entrainement** : redimensionnement en 224x224, retournement horizontal aleatoire, retournement vertical aleatoire, et rotation aleatoire de 15 degres maximum. Ces augmentations artificielles multiplient virtuellement la taille du dataset et aident le modele a generaliser.

   - **Transformations de test** : uniquement le redimensionnement en 224x224. Aucune augmentation n est appliquee car on veut evaluer le modele sur des donnees telles quelles.

### Separation des donnees en trois ensembles

La fonction `charger_donnees()` divise les fichiers disponibles en trois groupes distincts :

| Ensemble       | Proportion approximative | Role                                                        |
|----------------|--------------------------|-------------------------------------------------------------|
| Entrainement   | 70%                      | Les donnees sur lesquelles le modele apprend.               |
| Validation     | 15%                      | Les donnees utilisees pour surveiller l apprentissage.      |
| Test           | 15%                      | Les donnees utilisees pour l evaluation finale du modele.   |

La separation se fait avec la fonction `train_test_split` de scikit-learn. Le parametre `stratify` garantit que chaque ensemble contient la meme proportion de chaque classe. Cela evite, par exemple, que l ensemble de test ne contienne que des echantillons "avec_fourreau" et aucun "sans_fourreau", ce qui fausserait l evaluation.

Le parametre `random_state=42` assure que la separation est identique a chaque execution du programme. Cela permet d obtenir des resultats reproductibles.

Si le dataset contient moins de 5 fichiers, le programme detecte automatiquement que le dataset est trop petit pour etre divise normalement. Dans ce cas, les trois ensembles utilisent les memes donnees pour eviter des erreurs.

### Le DataLoader

Chaque ensemble est ensuite encapsule dans un `DataLoader` PyTorch qui gere :
- Le chargement par lots (batches) de 64 echantillons a la fois.
- Le melange aleatoire des donnees d entrainement a chaque epoque (`shuffle=True`), ce qui evite que le modele memorise l ordre des donnees.
- Les donnees de validation et de test ne sont pas melangees (`shuffle=False`) car l ordre n a pas d impact sur l evaluation.

---

## Etape 3 : Les trois architectures de modeles

Trois modeles de complexite croissante sont proposes. Tous recoivent en entree un tenseur de dimensions (Batch, 4, 224, 224), c est-a-dire un lot d images a 4 canaux de 224x224 pixels. Tous produisent un tenseur de sortie de dimensions (Batch, 2), contenant un score pour chaque classe.

### 3.1 CNN Simple (cnn_simple.py) — Le modele de reference

C est l architecture la plus basique. Elle sert de point de comparaison (baseline).

**Architecture :**

La partie convolutive est composee de 3 blocs identiques empiles. Chaque bloc contient :
- Une couche de convolution 2D (`Conv2d`) avec un noyau de 3x3 pixels. Cette couche applique des filtres qui detectent des motifs locaux dans les donnees (bords, textures, formes).
- Une couche `BatchNorm2d` qui normalise les sorties du bloc. Cela stabilise l apprentissage et permet d utiliser des taux d apprentissage plus eleves.
- Une fonction d activation `ReLU` qui introduit de la non-linearite dans le modele. Sans elle, le reseau ne pourrait apprendre que des relations lineaires.
- Une couche `MaxPool2d` de taille 2 qui divise les dimensions spatiales par 2. Cela reduit la quantite de calculs et force le reseau a se concentrer sur les motifs importants.

Les filtres augmentent progressivement : 4 canaux en entree, puis 32, puis 64, puis 128 filtres. Plus on avance dans le reseau, plus les motifs detectes sont abstraits et complexes.

La partie classification prend les sorties de la partie convolutive et les transforme en predictions :
- `Flatten` : transforme la matrice 3D en un vecteur 1D.
- Trois couches `Linear` (couches denses) avec progressivement moins de neurones : taille_flatten vers 256, puis 64, puis 2 (nombre de classes).
- `Dropout` avec des probabilites de 0.5 et 0.3 : pendant l entrainement, ces couches desactivent aleatoirement une proportion des neurones. Cela empeche le reseau de s appuyer trop fortement sur certains neurones specifiques et ameliore la generalisation.

**Nombre de parametres :** environ 388 000 parametres.

### 3.2 CNN Ameliore (cnn_ameliore.py) — Version optimisee

Ce modele a la meme partie convolutive que le CNN Simple (3 blocs de convolutions avec 32, 64,  et 128 filtres), mais avec deux differences majeures :

1. **Pas de MaxPool apres le troisieme bloc** : la resolution spatiale est reduite par les deux premiers MaxPool seulement.

2. **Global Average Pooling (GAP)** : au lieu d utiliser `Flatten` qui concatene tous les pixels en un long vecteur, le GAP (`AdaptiveAvgPool2d(1)`) calcule la moyenne de chaque carte de caracteristiques et produit un vecteur de taille 128 (le nombre de filtres du dernier bloc). Cette technique a deux avantages :
   - Elle reduit drastiquement le nombre de parametres car on n a plus besoin de grandes couches `Linear`.
   - Elle rend le modele independant de la taille des images en entree.

3. **Partie classification simplifiee** : seulement 2 couches `Linear` (128 vers 64, puis 64 vers 2) avec un seul `Dropout` de 0.3.

**Nombre de parametres :** environ 150 000 parametres, soit environ 2.5 fois moins que le CNN Simple.

### 3.3 U-Net Classification (unet.py) — Architecture avancee

Le U-Net est une architecture plus complexe, composee de 4 sous-modules :

#### BlocConvolution

C est le bloc de base. Il applique deux convolutions 3x3 successives, chacune suivie d un `BatchNorm` et d un `ReLU`. Contrairement aux CNN precedents qui n ont qu une seule convolution par bloc, le double passage permet de mieux extraire les caracteristiques a chaque niveau.

#### Encodeur (partie contractante)

L encodeur contient 4 niveaux. A chaque niveau :
- Un `BlocConvolution` traite les donnees et produit des cartes de caracteristiques.
- Un `MaxPool2d` reduit les dimensions spatiales par 2.
- Les sorties de chaque niveau sont enregistrees dans une liste appelee `sorties_intermediaires`. Ces sorties seront reutilisees plus tard par le decodeur (c est le principe des "skip connections").

Les filtres augmentent progressivement : 4 (entree) vers 32, puis 64, 128, et 256.

#### Bottleneck

C est le point le plus profond du reseau, entre l encodeur et le decodeur. Il contient un `BlocConvolution` qui passe de 256 a 512 filtres. A ce stade, les donnees ont ete reduites a une tres petite taille spatiale mais contiennent une representation tres comprimee et abstraite de l information.

#### Decodeur (partie expansive)

Le decodeur inverse le processus de l encodeur. A chaque niveau :
- Un `ConvTranspose2d` (convolution transposee) double les dimensions spatiales. C est l operation inverse du `MaxPool`.
- La sortie est concatenee avec les `sorties_intermediaires` correspondantes de l encodeur (skip connection). Cette concatenation permet au decodeur d avoir acces a la fois aux details fins (provenant de l encodeur) et aux informations globales (provenant des niveaux plus profonds).
- Un `BlocConvolution` traite le resultat de la concatenation.

Les skip connections sont l innovation principale du U-Net : elles permettent au reseau de capturer simultanement des informations a differentes echelles, ce qui est particulierement utile pour des donnees de capteurs ou l information utile peut etre a differentes resolutions.

#### Tete de classification

Apres le decodeur, un `AdaptiveAvgPool2d(1)` (Global Average Pooling) comprime les cartes de caracteristiques en un vecteur, suivi de couches `Linear` pour produire les 2 scores de classification.

---

## Etape 4 : Entrainement des modeles (entrainement.py)

La classe `Entraineur` gere tout le processus d apprentissage. Elle est initialisee avec un modele, un chargeur de donnees d entrainement et un chargeur de validation.

### Initialisation

Lors de la creation d un objet `Entraineur`, les elements suivants sont configures :

- **Le modele** est transfere sur l appareil de calcul (GPU ou CPU).
- **La fonction de perte** : `CrossEntropyLoss`. Cette fonction mesure l ecart entre les predictions du modele et les vraies etiquettes. Elle combine deux operations : appliquer un `softmax` (qui convertit les scores bruts en probabilites entre 0 et 1) et calculer l entropie croisee (qui penalise les mauvaises predictions). Plus la prediction est eloignee de la realite, plus la perte est elevee.
- **L optimiseur** : `Adam` avec un taux d apprentissage de 0.001. Adam est un optimiseur adaptatif qui ajuste automatiquement le taux d apprentissage pour chaque parametre du reseau. Il est generalement plus efficace que l optimiseur classique SGD (Stochastic Gradient Descent).
- **Le planificateur de taux d apprentissage** : `ReduceLROnPlateau`. Si la perte de validation ne s ameliore pas pendant 3 epoques, le taux d apprentissage est divise par 2 (facteur = 0.5). Cela permet au modele de faire des ajustements plus fins quand il approche d un minimum de perte.
- **L historique** : un dictionnaire qui enregistre la perte et la precision a chaque epoque, pour le train et la validation.

### Phase d entrainement (methode _entrainer_une_epoque)

Voici ce qui se passe concretement a chaque epoque d entrainement :

1. Le modele passe en mode entrainement via `modele.train()`. Cela active les couches `Dropout` (qui desactivent aleatoirement des neurones) et `BatchNorm` (qui calcule les statistiques sur le batch courant). Ces mecanismes ne sont actifs que pendant l entrainement.

2. Pour chaque lot de 64 images :

   a. Les images et les etiquettes sont transferees sur l appareil de calcul (GPU ou CPU).

   b. Les gradients des parametres sont remis a zero avec `optimiseur.zero_grad()`. C est necessaire car par defaut PyTorch accumule les gradients a chaque appel de `backward()`.

   c. Les images sont passees dans le modele (propagation avant, ou "forward pass"). Le modele produit un tenseur de taille (64, 2) contenant un score pour chaque classe et chaque image du lot.

   d. La perte est calculee en comparant les scores predits avec les vraies etiquettes. La fonction `CrossEntropyLoss` produit un scalaire qui represente l erreur moyenne sur le lot.

   e. La retropropagation est declenchee avec `perte.backward()`. PyTorch remonte le graphe de calculs et calcule le gradient de la perte par rapport a chaque parametre du modele. Le gradient indique dans quelle direction et de combien chaque poids devrait changer pour reduire la perte.

   f. L optimiseur met a jour tous les poids du modele avec `optimiseur.step()`, en utilisant les gradients calcules a l etape precedente. Chaque poids est legerement ajuste dans la direction qui reduit la perte.

   g. On accumule les statistiques du lot : la perte ponderee par la taille du lot, et le nombre de bonnes predictions (ou la classe predite correspond a la vraie etiquette).

3. A la fin de l epoque, on calcule la perte moyenne et la precision moyenne sur tous les lots.

### Phase de validation (methode _valider)

Cette phase intervient apres chaque epoque d entrainement. Elle est tres similaire a la phase d entrainement, mais avec des differences fondamentales :

1. Le modele passe en mode evaluation via `modele.eval()`. Cela desactive `Dropout` (tous les neurones sont actifs, le modele utilise toute sa capacite) et `BatchNorm` utilise les statistiques calculees pendant l entrainement au lieu de celles du lot courant.

2. Le decorateur `@torch.no_grad()` est utilise pour indiquer a PyTorch de ne pas calculer les gradients. Cela economise de la memoire et accelere les calculs.

3. Les donnees de validation passent dans le modele exactement comme pendant le train (propagation avant), et la perte et la precision sont calculees.

4. Aucune modification des poids n est effectuee. Il n y a pas de `backward()` ni de `optimiseur.step()`.

**Le but de la validation est de savoir si le modele se comporte bien sur des donnees qu il n a jamais vues pendant l entrainement.** Si la perte d entrainement diminue mais que la perte de validation augmente, cela signifie que le modele est en train de memoriser les donnees d entrainement au lieu d apprendre des regles generales. Ce phenomene s appelle le surapprentissage (overfitting).

### Mecanisme d arret premature (Early Stopping)

A chaque epoque, apres la validation, le programme verifie si la perte de validation s est amelioree :

- **Si oui** : c est le meilleur modele jusqu a present. Les poids du modele sont sauvegardes en memoire via `copy.deepcopy`. Le compteur de patience est remis a zero.
- **Si non** : le compteur de patience est incremente de 1. Si le compteur atteint la valeur de `PATIENCE` (5 dans notre configuration), l entrainement est arrete prematurement. Cela evite de continuer a entrainer un modele qui ne s ameliore plus.

A la fin de l entrainement (que ce soit par early stopping ou apres les 25 epoques), les meilleurs poids sauvegardes sont recharges dans le modele avec `load_state_dict`. Le modele est aussi sauvegarde sur le disque dans le fichier `resultats/meilleur_modele.pth`.

---

## Etape 5 : Evaluation des modeles (evaluation.py)

L evaluation se fait une seule fois par modele, apres son entrainement complet.

### Evaluation sur l ensemble de test (evaluer_modele)

Le modele est place en mode evaluation. Toutes les donnees de test sont passees dans le modele, lot par lot. Pour chaque lot :
- Le modele produit des scores pour chaque classe.
- La classe predite est celle avec le score le plus eleve (obtenu via `torch.max`).
- Les predictions et les vraies etiquettes sont collectees dans des listes.
- Le nombre de bonnes predictions est compte.

La precision finale est calculee comme le ratio de bonnes predictions sur le nombre total d echantillons.

### Courbes d apprentissage (afficher_courbes_apprentissage)

Cette fonction genere un graphique a deux panneaux :

- **Panneau gauche - Perte (Loss)** : montre l evolution de la perte sur l ensemble d entrainement (en bleu) et de validation (en rouge) a chaque epoque. Idealement, les deux courbes devraient diminuer ensemble. Si la courbe de validation remonte alors que celle d entrainement continue de descendre, c est un signe de surapprentissage.

- **Panneau droit - Precision (Accuracy)** : montre l evolution de la precision sur les memes ensembles. Idealement, les deux courbes devraient augmenter ensemble et se stabiliser.

Le graphique est sauvegarde dans `resultats/courbes_apprentissage.png`.

### Matrice de confusion (afficher_matrice_confusion)

La matrice de confusion est un tableau 2x2 qui resume les predictions :

|                        | Predit: sans_fourreau | Predit: avec_fourreau |
|------------------------|-----------------------|-----------------------|
| Reel: sans_fourreau    | Vrais Negatifs (VN)   | Faux Positifs (FP)    |
| Reel: avec_fourreau    | Faux Negatifs (FN)    | Vrais Positifs (VP)   |

- **Vrais Negatifs** : le modele a correctement predit "sans_fourreau".
- **Vrais Positifs** : le modele a correctement predit "avec_fourreau".
- **Faux Positifs** : le modele a predit "avec_fourreau" mais c etait en realite "sans_fourreau".
- **Faux Negatifs** : le modele a predit "sans_fourreau" mais c etait en realite "avec_fourreau".

Le rapport de classification genere par scikit-learn fournit pour chaque classe :
- **Precision** : parmi les predictions d une classe, combien sont correctes.
- **Rappel** : parmi les echantillons reels d une classe, combien ont ete correctement identifies.
- **F1-Score** : la moyenne harmonique de la precision et du rappel.

### Exemples de predictions (afficher_exemples_predictions)

Cette fonction prend un lot de donnees de test, genere les predictions du modele, et affiche les images avec :
- En titre, la vraie classe et la classe predite.
- En vert si la prediction est correcte, en rouge si elle est fausse.

Cela permet de voir visuellement sur quels types d echantillons le modele se trompe.

---

## Etape 6 : Comparaison finale et selection du meilleur modele (main.py)

### Pipeline comparatif

La fonction `pipeline_comparatif()` dans `main.py` orchestre tout le processus :

1. Les donnees sont chargees une seule fois et partagees entre les trois modeles. Cela garantit que tous les modeles voient exactement les memes donnees d entrainement, de validation et de test, rendant la comparaison equitable.

2. Chaque modele ("cnn_simple", "cnn_ameliore", "unet") est cree, entraine et evalue via la fonction `entrainer_un_modele()`.

3. Les resultats de chaque modele (precision et nombre de parametres) sont stockes dans un dictionnaire.

4. La fonction `comparer_modeles()` genere un graphique comparatif a deux panneaux :
   - **Precision sur le test** : un diagramme en barres montrant la precision de chaque modele en pourcentage.
   - **Taille du modele** : un diagramme en barres montrant le nombre de parametres de chaque modele.

   Ce graphique permet de visualiser le compromis entre la performance et la complexite de chaque modele.

5. Le meilleur modele est automatiquement identifie (celui avec la precision la plus elevee), et des exemples de ses predictions sont affiches.

### Comment lancer le programme

```bash
python main.py
```

Cette commande execute l integralite du pipeline : chargement des donnees, entrainement des trois modeles, evaluation, et comparaison.

---

## Resume du flux complet

```
Donnees brutes (.npz)
        |
        v
[Etape 1] Nettoyage (nettoyage_donnees.py)
   - Suppression des zones vides
   - Remplacement des NaN par 0
   - Recadrage autour des zones valides
        |
        v
Donnees nettoyees (data/nettoye/)
        |
        v
[Etape 2] Chargement et preparation (donnees.py)
   - Lecture des fichiers .npz
   - Normalisation entre 0 et 1
   - Augmentation des donnees (train uniquement)
   - Separation en Train (70%), Validation (15%), Test (15%)
        |
        v
[Etape 3] Creation des modeles (modeles/)
   - CNN Simple (388K parametres)
   - CNN Ameliore (150K parametres)
   - U-Net Classification
        |
        v
[Etape 4] Entrainement (entrainement.py)
   - Boucle de 25 epoques maximum
   - Propagation avant + retropropagation + mise a jour des poids
   - Validation apres chaque epoque
   - Arret premature si pas d amelioration pendant 5 epoques
   - Sauvegarde automatique du meilleur modele
        |
        v
[Etape 5] Evaluation (evaluation.py)
   - Precision sur l ensemble de test
   - Courbes d apprentissage
   - Matrice de confusion
   - Rapport de classification
        |
        v
[Etape 6] Comparaison finale (main.py)
   - Graphique comparatif des 3 modeles
   - Selection automatique du meilleur modele
   - Affichage d exemples de predictions
```

---

## Fichiers de sortie

Tous les fichiers generes sont sauvegardes dans le dossier `resultats/` :

| Fichier                          | Description                                                 |
|----------------------------------|-------------------------------------------------------------|
| courbes_apprentissage.png        | Courbes de perte et precision par epoque                    |
| matrice_confusion.png            | Matrice de confusion du dernier modele evalue               |
| exemples_predictions.png         | Exemples visuels de predictions correctes et incorrectes    |
| distribution_classes.png         | Repartition des classes dans les ensembles train et test    |
| comparaison_modeles.png          | Graphique comparatif des 3 modeles (precision + parametres) |
| meilleur_modele.pth              | Poids du meilleur modele sauvegarde                         |

---

## Dependances

Le projet utilise les librairies Python suivantes :

| Librairie         | Usage dans le projet                                              |
|-------------------|-------------------------------------------------------------------|
| torch (PyTorch)   | Construction et entrainement des reseaux de neurones              |
| torchvision       | Transformations d images (redimensionnement, augmentations)       |
| numpy             | Manipulation de matrices et calculs numeriques                    |
| matplotlib        | Generation de graphiques et visualisations                        |
| scikit-learn      | Separation des donnees, matrice de confusion, rapport de classif  |
| opencv (cv2)      | Detection de contours pour le nettoyage des donnees               |
| tqdm              | Barres de progression pendant le nettoyage et l entrainement      |
