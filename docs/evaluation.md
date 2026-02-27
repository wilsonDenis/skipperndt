# Questions et Reponses -- Projet SkipperNDT

Tout ce qu'il faut savoir sur le projet, explique sous forme de questions-reponses.


---


# PARTIE 1 : LE PROJET ET LES DONNEES


## Q1 : C'est quoi le projet SkipperNDT ?

C'est un projet de detection de fourreaux (conduites/tuyaux souterrains) a l'aide de l'intelligence artificielle. Un drone survole le sol avec un capteur magnetique. Le capteur mesure l'intensite du champ magnetique a differents points. Ces mesures sont stockees sous forme de tableaux 2D (comme des images). Notre modele d'IA analyse ces "images magnetiques" pour repondre a deux questions :
- **Y a-t-il un fourreau ?** (classification : oui ou non)
- **Quelle est sa largeur ?** (regression : un nombre en metres)


## Q2 : C'est quoi un fichier `.npz` ?

C'est un format de fichier numpy (librairie Python pour le calcul numerique). Il contient un tableau de nombres (comme un tableur Excel, mais en plusieurs dimensions). Dans notre cas, chaque fichier `.npz` contient un tableau de forme `(hauteur, largeur, 4)` qui represente les mesures du champ magnetique. Le "4" signifie qu'il y a 4 couches d'information (4 canaux) pour chaque point de mesure.


## Q3 : Pourquoi 4 canaux et pas 3 comme une image normale ?

Une image couleur classique (photo) utilise 3 canaux : Rouge, Vert, Bleu (RGB). Ici, les 4 canaux representent 4 composantes differentes du champ magnetique mesure par le drone. Ce ne sont pas des couleurs, mais des intensites physiques. C'est pour cela qu'on configure `NOMBRE_CANAUX = 4` dans notre code.


## Q4 : C'est quoi les NaN dans les donnees ?

NaN signifie "Not a Number" (pas un nombre). Dans nos donnees, les zones du terrain qui sont trop eloignees du fourreau n'emettent quasiment pas de signal magnetique. Ces zones sont marquees comme NaN car il n'y a rien a mesurer. C'est un comportement physique normal. Dans le code, on les remplace par des zeros :

```python
matrice = np.nan_to_num(matrice, nan=0.0)
```

`np.nan_to_num` parcourt tout le tableau et remplace chaque NaN par 0.0.


## Q5 : Quelles sont les deux types de donnees ?

1. **Donnees synthetiques** (2833 echantillons) : generees par ordinateur (simulation). Elles imitent ce que le drone capterait dans differentes conditions. On en a beaucoup.
2. **Donnees reelles** (102 echantillons) : mesurees sur le terrain par de vrais drones. Elles sont peu nombreuses mais representent la realite.


## Q6 : Combien d'echantillons de chaque type ?

| Type | Avec fourreau (label=1) | Sans fourreau (label=0) | Total |
|------|:-:|:-:|:-:|
| Synthetiques | 1700 | 1133 | 2833 |
| Reelles | 51 | 51 | 102 |
| **Total** | **1751** | **1184** | **2935** |


## Q7 : C'est quoi le fichier CSV et a quoi sert-il ?

Le fichier `pipe_presence_width_detection_label.csv` est un tableau (separateur `;`) qui contient les informations de chaque echantillon :

```
field_file;label;width_m;coverage_type;shape;noisy;noise_type;pipe_type
sample_00000_perfect_straight_clean_field.npz;1;62.4;perfect;straight;False;N/A;single
```

- `field_file` : nom du fichier `.npz`
- `label` : 1 (fourreau present) ou 0 (pas de fourreau)
- `width_m` : largeur du fourreau en metres (N/A si pas de fourreau)

Le code le lit avec :
```python
df = pd.read_csv(FICHIER_CSV, sep=';', keep_default_na=False)
```

Le `keep_default_na=False` empeche pandas de transformer le texte "N/A" en NaN. On veut garder "N/A" comme texte pour le gerer nous-memes.


## Q8 : Comment le code trouve-t-il les fichiers .npz sur le disque ?

La fonction `_resoudre_chemin_fichier` prend le nom du fichier (ex: `sample_00042_...npz`) et cherche s'il existe dans l'un des dossiers possibles :

```python
def _resoudre_chemin_fichier(nom_fichier):
    if nom_fichier.startswith('real_data'):
        chemin = os.path.join(DOSSIER_DONNEES_REELLES, nom_fichier)  # real_data/
    else:
        chemin = os.path.join(DOSSIER_DONNEES, 'avec_fourreau', nom_fichier)  # data/nettoye/avec_fourreau/
        if not os.path.exists(chemin):
            chemin = os.path.join(DOSSIER_DONNEES, 'sans_fourreau', nom_fichier)  # data/nettoye/sans_fourreau/
```

Les fichiers synthetiques sont dans `data/nettoye/avec_fourreau/` ou `data/nettoye/sans_fourreau/`. Les fichiers reels sont dans `real_data/`.


---


# PARTIE 2 : LA PREPARATION DES DONNEES


## Q9 : C'est quoi la normalisation et pourquoi normaliser ?

La normalisation consiste a mettre toutes les valeurs entre 0 et 1. Sans normalisation, les mesures magnetiques pourraient avoir des valeurs tres differentes d'un echantillon a l'autre (ex: un fichier avec des valeurs de 0 a 1000, un autre de 0 a 50). Le modele apprendrait mal car il serait perturbe par ces echelles differentes.

La formule de normalisation Min-Max :

```
valeur_normalisee = (valeur - minimum) / (maximum - minimum)
```

Dans le code :
```python
mat_min, mat_max = matrice.min(), matrice.max()
if mat_max > mat_min:
    matrice = (matrice - mat_min) / (mat_max - mat_min)
```

Exemple concret : si les valeurs vont de 200 a 800, la valeur 500 devient (500-200)/(800-200) = 300/600 = 0.5.


## Q10 : C'est quoi le redimensionnement et pourquoi ?

Chaque fichier `.npz` a une taille differente (150x200, 3000x4000...). Or un reseau de neurones a besoin que toutes les entrees aient **exactement la meme taille**. On redimensionne donc toutes les images a 224x224 pixels.

```python
transforms.Resize((224, 224))
```

224x224 est une taille standard en deep learning, assez grande pour garder les details importants, assez petite pour calculer rapidement.


## Q11 : C'est quoi l'augmentation de donnees ?

L'augmentation consiste a appliquer des transformations aleatoires aux images pendant l'entrainement pour creer artificiellement plus de variations. A chaque epoque, la meme image apparait un peu differemment.

```python
transformation_entrainement = transforms.Compose([
    transforms.ToTensor(),              # Convertit en format PyTorch
    transforms.Resize((224, 224)),      # Redimensionne
    transforms.RandomHorizontalFlip(), # Retourne horizontalement (50% de chance)
    transforms.RandomVerticalFlip(),   # Retourne verticalement (50% de chance)
    transforms.RandomRotation(15),     # Rotation aleatoire entre -15 et +15 degres
])
```

Cela empeche le modele de memoriser les images par coeur et le force a apprendre les vrais patterns (formes, intensites) plutot que la position exacte des pixels.


## Q12 : C'est quoi le surechantillonnage et pourquoi x20 ?

Le surechantillonnage (oversampling) consiste a repeter les donnees reelles plusieurs fois dans le jeu d'entrainement.

**Le probleme** : on a 1982 echantillons synthetiques mais seulement 71 reels dans le train. Les donnees reelles representent 3.5% du total. Le modele ne les voit presque pas et n'apprend pas a les reconnaitre.

**La solution** : on repete les 71 echantillons reels 20 fois = 1420 copies dans le train.

```python
facteur_surechantillonnage = 20
idx_train_reel_augmente = np.tile(idx_train_reel, facteur_surechantillonnage)
```

`np.tile` repete un tableau. `np.tile([A, B, C], 3)` donne `[A, B, C, A, B, C, A, B, C]`.

Comme chaque copie subit des augmentations aleatoires differentes (flips, rotations), le modele voit 1420 versions differentes des donnees reelles, pas 1420 copies identiques.

**Impact** : sans surechantillonnage, 50% de precision sur le reel. Avec x20, **100%**.


## Q13 : Comment les donnees sont-elles separees (train/val/test) ?

On utilise `train_test_split` de scikit-learn pour diviser les donnees :

```python
# Etape 1 : On prend 85% pour temp et 15% pour test
idx_temp, idx_test, _, _ = train_test_split(indices, labels, test_size=0.15, stratify=labels)

# Etape 2 : Parmi les 85%, on prend 82.35% pour train et 17.65% pour val
idx_train, idx_val, _, _ = train_test_split(idx_temp, labels_temp, test_size=0.1765, stratify=labels)
```

Le parametre `stratify=labels` garantit que chaque split a la meme proportion de classes que l'ensemble original. `random_state=42` rend la separation reproductible (meme resultat a chaque lancement).

| Split | Synth | Reel (x20) | Total |
|-------|:-----:|:----------:|:-----:|
| Train | 1982 | 1420 | 3402 |
| Val | 426 | 0 | 426 |
| Test synth | 425 | 0 | 425 |
| Test reel | 0 | 31 | 31 |


## Q14 : C'est quoi un DataLoader et pourquoi batch_size=64 ?

Un `DataLoader` de PyTorch groupe les echantillons en "lots" (batches) et les distribue au modele pendant l'entrainement.

```python
chargeur_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
```

- `batch_size=64` : le modele traite 64 images a la fois (pas une par une, ni toutes d'un coup)
- `shuffle=True` : melange les donnees a chaque epoque pour que le modele ne voie pas toujours les memes sequences

Pourquoi 64 et pas 1 ou 3000 ?
- Si batch_size=1 : l'apprentissage est tres lent et instable (trop de bruit)
- Si batch_size=3000 : ca ne rentre pas en memoire GPU
- 64 est un bon compromis entre vitesse, stabilite et memoire


## Q15 : C'est quoi un Dataset en PyTorch ?

C'est une classe Python qui definit comment acceder aux donnees. Elle doit definir deux methodes :

```python
class DatasetMultiTask(Dataset):
    def __len__(self):
        return len(self.chemins_fichiers)  # Combien d'echantillons au total

    def __getitem__(self, idx):
        # Comment charger un echantillon a l'index idx
        matrice = np.load(self.chemins_fichiers[idx])  # Charge le fichier
        # ... normalise, redimensionne ...
        return matrice, etiquette, largeur  # Retourne image + label + largeur
```

Le DataLoader appelle `__getitem__` automatiquement pour chaque index dans le batch.


---


# PARTIE 3 : L'ARCHITECTURE DU MODELE


## Q16 : C'est quoi un CNN (Convolutional Neural Network) ?

Un CNN est un type de reseau de neurones specialise dans l'analyse d'images. Il utilise des "filtres" (aussi appeles "noyaux de convolution") qui glissent sur l'image pour detecter des motifs : contours, textures, formes.

Imagine une loupe qui scanne l'image de gauche a droite, de haut en bas. Chaque filtre detecte un type de motif specifique. Les premiers filtres detectent des choses simples (lignes, bords). Les filtres plus profonds detectent des choses complexes (formes, patterns).


## Q17 : C'est quoi une couche Conv2d ?

`Conv2d` applique un filtre de convolution 2D sur une image.

```python
nn.Conv2d(4, 32, kernel_size=3, padding=1)
```

- `4` : nombre de canaux en entree (nos 4 canaux magnetiques)
- `32` : nombre de filtres en sortie (32 types de motifs differents a detecter)
- `kernel_size=3` : taille du filtre = 3x3 pixels (le filtre regarde 9 pixels a la fois)
- `padding=1` : ajoute 1 pixel de bordure autour de l'image pour que la sortie ait la meme taille que l'entree

Mathematiquement, pour chaque position (i,j) de l'image :
```
sortie(i,j) = somme( filtre * zone_3x3_autour_de(i,j) ) + biais
```


## Q18 : C'est quoi BatchNorm2d et a quoi ca sert ?

`BatchNorm2d` normalise les valeurs de sortie de chaque couche pour qu'elles aient une moyenne proche de 0 et un ecart-type proche de 1.

```python
nn.BatchNorm2d(32)  # Normalise les 32 canaux de sortie
```

Sans BatchNorm, les valeurs entre les couches peuvent devenir tres grandes ou tres petites au fil de l'entrainement, ce qui rend l'apprentissage instable. BatchNorm stabilise et accelere l'entrainement.

Formule : `sortie = (valeur - moyenne_du_batch) / ecart_type_du_batch`


## Q19 : C'est quoi ReLU ?

ReLU (Rectified Linear Unit) est une fonction d'activation. Elle remplace les valeurs negatives par zero et garde les positives telles quelles.

```python
nn.ReLU()
```

Formule : `ReLU(x) = max(0, x)`

Exemples : ReLU(-5) = 0, ReLU(3) = 3, ReLU(-0.1) = 0, ReLU(100) = 100

Pourquoi ? Sans fonction d'activation, un reseau de neurones ne pourrait apprendre que des relations lineaires (des lignes droites). ReLU introduit de la non-linearite, permettant au modele d'apprendre des patterns complexes.


## Q20 : C'est quoi MaxPool2d ?

`MaxPool2d(2)` divise l'image en carres de 2x2 pixels et ne garde que la valeur maximale de chaque carre.

```python
nn.MaxPool2d(2)
```

Exemple pour un carre 2x2 :
```
[3, 7]
[1, 5]   -> Max = 7
```

Effet : l'image passe de 224x224 a 112x112 (taille divisee par 2). Cela :
1. Reduit le nombre de calculs
2. Force le modele a se concentrer sur les informations les plus importantes
3. Rend le modele plus robuste aux petits decalages dans l'image


## Q21 : A quoi sert chaque bloc convolutif ?

Notre CNN a 3 blocs :

```python
# Bloc 1 : 4 -> 32 filtres (entree: 224x224, sortie: 112x112)
nn.Conv2d(4, 32, kernel_size=3, padding=1)   # Detecte des motifs simples
nn.BatchNorm2d(32)                            # Stabilise
nn.ReLU()                                     # Active
nn.MaxPool2d(2)                               # Reduit 224 -> 112

# Bloc 2 : 32 -> 64 filtres (entree: 112x112, sortie: 56x56)
nn.Conv2d(32, 64, kernel_size=3, padding=1)   # Detecte des motifs moyens
nn.BatchNorm2d(64)
nn.ReLU()
nn.MaxPool2d(2)                               # Reduit 112 -> 56

# Bloc 3 : 64 -> 128 filtres (entree: 56x56, sortie: 28x28)
nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Detecte des motifs complexes
nn.BatchNorm2d(128)
nn.ReLU()
nn.MaxPool2d(2)                               # Reduit 56 -> 28
```

A la fin, une image de 224x224 pixels avec 4 canaux est devenue un tableau de 28x28 avec 128 canaux = 100,352 valeurs.


## Q22 : C'est quoi Flatten et a quoi ca sert ?

`nn.Flatten()` prend un tableau multi-dimensionnel et l'aplatit en un vecteur 1D.

```python
nn.Flatten()  # (batch, 128, 28, 28) -> (batch, 100352)
```

C'est necessaire car les couches `Linear` (couches denses) ne travaillent qu'avec des vecteurs 1D, pas des images 3D.


## Q23 : C'est quoi une couche Linear ?

`nn.Linear(entree, sortie)` est une couche "dense" ou "fully connected". Chaque neurone de sortie est connecte a TOUS les neurones d'entree.

```python
nn.Linear(100352, 256)  # 100352 entrees -> 256 sorties
```

Mathematiquement : `sortie = poids * entree + biais`

C'est une multiplication de matrice. Les 100,352 valeurs sont transformees en 256 valeurs essentielles qui resument l'information de l'image.


## Q24 : C'est quoi Dropout et pourquoi 0.5 ?

`nn.Dropout(0.5)` desactive aleatoirement 50% des neurones pendant l'entrainement.

```python
nn.Dropout(0.5)  # 50% des neurones sont mis a zero aleatoirement
```

Imagine une equipe de 10 joueurs. Pendant l'entrainement, on retire 5 joueurs aleatoirement a chaque match. Les joueurs restants doivent compenser et devenir plus polyvalents. Au match final (en evaluation), tous les 10 jouent ensemble et sont chacun plus forts.

Cela empeche le **surapprentissage** (overfitting) : le modele n'apprend pas par coeur, il generalise.

Pendant l'evaluation (`model.eval()`), Dropout est desactive automatiquement.


## Q25 : C'est quoi le Multi-Task Learning ?

C'est l'idee d'entrainer UN SEUL modele pour faire DEUX taches en meme temps. Dans notre cas :

```python
def forward(self, tenseur_entree):
    # 1. L'image passe par les couches convolutives (partie partagee)
    caracteristiques = self.couches_convolution(tenseur_entree)
    features = self.features_partagees(caracteristiques)

    # 2. Les memes features vont dans DEUX directions differentes
    sortie_classification = self.tete_classification(features)  # -> [score_0, score_1]
    sortie_regression = self.tete_regression(features).squeeze(-1)  # -> largeur_en_metres

    return sortie_classification, sortie_regression
```

Avantage : la regression aide la classification et vice-versa. Le modele apprend de meilleurs features car il doit satisfaire deux objectifs a la fois.


## Q26 : Que retourne le modele exactement ?

Pour un batch de 64 images, la methode `forward()` retourne :

1. `sortie_classification` : un tenseur de taille `(64, 2)` -- pour chaque image, 2 scores (un pour "sans_fourreau", un pour "avec_fourreau"). La classe predite est celle avec le score le plus eleve.
2. `sortie_regression` : un tenseur de taille `(64,)` -- pour chaque image, un nombre (la largeur predite en metres).

```python
classif, reg = modele(images)
# classif.shape = (64, 2)
# reg.shape = (64,)
```


## Q27 : C'est quoi `.squeeze(-1)` dans la tete de regression ?

La derniere couche `nn.Linear(64, 1)` produit un tenseur de forme `(64, 1)` (64 images, 1 valeur chacune). Le `.squeeze(-1)` enleve cette dimension inutile pour obtenir `(64,)` qui est plus simple a manipuler.

```python
# Avant squeeze : tensor([[25.3], [42.1], [10.5]])  -> forme (3, 1)
# Apres squeeze : tensor([25.3, 42.1, 10.5])        -> forme (3,)
```


## Q28 : Combien de parametres a le modele et c'est quoi un parametre ?

Le modele a **25,817,443 parametres** (~25.8 millions). Un parametre est une valeur numerique que le modele ajuste pendant l'entrainement pour faire de meilleures predictions. Ce sont les "poids" des connexions entre les neurones.

```python
total = sum(p.numel() for p in modele.parameters())  # 25,817,443
```

Chaque filtre de convolution, chaque poids de couche dense, chaque biais est un parametre. Plus il y a de parametres, plus le modele peut apprendre des patterns complexes, mais aussi plus il risque le surapprentissage.


---


# PARTIE 4 : L'ENTRAINEMENT


## Q29 : C'est quoi une fonction de perte (loss function) ?

Une fonction de perte mesure a quel point les predictions du modele sont eloignees de la verite. Plus la perte est basse, mieux le modele predit. L'objectif de l'entrainement est de **minimiser cette perte**.


## Q30 : C'est quoi CrossEntropyLoss et comment ca marche ?

`CrossEntropyLoss` est la fonction de perte pour la **classification**. Elle compare les scores predits avec le vrai label.

```python
perte_classification = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0]))
```

Si le modele predit `[0.1, 0.9]` pour une image avec fourreau (label=1), la perte est faible (bonne prediction). S'il predit `[0.8, 0.2]` pour la meme image, la perte est forte (mauvaise prediction).

La formule utilise le logarithme : `perte = -log(probabilite_de_la_bonne_classe)`. Quand la probabilite est proche de 1 (bonne prediction), `-log(1) = 0` (perte nulle). Quand elle est proche de 0 (mauvaise prediction), `-log(0) = infini` (perte enorme).


## Q31 : Pourquoi des poids de classes `[1.5, 1.0]` ?

Le dataset est desequilibre : 1700 "avec_fourreau" vs 1133 "sans_fourreau" (60% vs 40%). Sans correction, le modele aurait tendance a predire plus souvent "avec_fourreau" car c'est la classe majoritaire.

```python
poids = torch.tensor([1.5, 1.0])
perte_classification = nn.CrossEntropyLoss(weight=poids)
```

Le poids `1.5` sur "sans_fourreau" (classe 0) signifie : "chaque erreur sur une image sans fourreau compte 1.5 fois plus qu'une erreur sur une image avec fourreau". Cela force le modele a faire attention aux deux classes de maniere equilibree.


## Q32 : C'est quoi MSELoss et comment ca marche ?

`MSELoss` (Mean Squared Error) est la fonction de perte pour la **regression**. Elle mesure l'ecart au carre entre la valeur predite et la valeur reelle.

```python
perte_regression = nn.MSELoss()
```

Formule : `MSE = (prediction - realite)^2`

Exemples :
- Prediction = 30m, Realite = 25m : MSE = (30-25)^2 = 25
- Prediction = 30m, Realite = 10m : MSE = (30-10)^2 = 400

Le carre penalise beaucoup les grosses erreurs : une erreur de 20m est 16 fois plus penalisee qu'une erreur de 5m.


## Q33 : Pourquoi la regression n'est calculee que sur les echantillons avec fourreau ?

Si un echantillon n'a PAS de fourreau, il n'y a pas de largeur a predire. Ca n'aurait aucun sens de penaliser le modele pour avoir predit "15m" quand il n'y a pas de fourreau.

```python
masque_pipe = (etiquettes == 1)  # [True, False, True, True, False...]
if masque_pipe.sum() > 0:
    perte_reg = self.perte_regression(
        sorties_reg[masque_pipe],   # Predictions UNIQUEMENT des echantillons avec pipe
        largeurs[masque_pipe]       # Largeurs reelles correspondantes
    )
```

Le `masque_pipe` est un tableau de True/False. `sorties_reg[masque_pipe]` ne garde que les valeurs dont le masque est True.


## Q34 : Comment la perte totale est-elle calculee ?

```python
perte_totale = perte_classification + 0.01 * perte_regression
```

Les deux pertes sont combinees. Le coefficient `0.01` (LAMBDA_REGRESSION) est necessaire car :
- `perte_classification` donne des valeurs ~0.1 a 1.0
- `perte_regression` (MSE) donne des valeurs ~100 a 1000

Sans le coefficient 0.01, la regression dominerait et le modele ignorerait la classification. Le 0.01 equilibre les deux :
```
perte_totale = 0.5 (classif) + 0.01 * 400 (MSE) = 0.5 + 4.0 = 4.5
```


## Q35 : C'est quoi l'optimiseur Adam ?

Adam (Adaptive Moment Estimation) est l'algorithme qui ajuste les parametres du modele pour reduire la perte.

```python
optimiseur = torch.optim.Adam(modele.parameters(), lr=0.001)
```

A chaque batch, Adam :
1. Calcule le gradient (la direction dans laquelle chaque parametre devrait bouger pour reduire la perte)
2. Ajuste chaque parametre en fonction de ce gradient
3. Adapte la vitesse d'ajustement pour chaque parametre individuellement

Le `lr=0.001` (learning rate) controle la taille des pas. C'est comme descendre une montagne : pas trop grands (on risque de sauter par-dessus la vallee), pas trop petits (on n'arrivera jamais en bas).


## Q36 : C'est quoi un gradient et la backpropagation ?

Le **gradient** est la derivee de la perte par rapport a chaque parametre. Il indique dans quelle direction et de combien chaque parametre doit changer pour diminuer la perte.

La **backpropagation** (retropropagation) est l'algorithme qui calcule ces gradients en remontant de la sortie vers l'entree du reseau.

Dans le code :
```python
perte.backward()      # Calcule les gradients (backpropagation)
optimiseur.step()     # Ajuste les parametres en utilisant les gradients
optimiseur.zero_grad()  # Remet les gradients a zero pour le prochain batch
```


## Q37 : C'est quoi ReduceLROnPlateau ?

C'est un planificateur qui reduit automatiquement le learning rate quand le modele stagne.

```python
planificateur = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiseur, mode='min', factor=0.5, patience=3
)
```

- `mode='min'` : on surveille une valeur a minimiser (la perte de validation)
- `factor=0.5` : quand on reduit, on divise le learning rate par 2
- `patience=3` : on attend 3 epoques sans amelioration avant de reduire

C'est comme un marcheur qui fait d'abord de grands pas, puis des petits pas quand il s'approche du but pour ne pas le depasser.


## Q38 : C'est quoi l'Early Stopping ?

L'Early Stopping arrete l'entrainement quand le modele ne s'ameliore plus, pour eviter le surapprentissage.

```python
if perte_val < meilleure_perte_validation:
    meilleure_perte_validation = perte_val
    compteur_patience = 0
    meilleur_modele = copy.deepcopy(self.modele.state_dict())  # Sauvegarde
else:
    compteur_patience += 1
    if compteur_patience >= 7:  # PATIENCE = 7
        break  # Arret !
```

On surveille la perte de **validation** (pas la perte d'entrainement). Si elle ne diminue plus pendant 7 epoques consecutives, on arrete et on garde le meilleur modele.


## Q39 : C'est quoi une epoque ?

Une **epoque** = un passage complet sur toutes les donnees d'entrainement. Si on a 3402 echantillons et un batch_size de 64, une epoque = 3402/64 = environ 54 batches.

Notre modele est configure pour maximum 35 epoques (`NOMBRE_EPOQUES = 35`), mais l'early stopping peut l'arreter plus tot.


## Q40 : Que se passe-t-il exactement dans une epoque d'entrainement ?

1. Le modele passe en **mode entrainement** : `self.modele.train()` (Dropout et BatchNorm sont actifs)
2. Pour chaque batch de 64 images :
   - Le modele fait ses predictions : `sorties_classif, sorties_reg = self.modele(images)`
   - On calcule la perte combinee : classification + 0.01 * regression
   - On calcule les gradients : `perte.backward()`
   - On ajuste les parametres : `self.optimiseur.step()`
3. Le modele passe en **mode evaluation** : `self.modele.eval()` (Dropout desactive)
4. On evalue sur les donnees de validation (sans ajuster les parametres)
5. On enregistre les metriques dans l'historique


## Q41 : C'est quoi la difference entre `model.train()` et `model.eval()` ?

| | `model.train()` | `model.eval()` |
|---|---|---|
| Quand | Pendant l'entrainement | Pendant la validation/test |
| Dropout | Actif (desactive 50% des neurones) | Desactive (tous les neurones actifs) |
| BatchNorm | Utilise les stats du batch courant | Utilise les stats accumulees pendant l'entrainement |
| Gradients | Calcules | Pas calcules (avec `@torch.no_grad()`) |


## Q42 : C'est quoi `@torch.no_grad()` ?

C'est un decorateur Python qui desactive le calcul des gradients. Pendant la validation et le test, on n'a pas besoin de gradients car on ne modifie pas les parametres.

```python
@torch.no_grad()
def _valider(self):
    self.modele.eval()
    # ... evaluation sans calcul de gradients ...
```

Cela economise de la memoire et accelere les calculs.


## Q43 : C'est quoi `copy.deepcopy(self.modele.state_dict())` ?

`state_dict()` retourne un dictionnaire contenant tous les parametres du modele (poids, biais). `copy.deepcopy()` en fait une copie complete et independante.

On sauvegarde cette copie a chaque fois que le modele s'ameliore. A la fin de l'entrainement, on restaure les parametres de la meilleure version (pas la derniere, qui peut etre moins bonne a cause du surapprentissage).


---


# PARTIE 5 : L'EVALUATION


## Q44 : C'est quoi la Precision (Accuracy) ?

C'est le pourcentage de predictions correctes sur le total.

```python
precision = bonnes_predictions / total
```

Si le modele classe correctement 31 images sur 31, precision = 31/31 = 100%.


## Q45 : C'est quoi la matrice de confusion ?

C'est un tableau qui montre les erreurs du modele en detail :

```
                    Predit: sans     Predit: avec
Vrai: sans    |      16 (VP)     |      0 (FP)
Vrai: avec    |       0 (FN)     |     15 (VP)
```

- VP (Vrai Positif) : le modele dit "avec" et c'est vrai = "avec"
- FP (Faux Positif) : le modele dit "avec" mais c'est faux, c'est "sans"
- FN (Faux Negatif) : le modele dit "sans" mais c'est faux, c'est "avec"

Dans le code :
```python
matrice = confusion_matrix(resultats['etiquettes'], resultats['predictions'])
```


## Q46 : C'est quoi Precision, Recall et F1-Score (les metriques sklearn) ?

Ce sont trois metriques complementaires (attention, "Precision" ici est different de "Accuracy") :

**Precision** = Parmi tout ce que le modele a classifie comme "avec_fourreau", combien l'etaient vraiment ?
```
Precision = Vrais Positifs / (Vrais Positifs + Faux Positifs)
```

**Recall (Rappel)** = Parmi tous les vrais "avec_fourreau", combien le modele a-t-il detecte ?
```
Recall = Vrais Positifs / (Vrais Positifs + Faux Negatifs)
```

**F1-Score** = Moyenne harmonique de Precision et Recall. Utile quand on veut une seule metrique qui combine les deux.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Notre resultat final : Precision=1.00, Recall=1.00, F1=1.00 pour les deux classes = parfait.


## Q47 : C'est quoi la MAE (Mean Absolute Error) ?

La MAE mesure l'erreur moyenne de la regression en metres.

```python
erreurs = np.abs(largeurs_pred - largeurs_vraies)  # Valeur absolue des ecarts
mae = erreurs.mean()  # Moyenne
```

Si les predictions sont [20, 35, 15] et les valeurs reelles sont [25, 30, 18] :
```
erreurs = [|20-25|, |35-30|, |15-18|] = [5, 5, 3]
MAE = (5 + 5 + 3) / 3 = 4.33 metres
```

Notre MAE finale sur les donnees reelles : **14.91m** pour la regression CNN, mais **3.02m** avec la mesure physique (voir Partie 9).


## Q48 : C'est quoi le RMSE (Root Mean Squared Error) ?

Le RMSE est comme la MAE mais penalise davantage les grosses erreurs.

```python
rmse = np.sqrt(((largeurs_pred - largeurs_vraies) ** 2).mean())
```

Si les erreurs sont [5, 5, 3] :
```
MSE = (25 + 25 + 9) / 3 = 19.67
RMSE = sqrt(19.67) = 4.43 metres
```

Le RMSE est toujours >= MAE. Plus la difference RMSE-MAE est grande, plus il y a des erreurs extremes.


## Q49 : Comment `torch.max` determine-t-il la classe predite ?

```python
_, predictions = torch.max(sorties_classif, 1)
```

`torch.max(tensor, 1)` retourne deux choses :
1. Les valeurs maximales (qu'on ignore avec `_`)
2. Les indices des maximums (la classe predite)

Si `sorties_classif = [[0.2, 0.8], [0.9, 0.1]]` :
- Image 1 : max a l'indice 1 = "avec_fourreau"
- Image 2 : max a l'indice 0 = "sans_fourreau"


---


# PARTIE 6 : LA CONFIGURATION


## Q50 : Que contient le fichier config.py ?

Tous les hyperparametres du projet, centralises en un seul endroit :

```python
APPAREIL = torch.device('mps')      # GPU Apple Silicon (ou 'cuda' pour NVIDIA, ou 'cpu')
TAILLE_IMAGE = 224                   # Toutes les images redimensionnees en 224x224
NOMBRE_CANAUX = 4                    # 4 canaux magnetiques par image
NOMBRE_CLASSES = 2                   # 2 classes : sans/avec fourreau
TAILLE_LOT = 64                      # 64 images par batch
NOMBRE_EPOQUES = 35                  # Maximum 35 epoques d'entrainement
TAUX_APPRENTISSAGE = 0.001           # Learning rate pour l'optimiseur Adam
PATIENCE = 7                         # Early stopping apres 7 epoques sans amelioration
LAMBDA_REGRESSION = 0.01             # Poids de la regression dans la perte totale
POIDS_CLASSES = [1.5, 1.0]           # Compensation du desequilibre de classes
NOMS_CLASSES = ['sans_fourreau', 'avec_fourreau']
```


## Q51 : Comment le code detecte-t-il automatiquement le GPU ?

```python
if torch.cuda.is_available():
    APPAREIL = torch.device('cuda')       # GPU NVIDIA
elif torch.backends.mps.is_available():
    APPAREIL = torch.device('mps')        # GPU Apple Silicon (M1/M2/M3)
else:
    APPAREIL = torch.device('cpu')        # Pas de GPU, utilise le processeur
```

Le GPU accelere enormement les calculs matriciels (10 a 100 fois plus rapide que le CPU pour le deep learning).


---


# PARTIE 7 : LES RESULTATS ET L'EVOLUTION


## Q52 : Pourquoi le modele avait-il 50% sur les donnees reelles au debut ?

Le modele etait entraine uniquement sur des donnees synthetiques. Les donnees reelles ont des caracteristiques differentes (bruit, conditions de mesure, echelle). Le modele n'avait jamais vu de donnees reelles, il ne savait pas les interpreter. Il classait TOUT comme "avec_fourreau" (la classe majoritaire), ce qui donnait 50% car le test reel est equilibre (51 avec + 51 sans = 50/50).

C'est le probleme classique du **domain gap** (ecart de domaine) entre donnees synthetiques et reelles.


## Q53 : Qu'est-ce qui a fait passer de 51% a 96.77% ?

Le surechantillonnage x10 des donnees reelles. En repetant les 71 echantillons reels 10 fois dans le train (= 710 copies), le modele a vu suffisamment de donnees reelles pour apprendre leurs specificites. Chaque copie etait differente grace aux augmentations aleatoires (flips, rotations).


## Q54 : Qu'est-ce qui a fait passer de 96.77% a 100% ?

Trois optimisations combinees :
1. Surechantillonnage passe de x10 a **x20** (1420 copies reelles au lieu de 710)
2. Plus d'epoques (**35** au lieu de 25) pour laisser le modele converger completement
3. Plus de patience (**7** au lieu de 5) pour eviter un arret premature


## Q55 : Pourquoi le U-Net a-t-il ete moins bon que le CNN Simple ?

Le U-Net a obtenu 74% sur le reel contre 96.77% pour le CNN Simple. Le U-Net est une architecture concue pour la **segmentation d'images** (colorier chaque pixel). Pour notre tache de **classification globale** (oui/non pour l'image entiere), le CNN Simple avec ses couches denses est plus adapte. Le U-Net avait tendance a voir des fourreaux partout (recall sans_fourreau = 0.50), probablement parce que son architecture spatiale captait trop de details parasites dans les donnees reelles.


## Q56 : Tableau recapitulatif de toutes les versions

| Version | Changement | Classif Synth | Classif Reel | MAE Reel |
|:---:|---|:---:|:---:|:---:|
| V1 | Synthetique seul | 99.76% | 50.00% | 24.44m |
| V2 | +reel dans train | 99.76% | 51.61% | 13.03m |
| V3 | +surechantillonnage x10 | 99.06% | 96.77% | 14.49m |
| V4 | +surechant. x20, 35 epoques | 96.94% | 100.00% | 14.13m (CNN) |
| **V5** | **+mesure physique (pixels x 20cm)** | **98.59%** | **100.00%** | **3.02m** |


---


# PARTIE 8 : LE PIPELINE COMPLET (main.py)


## Q57 : Quelles sont les etapes du pipeline dans l'ordre ?

```python
def pipeline():
    # 1. Charger les donnees depuis le CSV
    resultat = charger_donnees()

    # 2. Afficher la distribution des classes
    afficher_distribution(dataset_train, dataset_test_synth, dataset_test_reel)

    # 3. Creer le modele CNN Simple Multi-Task
    modele = creer_modele('cnn_simple')

    # 4. Entrainer le modele
    entraineur = Entraineur(modele, chargeur_train, chargeur_val)
    historique = entraineur.entrainer()

    # 5. Afficher les courbes d'apprentissage
    afficher_courbes_apprentissage(historique)

    # 6. Evaluer sur les donnees synthetiques
    resultats_synth = evaluer_modele(modele, chargeur_test_synth)

    # 7. Evaluer sur les donnees reelles
    resultats_reel = evaluer_modele(modele, chargeur_test_reel)

    # 8. Mesure physique de largeur (pixels x 20cm)
    resultats_physique = evaluer_mesure_physique(dataset_test_reel)

    # 9. Afficher le resume final
    print(f'Classification (reel) : {resultats_reel["precision"] * 100:.2f}%')
    print(f'Regression MAE Phys (reel) : {resultats_physique["mae"]:.2f} m')
```


## Q58 : Quels fichiers sont generes dans resultats/ ?

| Fichier | Contenu |
|---------|---------|
| `meilleur_modele.pth` | Les parametres du meilleur modele sauvegarde |
| `courbes_apprentissage.png` | 3 graphiques : perte, precision, MAE au fil des epoques |
| `distribution_classes.png` | Nombre d'echantillons par classe dans chaque split |
| `matrice_confusion_*.png` | Matrices de confusion (synth et reel) |
| `regression_*.png` | Scatter plots predictions vs realite + histogramme des erreurs |
| `exemples_predictions.png` | Exemples visuels de predictions du modele |


## Q59 : Comment relancer le modele sauvegarde sans re-entrainer ?

```python
import torch
from src.modeles import creer_modele

modele = creer_modele('cnn_simple')
modele.load_state_dict(torch.load('resultats/meilleur_modele.pth'))
modele.eval()
# Le modele est pret a faire des predictions sans re-entrainement
```


## Q60 : Comment lancer le projet ?

```bash
# 1. Installer les dependances
pip install -r requirements.txt

# 2. Lancer l'entrainement + evaluation
python main.py
```

Les resultats (graphiques + modele) sont sauvegardes automatiquement dans `resultats/`.


---


# PARTIE 9 : LA MESURE PHYSIQUE DE LARGEUR (mesure_largeur.py)


## Q61 : Pourquoi la regression CNN se trompe-t-elle autant (~14m) ?

Le CNN recoit des images redimensionnees a 224x224 pixels. Or les fichiers originaux ont des tailles tres differentes (219x288, 353x773, 443x453...). En redimensionnant tout a la meme taille, on **perd l'information d'echelle** : un fourreau de 10m et un fourreau de 50m peuvent avoir exactement la meme apparence apres le resize.

Le CNN ne "mesure" pas, il "devine" la largeur a partir de patterns visuels. C'est comme demander a quelqu'un d'estimer la taille d'un batiment a partir d'une photo sans echelle.


## Q62 : C'est quoi l'approche hybride ?

Au lieu de tout faire avec le CNN, on separe les taches :
- **Le CNN fait la classification** : avec/sans fourreau (100% de precision)
- **Un calcul physique fait la regression** : mesure de la largeur en pixels, puis conversion en metres

Chaque outil fait ce pour quoi il est le meilleur. Le CNN excelle en reconnaissance de patterns, le calcul physique excelle en mesure.


## Q63 : Comment fonctionne la mesure physique (mesure_largeur.py) ?

L'algorithme en 7 etapes :

```python
def mesurer_largeur_physique(chemin_fichier):
    # 1. Charger les 4 canaux magnetiques
    mat = np.load(chemin_fichier)['data']

    # 2. Calculer la norme = intensite totale du signal
    norme = sqrt(canal_0^2 + canal_1^2 + canal_2^2 + canal_3^2)

    # 3. Normaliser entre 0 et 1
    norme = norme / norme.max()

    # 4. Trouver le centre du fourreau
    centre_y, centre_x = center_of_mass(norme)  # scipy

    # 5. Prendre un cross-section au centre (5 pixels de large)
    profil_horizontal = norme[centre_y-2:centre_y+3, :].mean(axis=0)
    profil_vertical = norme[:, centre_x-2:centre_x+3].mean(axis=1)

    # 6. Mesurer la largeur a 30% du pic d'intensite
    largeur_h = largeur_a_seuil(profil_horizontal, seuil=0.3)
    largeur_v = largeur_a_seuil(profil_vertical, seuil=0.3)

    # 7. Prendre le minimum et convertir en metres
    largeur = min(largeur_h, largeur_v) * 0.20  # 1 pixel = 20cm
```


## Q64 : Pourquoi prendre le minimum des deux dimensions ?

Un fourreau est un objet lineaire : il est **long** dans un sens et **etroit** dans l'autre. Si on mesure 50 pixels dans un sens (longueur du fourreau) et 15 pixels dans l'autre (largeur), la vraie largeur est 15 pixels = 3.0m.

Prendre le minimum permet de toujours capturer la dimension la plus etroite, qui est la largeur.


## Q65 : Pourquoi le seuil de 30% et pas 50% ?

On a teste plusieurs seuils :

| Seuil | MAE |
|:---:|:---:|
| 0.3 | **2.66m** |
| 0.4 | 3.13m |
| 0.5 | 4.68m |
| 0.6 | 7.49m |
| 0.7 | 11.00m |
| 0.8 | 13.75m |

Le seuil de 30% capte mieux les bords du signal magnetique car le champ decroit progressivement (il ne s'arrete pas net). A 50% (FWHM classique), on coupe trop tot et on sous-estime la largeur.

Resultat final : **MAE 3.02m** avec la mesure physique contre **14.91m** avec la regression CNN = **5x plus precis**.
