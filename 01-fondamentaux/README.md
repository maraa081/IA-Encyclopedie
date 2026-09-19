# Fondamentaux du machine learning

> Tout ce qui est vrai du plus petit modele lineaire reste vrai d'un LLM : donnees,
> decoupage, fonction de cout, descente de gradient, sur-apprentissage, metriques.
> Ce chapitre donne ce socle, avec les ordres de grandeur et les pieges qui coutent
> le plus cher en pratique.

## 1. Le vocabulaire minimal

| Terme | Definition | Exemple |
|---|---|---|
| Echantillon (exemple) | Une unite d'observation | Un email |
| Feature (variable, attribut) | Une caracteristique mesurable de l'echantillon | Nombre de liens, longueur du sujet |
| Label (etiquette, cible) | La reponse attendue | Spam / non spam |
| Dataset (jeu de donnees) | L'ensemble des echantillons | 50 000 emails annotees |
| Modele | La fonction parametree qu'on ajuste | Un classifieur logistique |
| Parametre | Une valeur apprise | Un poids de 0,83 |
| Hyperparametre | Une valeur choisie par l'humain avant l'entrainement | Learning rate de 0,001 |

Regle empirique souvent verifiee : **80 pour cent du resultat vient des donnees et
de la definition de la tache**, pas du choix de l'architecture. Un projet ML qui
echoue echoue presque toujours sur la tache mal definie, les etiquettes bruitees ou
la fuite de donnees.

## 2. Les cinq regimes d'apprentissage

```
SUPERVISE          entrees + reponses attendues  -> apprendre la correspondance
NON SUPERVISE      entrees seules                -> trouver une structure
SEMI-SUPERVISE     peu d'etiquettes + beaucoup de donnees brutes
AUTO-SUPERVISE     l'etiquette vient des donnees elles-memes (mot masque,
                   token suivant, paire image/texte)
RENFORCEMENT       un agent agit, recoit une recompense, ajuste sa politique
```

| Regime | Signal d'apprentissage | Cout d'annotation | Exemples d'usage |
|---|---|---|---|
| Supervise | Etiquette exacte | Eleve | Classification, detection, regression |
| Non supervise | Aucun | Nul | Clustering, reduction de dimension, detection d'anomalies |
| Semi-supervise | Quelques etiquettes | Faible | Imagerie medicale (peu de cas annotes) |
| Auto-supervise | Le texte ou l'image lui-meme | Nul, mais enorme volume | Pretraining des LLM, Wav2Vec, CLIP |
| Renforcement | Recompense scalaire | Variable | Jeux, robotique, RLHF |

Le regime auto-supervise est celui qui a change l'echelle du domaine : transformer
« predire le mot suivant » en tache d'entrainement permet d'utiliser des milliards de
pages web sans aucune annotation humaine. Voir [chapitre 05](../05-llm/README.md).

## 3. Decouper les donnees : la regle qu'on croit connaitre

Trois jeux separes, chacun avec un role distinct :

- **Entrainement** : les parametres sont ajustes dessus.
- **Validation** : on y choisit les hyperparametres et on y arrete l'entrainement.
- **Test** : on n'y touche qu'une fois, a la fin, pour estimer la performance reelle.

```
+-------------------+------------+-------+
|   ENTRAINEMENT    | VALIDATION | TEST  |
|   ~70-80 %        |  ~10-15 %  | ~10 % |
+-------------------+------------+-------+
   on apprend         on regle     on mesure (une seule fois)
```

Ratios typiques : 80/10/10 pour de grands jeux, 60/20/20 pour de petits jeux ou la
validation croisee devient preferable.

**Fuite de donnees (data leakage)** : toute information du test qui contamine
l'entrainement. C'est l'erreur la plus couteuse du domaine, parce qu'elle gonfle les
scores sans jamais lever d'alerte. Formes classiques :

- Melanger les lignes avant de decouper, alors que plusieurs lignes concernent le
  meme individu (le meme patient dans train et test).
- Normaliser (moyenne, ecart-type) avant de decouper : les statistiques du test
  fuitent dans l'entrainement.
- Utiliser une variable disponible seulement *apres* l'evenement qu'on veut predire.
- Entrainer sur des documents dont la version du test est quasi identique (doublons).

**Validation croisee** : on decoupe en k blocs, on entraine k fois en laissant un
bloc de cote. Couteux (k entrainements) mais robuste, indispensable en dessous de
quelques milliers d'exemples.

## 4. Optimiser : la descente de gradient

Le modele produit une prediction, une **fonction de cout** mesure l'ecart a la cible,
et on deplace les parametres dans la direction opposee au gradient.

```
pseudo-code de la boucle d'entrainement
---------------------------------------
pour chaque epoch:
    pour chaque mini-batch (x, y):
        y_pred = modele(x)               # propagation avant
        perte  = cout(y_pred, y)         # mesure de l'erreur
        grad   = retroprop(perte)        # gradient de la perte / parametres
        theta  = theta - lr * grad       # pas de descente
```

| Hyperparametre | Effet quand trop petit | Effet quand trop grand |
|---|---|---|
| Learning rate (lr) | Convergence interminable, minimum mediocre | Divergence, perte qui explose |
| Batch size | Bruit eleve, instabilite, GPU sous-utilise | Memoire saturee, generalisation parfois degradee |
| Nombre d'epochs | Sous-apprentissage | Sur-apprentissage |
| Nombre de parametres | Sous-apprentissage | Sur-apprentissage |

Ordres de grandeur usuels : lr entre 1e-4 et 1e-3 pour un MLP entraine de zero,
1e-5 a 2e-5 pour un fine-tuning de LLM, batch size de 32 a 512 en vision, de
quelques dizaines de milliers de tokens en pretraining de LLM.

**SGD, momentum, Adam** : le SGD pur avance juste a contre-gradient. Le momentum
ajoute de l'inertie. Adam adapte le pas par parametre et combine moment du gradient
et moment du carre du gradient. Adam/AdamW est le choix par defaut en pratique,
SGD avec momentum reste competitif en vision quand on cherche la meilleure
generalisation finale.

## 5. Sous-apprendre, sur-apprendre, arbitrer

```
perte
  |\
  | \                      perte d'entrainement
  |  \______________________
  |                         \  perte de validation
  |        ___               \
  |     __/   \___             \        <- sur-apprentissage :
  |  __/          \_______     \           l'entrainement continue de
  | /                      \    \          s'ameliorer, la validation se degrade
  |/                        \____\______
  +------------------------------------------- epochs
        sous-apprentissage | zone utile | sur-apprentissage
```

| Symptome | Perte entrainement | Perte validation | Remede |
|---|---|---|---|
| Sous-apprentissage | Elevee | Elevee | Plus de capacite, plus d'epochs, moins de regularisation, meilleures features |
| Sur-apprentissage | Faible | Elevee | Plus de donnees, augmentation, dropout, L1/L2, early stopping, modele plus petit |
| Juste | Faible | Faible | Rien a faire, mesurer sur le test |

**Compromis biais-variance** : un modele trop simple a un biais eleve (il se trompe
systematiquement), un modele trop flexible a une variance elevee (il change beaucoup
avec l'echantillon d'entrainement). On cherche le point ou l'erreur de test est
minimale, pas celui ou l'erreur d'entrainement est minimale.

## 6. Regularisation : les cinq leviers

| Technique | Principe | Cout |
|---|---|---|
| L2 (weight decay) | Penaliser la somme des carres des poids : ils restent petits | Un terme dans la perte |
| L1 | Penaliser la somme des valeurs absolues : beaucoup de poids deviennent exactement nuls | Selection de variables, mais solution plus instable |
| Dropout | Desactiver aleatoirement une fraction des neurones a chaque pas | Entrainement plus long, rien a l'inference |
| Early stopping | Arreter quand la perte de validation remonte | Aucun, mais exige un jeu de validation propre |
| Augmentation | Fabriquer des exemples transformes (rotations, paraphrases, bruit) | Du calcul, et un risque d'invalider la tache si la transformation change la classe |

## 7. Metriques : choisir la bonne change le comportement du modele

**Classification binaire** — matrice de confusion :

```
                 predit positif   predit negatif
reel positif     vrai positif     faux negatif
reel negatif     faux positif     vrai negatif
```

| Metrique | Formule | Quand l'utiliser |
|---|---|---|
| Accuracy | (VP + VN) / total | Classes equilibrees uniquement |
| Precision | VP / (VP + FP) | Quand un faux positif coute cher (spam, alerte) |
| Rappel (recall) | VP / (VP + FN) | Quand un faux negatif coute cher (maladie, fraude) |
| F1 | 2 x (P x R) / (P + R) | Compromis precision/rappel, classes desequilibrees |
| ROC-AUC | Aire sous la courbe VP/FP | Classement global, insensible au seuil |
| PR-AUC | Aire sous la courbe precision/rappel | Meilleure que ROC-AUC en fort desequilibre |

**Regression** : RMSE (penalise les grandes erreurs, meme unite que la cible),
MAE (plus robuste aux valeurs extremes), MAPE (en pourcentage, instable pres de zero).

**Generation de texte** : perplexite (nombre moyen de choix equiprobables au token
suivant, plus bas est mieux), exact match, ROUGE, et surtout des evaluations ciblees
et humaines. Voir [chapitre 05](../05-llm/README.md).

**Attention au desequilibre de classes** : avec 1 pour cent de positifs, un modele
qui repond toujours « negatif » atteint 99 pour cent d'accuracy et ne sert a rien.
Il faut ponderer les classes, reechantillonner, ou changer de metrique.

## 8. ML classique ou deep learning ?

```
Le signal est-il spatial, sequentiel, ou perceptuel (image, son, texte) ?
|--- NON  -> commence par un arbre booste (XGBoost, LightGBM). Rapide, robuste,
|           interprete, excellent sur donnees tabulaires. C'est souvent la bonne
|           reponse en production.
|--- OUI  -> deep learning. Mais commence par un petit modele et une baseline simple.
```

| Critere | ML classique | Deep learning |
|---|---|---|
| Donnees necessaires | Des centaines a milliers d'exemples | Des dizaines de milliers a milliards |
| Donnees tabulaires | Excellent | Souvent moins bon qu'un arbre |
| Image, son, texte | Necessite des features faites main | Excellent, apprend les features |
| Interpretabilite | Bonne a moyenne | Faible, necessite des outils dedies |
| Cout de calcul | Faible (CPU suffit) | Eleve (GPU souvent requis) |

## 9. Types d'apprentissage non supervises utiles au quotidien

- **Clustering** (k-means, DBSCAN) : segmenter des clients, grouper des documents.
- **Reduction de dimension** (ACP/PCA, UMAP, t-SNE) : visualiser, compacter, debruiter.
- **Detection d'anomalies** : isoler ce qui s'ecarte du comportement habituel, sans
  labels d'anomalie.
- **Modelisation de densite** : estimer la plausibilite d'un echantillon.

## 10. Le pipeline complet, de la question au service

```
1. Definir la tache et la metrique         <- etape la plus sous-estimee
2. Collecter et annoter les donnees
3. Decouper train / validation / test (avant toute transformation !)
4. Etablir une baseline bete (regle simple, modele lineaire, classe majoritaire)
5. Entrainer, regler les hyperparametres sur la validation
6. Evaluer une fois sur le test, analyser les erreurs
7. Deployer en surveillant la derive (les distributions changent avec le temps)
8. Reentrainer periodiquement
```

## Ce qu'il faut retenir

- Parametres (appris) et hyperparametres (choisis) sont deux categories distinctes.
- Le decoupage train/validation/test doit etre fait avant toute transformation pour
  eviter la fuite de donnees, cause n°1 de scores trop optimistes.
- La descente de gradient ne fait qu'un pas a contre-gradient ; le learning rate est
  l'hyperparametre le plus sensible.
- Sur-apprentissage et sous-apprentissage se diagnostiquent en comparant les pertes
  d'entrainement et de validation, pas en regardant la seule performance finale.
- La metrique choisie definit ce que le modele optimise : elle doit refleter le cout
  metier des erreurs, pas la facilite de calcul.
- Sur donnees tabulaires, un arbre booste bat souvent un reseau profond pour beaucoup
  moins cher.
- Une baseline bete est obligatoire : sans elle, on ne sait pas si le modele apporte
  quoi que ce soit.

## Erreurs frequentes / idees recues

- « Je normalise avant de decouper » -> fuite de donnees ; les statistiques de
  normalisation doivent etre calculees sur l'entrainement seul.
- « Mon accuracy est de 99 pour cent donc c'est bon » -> verifie la distribution des
  classes et le taux de la classe majoritaire avant de te rejouir.
- « Plus d'epochs, c'est toujours mieux » -> a partir d'un certain point, la
  validation se degrade alors que l'entrainement s'ameliore.
- « Un modele plus gros va resoudre mon probleme » -> une tache mal definie ou des
  etiquettes bruitees ne se rattrapent pas avec de la capacite.
- « Le modele est entraine une fois pour toutes » -> en production, les distributions
  derivent et le modele doit etre surveille et reentraine.

## Pour aller plus loin

- [Chapitre 02 : les maths de l'IA sans douleur](../02-mathematiques/README.md)
- [Chapitre 03 : reseaux de neurones](../03-reseaux-de-neurones/README.md)
- [Chapitre 12 : donnees et annotations](../12-donnees/README.md)
- [Chapitre 16 : mise en pratique](../16-pratique/README.md)
- Reference pratique : « An Introduction to Statistical Learning » (James, Witten,
  Hastie, Tibshirani), gratuit en ligne.
- Definition du desequilibre de classes et du choix de metrique : documentation
  scikit-learn, sections « Model evaluation ».
