# Fondamentaux du machine learning

> Tout ce qui est vrai du plus petit modèle linéaire reste vrai d'un LLM : données,
> découpage, fonction de coût, descente de gradient, sur-apprentissage, métriques.
> Ce chapitre donne ce socle, avec les ordres de grandeur et les pièges qui coûtent
> le plus cher en pratique.

## 1. Le vocabulaire minimal

| Terme | Définition | Exemple |
|---|---|---|
| Échantillon (exemple) | Une unité d'observation | Un email |
| Feature (variable, attribut) | Une caracteristique mesurable de l'échantillon | Nombre de liens, longueur du sujet |
| Label (étiquette, cible) | La réponse attendue | Spam / non spam |
| Dataset (jeu de données) | L'ensemble des échantillons | 50 000 emails annotées |
| Modèle | La fonction paramétrée qu'on ajuste | Un classifieur logistique |
| Paramètre | Une valeur apprise | Un poids de 0,83 |
| Hyperparamètre | Une valeur choisie par l'humain avant l'entraînement | Learning rate de 0,001 |

Règle empirique souvent vérifiée : **80 pour cent du résultat vient des données et
de la définition de la tâche**, pas du choix de l'architecture. Un projet ML qui
échoue échoue presque toujours sur la tâche mal définie, les étiquettes bruitées ou
la fuite de données.

## 2. Les cinq régimes d'apprentissage

```
Supervisé          entrées + réponses attendues  -> apprendre la correspondance
NON Supervisé      entrées seules                -> trouver une structure
SEMI-Supervisé     peu d'étiquettes + beaucoup de données brutes
AUTO-Supervisé     l'étiquette vient des données elles-mêmes (mot masqué,
                   token suivant, paire image/texte)
RENFORCEMENT       un agent agit, reçoit une récompense, ajuste sa politique
```

| Régime | Signal d'apprentissage | Coût d'annotation | Exemples d'usage |
|---|---|---|---|
| Supervisé | Étiquette exacte | Élevé | Classification, détection, régression |
| Non supervisé | Aucun | Nul | Clustering, réduction de dimension, détection d'anomalies |
| Semi-supervisé | Quelques étiquettes | Faible | Imagerie médicale (peu de cas annotés) |
| Auto-supervisé | Le texte ou l'image lui-même | Nul, mais énorme volume | Pré-entraînement des LLM, Wav2Vec, CLIP |
| Renforcement | Récompense scalaire | Variable | Jeux, robotique, RLHF |

Le régime auto-supervisé est celui qui a changé l'échelle du domaine : transformer
« prédire le mot suivant » en tâche d'entraînement permet d'utiliser des milliards de
pages web sans aucune annotation humaine. Voir [chapitre 05](../05-llm/README.md).

## 3. Découper les données : la règle qu'on croit connaître

Trois jeux séparés, chacun avec un rôle distinct :

- **Entraînement** : les paramètres sont ajustes dessus.
- **Validation** : on y choisit les hyperparamètres et on y arrête l'entraînement.
- **Test** : on n'y touche qu'une fois, à la fin, pour estimer la performance réelle.

```
+-------------------+------------+-------+
|   Entraînement    | VALIDATION | TEST  |
|   ~70-80 %        |  ~10-15 %  | ~10 % |
+-------------------+------------+-------+
   on apprend         on règle     on mesure (une seule fois)
```

Ratios typiques : 80/10/10 pour de grands jeux, 60/20/20 pour de petits jeux ou la
validation croisee devient préférable.

**Fuite de données (data leakage)** : toute information du test qui contamine
l'entraînement. C'est l'erreur la plus coûteuse du domaine, parce qu'elle gonfle les
scores sans jamais lever d'alerte. Formes classiques :

- Mélanger les lignes avant de découper, alors que plusieurs lignes concernent le
  même individu (le même patient dans train et test).
- Normaliser (moyenne, ecart-type) avant de découper : les statistiques du test
  fuitent dans l'entraînement.
- Utiliser une variable disponible seulement *après* l'événement qu'on veut prédire.
- Entraîner sur des documents dont la version du test est quasi identique (doublons).

**Validation croisee** : on découpe en k blocs, on entraîne k fois en laissant un
bloc de côté. Coûteux (k entraînements) mais robuste, indispensable en dessous de
quelques milliers d'exemples.

## 4. Optimiser : la descente de gradient

Le modèle produit une prediction, une **fonction de coût** mesure l'écart à la cible,
et on déplace les paramètres dans la direction opposée au gradient.

```
pseudo-code de la boucle d'entraînement
---------------------------------------
pour chaque epoch:
    pour chaque mini-batch (x, y):
        y_pred = modele(x)               # propagation avant
        perte  = coût(y_pred, y)         # mesure de l'erreur
        grad   = retroprop(perte)        # gradient de la perte / paramètres
        theta  = theta - lr * grad       # pas de descente
```

| Hyperparamètre | Effet quand trop petit | Effet quand trop grand |
|---|---|---|
| Learning rate (lr) | Convergence interminable, minimum médiocre | Divergence, perte qui explose |
| Batch size | Bruit élevé, instabilité, GPU sous-utilisé | Mémoire saturée, generalisation parfois dégradée |
| Nombre d'epochs | Sous-apprentissage | Sur-apprentissage |
| Nombre de paramètres | Sous-apprentissage | Sur-apprentissage |

Ordres de grandeur usuels : lr entre 1e-4 et 1e-3 pour un MLP entraîné de zéro,
1e-5 a 2e-5 pour un fine-tuning de LLM, batch size de 32 a 512 en vision, de
quelques dizaines de milliers de tokens en pré-entraînement de LLM.

**SGD, momentum, Adam** : le SGD pur avance juste à contre-gradient. Le momentum
ajoute de l'inertie. Adam adapte le pas par paramètre et combine moment du gradient
et moment du carré du gradient. Adam/AdamW est le choix par défaut en pratique,
SGD avec momentum reste competitif en vision quand on cherche la meilleure
generalisation finale.

## 5. Sous-apprendre, sur-apprendre, arbitrer

```
perte
  |\
  | \                      perte d'entraînement
  |  \______________________
  |                         \  perte de validation
  |        ___               \
  |     __/   \___             \        <- sur-apprentissage :
  |  __/          \_______     \           l'entraînement continue de
  | /                      \    \          s'améliorer, la validation se dégrade
  |/                        \____\______
  +------------------------------------------- epochs
        sous-apprentissage | zone utile | sur-apprentissage
```

| Symptôme | Perte entraînement | Perte validation | Remède |
|---|---|---|---|
| Sous-apprentissage | Élevée | Élevée | Plus de capacité, plus d'epochs, moins de régularisation, meilleures features |
| Sur-apprentissage | Faible | Élevée | Plus de données, augmentation, dropout, L1/L2, early stopping, modèle plus petit |
| Juste | Faible | Faible | Rien à faire, mesurer sur le test |

**Compromis biais-variance** : un modèle trop simple a un biais élevé (il se trompe
systématiquement), un modèle trop flexible a une variance élevée (il change beaucoup
avec l'échantillon d'entraînement). On cherche le point où l'erreur de test est
minimale, pas celui où l'erreur d'entraînement est minimale.

## 6. Régularisation : les cinq leviers

| Technique | Principe | Coût |
|---|---|---|
| L2 (weight decay) | Pénaliser la somme des carrés des poids : ils restent petits | Un terme dans la perte |
| L1 | Pénaliser la somme des valeurs absolues : beaucoup de poids deviennent exactement nuls | Sélection de variables, mais solution plus instable |
| Dropout | Désactiver aléatoirement une fraction des neurones à chaque pas | Entraînement plus long, rien à l'inférence |
| Early stopping | Arrêter quand la perte de validation remonte | Aucun, mais exige un jeu de validation propre |
| Augmentation | Fabriquer des exemples transformes (rotations, paraphrases, bruit) | Du calcul, et un risque d'invalider la tâche si la transformation change la classe |

## 7. Métriques : choisir la bonne change le comportement du modèle

**Classification binaire** — matrice de confusion :

```
                 prédit positif   prédit négatif
réel positif     vrai positif     faux négatif
réel négatif     faux positif     vrai négatif
```

| Métrique | Formule | Quand l'utiliser |
|---|---|---|
| Accuracy | (VP + VN) / total | Classes equilibrees uniquement |
| Précision | VP / (VP + FP) | Quand un faux positif coûte cher (spam, alerte) |
| Rappel (recall) | VP / (VP + FN) | Quand un faux négatif coûte cher (maladie, fraude) |
| F1 | 2 x (P x R) / (P + R) | Compromis précision/rappel, classes déséquilibrées |
| ROC-AUC | Aire sous la courbe VP/FP | Classement global, insensible au seuil |
| PR-AUC | Aire sous la courbe précision/rappel | Meilleure que ROC-AUC en fort déséquilibre |

**Régression** : RMSE (pénalise les grandes erreurs, même unité que la cible),
MAE (plus robuste aux valeurs extrêmes), MAPE (en pourcentage, instable près de zero).

**Génération de texte** : perplexité (nombre moyen de choix equiprobables au token
suivant, plus bas est mieux), exact match, ROUGE, et surtout des évaluations ciblées
et humaines. Voir [chapitre 05](../05-llm/README.md).

**Attention au déséquilibre de classes** : avec 1 pour cent de positifs, un modèle
qui répond toujours « négatif » atteint 99 pour cent d'accuracy et ne sert à rien.
Il faut pondérer les classes, rééchantillonner, ou changer de métrique.

## 8. ML classique ou deep learning ?

```
Le signal est-il spatial, séquentiel, ou perceptuel (image, son, texte) ?
|--- NON  -> commence par un arbre boosté (XGBoost, LightGBM). Rapide, robuste,
|           interprète, excellent sur données tabulaires. C'est souvent la bonne
|           réponse en production.
|--- OUI  -> deep learning. Mais commence par un petit modèle et une baseline simple.
```

| Critère | ML classique | Deep learning |
|---|---|---|
| Données nécessaires | Des centaines à milliers d'exemples | Des dizaines de milliers à milliards |
| Données tabulaires | Excellent | Souvent moins bon qu'un arbre |
| Image, son, texte | Nécessité des features faites main | Excellent, apprend les features |
| Interprétabilité | Bonne à moyenne | Faible, nécessité des outils dédiés |
| Coût de calcul | Faible (CPU suffit) | Élevé (GPU souvent requis) |

## 9. Types d'apprentissage non supervisés utiles au quotidien

- **Clustering** (k-means, DBSCAN) : segmenter des clients, grouper des documents.
- **Réduction de dimension** (ACP/PCA, UMAP, t-SNE) : visualiser, compacter, debruiter.
- **Détection d'anomalies** : isoler ce qui s'ecarte du comportement habituel, sans
  labels d'anomalie.
- **Modélisation de densité** : estimer la plausibilité d'un échantillon.

## 10. Le pipeline complet, de la question au service

```
1. Définir la tâche et la métrique         <- étape la plus sous-estimée
2. Collecter et annoter les données
3. Découper train / validation / test (avant toute transformation !)
4. Établir une baseline bête (règle simple, modèle linéaire, classe majoritaire)
5. Entraîner, régler les hyperparamètres sur la validation
6. Évaluer une fois sur le test, analyser les erreurs
7. Déployer en surveillant la dérive (les distributions changent avec le temps)
8. Réentraîner périodiquement
```

## Ce qu'il faut retenir

- Paramètres (appris) et hyperparamètres (choisis) sont deux catégories distinctes.
- Le découpage train/validation/test doit etre fait avant toute transformation pour
  éviter la fuite de données, cause n°1 de scores trop optimistes.
- La descente de gradient ne fait qu'un pas à contre-gradient ; le learning rate est
  l'hyperparamètre le plus sensible.
- Sur-apprentissage et sous-apprentissage se diagnostiquent en comparant les pertes
  d'entraînement et de validation, pas en regardant la seule performance finale.
- La métrique choisie définit ce que le modèle optimise : elle doit refléter le coût
  métier des erreurs, pas la facilité de calcul.
- Sur données tabulaires, un arbre boosté bat souvent un réseau profond pour beaucoup
  moins cher.
- Une baseline bête est obligatoire : sans elle, on ne sait pas si le modèle apporte
  quoi que ce soit.

## Erreurs fréquentes / idées reçues

- « Je normalise avant de découper » -> fuite de données ; les statistiques de
  normalisation doivent etre calculees sur l'entraînement seul.
- « Mon accuracy est de 99 pour cent donc c'est bon » -> vérifie la distribution des
  classes et le taux de la classe majoritaire avant de te rejouir.
- « Plus d'epochs, c'est toujours mieux » -> à partir d'un certain point, la
  validation se dégrade alors que l'entraînement s'améliore.
- « Un modèle plus gros va résoudre mon problème » -> une tâche mal définie ou des
  étiquettes bruitées ne se rattrapent pas avec de la capacité.
- « Le modèle est entraîné une fois pour toutes » -> en production, les distributions
  dérivent et le modèle doit etre surveille et reentraine.

## Pour aller plus loin

- [Chapitre 02 : les maths de l'IA sans douleur](../02-mathematiques/README.md)
- [Chapitre 03 : réseaux de neurones](../03-reseaux-de-neurones/README.md)
- [Chapitre 12 : données et annotations](../12-donnees/README.md)
- [Chapitre 16 : mise en pratique](../16-pratique/README.md)
- Référence pratique : « An Introduction to Statistical Learning » (James, Witten,
  Hastie, Tibshirani), gratuit en ligne.
- Définition du déséquilibre de classes et du choix de métrique : documentation
  scikit-learn, sections « Model évaluation ».
