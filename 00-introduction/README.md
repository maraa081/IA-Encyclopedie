# Introduction à l'intelligence artificielle

> Ce chapitre pose le vocabulaire commun et la carte du territoire : ce qu'on appelle
> IA, machine learning, deep learning et IA générative, d'où vient tout ça, et comment
> fonctionne un grand modèle de langage vu de loin. Il sert de socle aux dix-sept
> chapitres suivants.

## 1. Quatre mots qu'on confond tout le temps

« IA » n'est pas un synonyme de « ChatGPT », et « machine learning » n'est pas un
synonyme de « deep learning ». Ces termes s'emboîtent comme des poupées russes :
chacun est un sous-ensemble du précédent.

```
+-----------------------------------------------------------+
|  INTELLIGENCE ARTIFICIELLE (IA)                           |
|  Toute technique faisant faire à une machine une tâche    |
|  qui exige normalement de l'intelligence humaine.         |
|                                                           |
|   +---------------------------------------------------+   |
|   |  MACHINE LEARNING (ML)                            |   |
|   |  Sous-ensemble de l'IA ou la machine APPREND       |   |
|   |  à partir d'exemples au lieu de suivre des règles  |   |
|   |  écrites à la main.                                |   |
|   |                                                    |   |
|   |   +------------------------------------------+     |   |
|   |   |  DEEP LEARNING (DL)                      |     |   |
|   |   |  ML fonde sur des réseaux de neurones     |     |   |
|   |   |  profonds, qui apprennent eux-mêmes       |     |   |
|   |   |  leurs représentations.                   |     |   |
|   |   |                                          |     |   |
|   |   |   +----------------------------------+   |     |   |
|   |   |   |  IA Générative                   |   |     |   |
|   |   |   |  Produit du contenu neuf :       |   |     |   |
|   |   |   |  texte, image, audio, code.      |   |     |   |
|   |   |   +----------------------------------+   |     |   |
|   |   +------------------------------------------+     |   |
|   +---------------------------------------------------+   |
+-----------------------------------------------------------+
```

**Intelligence artificielle (IA)** : terme parapluie, apparu dans les années 1950.
Il couvre aussi bien les systèmes à base de règles logiques (IA symbolique : « si
température > 30 alors déclencher l'alarme ») que les réseaux de neurones modernes.
Une grande partie de l'IA industrielle d'aujourd'hui n'est pas « intelligente » au
sens philosophique : ce sont des systèmes experts, des planificateurs, des moteurs
de recherche de solution.

**Machine learning (ML, apprentissage automatique)** : au lieu d'écrire les règles,
on fournit des exemples et l'algorithme ajuste ses paramètres internes pour minimiser
une erreur. C'est un renversement de méthode : le programmeur décrit *ce qu'il veut*,
pas *comment le faire*. Un modèle de détection de spam n'a pas de règle « si le mot
'viagra' apparait... » : il a des poids numériques appris sur des milliers d'emails.

**Deep learning (DL, apprentissage profond)** : ML base sur des réseaux de neurones
a nombreuses couches. La différence essentielle avec le ML classique : le modèle
apprend lui-même les variables intermédiaires utiles (*représentation learning*),
là où le ML classique demande de les construire à la main (*feature engineering*).
Voir [chapitre 01](../01-fondamentaux/README.md) et [chapitre 03](../03-reseaux-de-neurones/README.md).

**IA générative** : sous-ensemble du DL qui produit du contenu nouveau plutôt que de
classer ou de prédire. Elle modélise la distribution statistique des données
d'entraînement et échantillonné dedans. Elle regroupe les grands modèles de langage
(texte, code), les modèles de diffusion (image, vidéo) et les modèles audio.

Un point qui évite beaucoup de confusion : **l'IA générative n'est pas la seule IA
utile**. Un modèle de détection de fraude, un système de recommandation ou un
algorithme de prevision de trafic ne génèrent rien et rendent pourtant des services
considérables.

## 2. Une histoire en cinq actes

L'histoire de l'IA n'est pas une progression continue : c'est une alternance de
vagues d'enthousiasme et de « hivers », quand les promesses dépassent les capacités
réelles.

| Periode | Acte | Ce qui se passe | Ce qui casse |
|---|---|---|---|
| 1950-1970 | Naissance symbolique | Test de Turing (1950), conférence de Dartmouth (1956) où le terme « intelligence artificielle » est invente, premiers programmes de demonstration de théorèmes, perceptron (1958) | Minsky et Papert montrent en 1969 les limites du perceptron simple (il ne résout pas XOR) |
| 1974-1980 | Premier hiver | Les promesses de traduction automatique et de robotique ne sont pas tenues, les financements s'arrêtent | Complexité de calcul insuffisante, données inexistantes |
| 1980-1987 | Systèmes experts | Programmes à base de règles métier, très efficaces dans des domaines étroits (diagnostic, configuration) et coûteux à maintenir | Fragilité hors du domaine, coût de maintenance, deuxième hiver |
| 1990-2010 | ML statistique | Les approches probabilistes et l'apprentissage statistique prennent le dessus : SVM, arbres, méthodes ensemblistes. Succès industriels réels (spam, crédit scoring, recommandation) | Les représentations restent construites a la main |
| 2012-2017 | Révolution du deep learning | AlexNet (2012) écrase la competition ImageNet avec un réseau convolutif et deux GPU. En 2016, AlphaGo bat un champion du monde de Go | Coût de calcul, besoin massif de données annotées |
| 2017-2022 | Ere du transformer | Le papier « Attention Is All You Need » (2017) remplace la récurrence par l'attention. GPT, BERT puis les premiers LLM suivent | Les modèles hallucinent, coûts d'entraînement faramineux |
| 2022-2026 | Ere générative et agents | ChatGPT popularise l'usage grand public, les poids ouverts se multiplient, les modèles deviennent multimodaux, le RAG et les agents se banalisent | Fiabilité, sécurité, droit, coût énergétique |

Deux leçons de cette chronologie :

1. **Le goulot d'étranglement a changé trois fois.** D'abord les données (années 1980),
   puis la puissance de calcul (années 2000), puis l'architecture (2017). Aujourd'hui
   le goulot est surtout la fiabilité et le coût.
2. **Chaque vague a produit des résultats réels**, plus modestes que les annonces,
   mais persistants. Les systèmes experts ne sont pas morts : ils tournent encore
   dans l'assurance et l'aéronautique.

## 3. Cartographie : à quoi sert chaque sous-domaine

| Sous-domaine | Entrée | Sortie typique | Exemples concrets |
|---|---|---|---|
| Vision par ordinateur | Image, vidéo | Classe, boîtes, masques, description | Contrôle qualité industriel, lecture de plaques, imagerie médicale |
| Traitement du langage (NLP) | Texte | Classe, entités, résumé, traduction, réponse | Anti-spam, extraction d'information, assistants, traduction |
| Parole (speech) | Signal audio | Transcription, locuteur, commande | Sous-titrage automatique, assistants vocaux, transcription de réunions |
| Apprentissage par renforcement | État d'un environnement | Action | Jeux, robotique, optimisation de ressources, pilotage |
| Graphes | Nœud et arêtes | Prediction sur les nœuds ou les arêtes | Détection de fraude en réseau, découverte de molécules |
| Séries temporelles | Historique date | Valeur future, anomalie | Prevision de consommation, maintenance predictive, marche |
| Recommandation | Interactions utilisateur/objet | Liste ordonnée | Films, musique, produits, contenus |
| IA générative | Prompt, bruit, données | Nouveau contenu | LLM, images, voix, code |

Ces domaines partagent leurs briques : les mêmes optimiseurs, les mêmes GPU, souvent
le même type d'architecture. Un progrès sur les transformers se répercute en vision,
en audio et en biologie.

## 4. Comment fonctionne un LLM, en vue gros grain

Pas besoin de maths pour comprendre l'essentiel. Un grand modèle de langage fait
une seule chose, en boucle : **prédire le token suivant**.

```
prompt : "La capitale de la France est"
        |
        v
  +-----------------+
  |   TOKENISATION  |  le texte devient une suite d'entiers
  +-----------------+  ["La"," capitale"," de"," la"," France"," est"]
        |
        v
  +-----------------+
  |   TRANSFORMER   |  des dizaines de couches qui mélangent
  |   (N couches)   |  l'information entre tous les tokens
  +-----------------+
        |
        v
  distribution de probabilité sur tout le vocabulaire :
     " Paris"  0.87   " Lyon"  0.04   " une"  0.02   ...
        |
        v
  on tire un token (ou on prend le plus probable)
        |
        v
  le token est ajouté à la suite, et on recommence
```

Trois conséquences directes de cette mécanique :

- **Le modèle ne « sait » rien au sens d'une base de données.** Il a compressé des
  régularités statistiques du langage dans des milliards de nombres. Quand il répond
  juste, c'est que la suite était statistiquement attendue dans son corpus.
- **Il ne peut pas vérifier ce qu'il dit.** Rien dans la boucle ci-dessus ne compare
  la phrase produite à une source. C'est l'origine structurelle des hallucinations
  (voir [chapitre 05](../05-llm/README.md)) et la raison d'etre du RAG
  ([chapitre 07](../07-rag/README.md)).
- **Deux paramètres pilotent le comportement** : la fenêtre de contexte (ce qu'il
  peut lire) et la température (à quel point il prend des risques au tirage).

## 5. L'IA est déjà partout (et pas sous forme de chatbot)

Lister ce qui utilise déjà un modèle est le meilleur moyen de sortir du fantasme :

- **Antispam et antifraude** : ML classique, souvent des arbres boostés, très rapides.
- **Recommandation** : contenus, produits, films, personnes. Un des plus gros usages
  industriels, invisible et très rentable.
- **Tri et priorisation** : files de support, tri de CV en amont, détection de
  documents anormaux.
- **Photographie computationnelle** : réduire le bruit, améliorer la netteté, choisir
  la meilleure pose sur un smartphone.
- **Transports** : prevision de trafic, optimisation d'itinéraires, aide à la conduite.
- **Santé** : aide à la lecture d'imagerie, priorisation de dossiers, aide à la
  rédaction de comptes rendus (avec validation humaine).
- **Code** : completion, génération de tests, revue automatique, migration de code.
- **Recherche scientifique** : prediction de structures de protéines, criblage de
  molécules, accélération de simulations.

Une remarque qui vaut son pesant de réalisation : ces systèmes sont souvent plus
performants, plus borels et moins « spectaculaires » que les LLM, parce qu'ils ont
été évalués sur une tâche précise et mesurable.

## 6. Cinq idées reçues qui coûtent cher

| Idée reçue | Ce qui est vrai |
|---|---|
| « L'IA comprend ce qu'elle dit » | Un LLM manipule des régularités statistiques. Il produit du texte plausible, sans modèle du monde vérifie ni intention. Le mot « comprendre » est au mieux métaphorique. |
| « Si ça hallucine, c'est inutilisable » | L'hallucination est inhérente à la génération probabiliste. On la réduit par l'ancrage documentaire (RAG), les outils, les citations obligatoires et la vérification. Aucune de ces méthodes ne la supprimé a 100 pour cent. |
| « L'AGI arrive dans deux ans » | Personne n'a de définition opérationnelle de l'AGI, donc personne ne peut mesurer son arrivée. Les systèmes actuels échouent sur des tâches élémentaires des qu'on change de contexte. |
| « Plus de paramètres = meilleur modèle » | Vrai à budget de calcul égal, faux en absolu. La qualité des données, le post-entraînement et le budget de raisonnement comptent autant. Un petit modèle bien affiné bat un gros modèle générique sur une tâche étroite. |
| « Un modèle, une réponse juste » | La même question peut produire des réponses différentes : le décodage est stochastique. Un système fiable doit etre conçu pour tolérer cette variabilité (tests, redondance, vérification). |

## Ce qu'il faut retenir

- L'IA englobe le ML, qui englobe le deep learning, qui englobe l'IA générative : ce
  sont des sous-ensembles emboîtés, pas des synonymes.
- Le renversement du ML est méthodologique : on fournit des exemples au lieu d'écrire
  des règles, et le modèle ajuste ses paramètres pour minimiser une erreur.
- L'histoire de l'IA alterne vagues d'enthousiasme et hivers : chaque vague a laissé
  des résultats réels, plus modestes que les annonces.
- Un LLM ne fait qu'une chose : prédire le token suivant, de façon autorégressive,
  à partir de tout ce qui se trouve dans son contexte.
- Cette mécanique explique à la fois la fluidité du texte produit et l'absence de
  garantie de vérité : les hallucinations sont structurelles, pas un bug.
- L'IA est déjà massivement déployée sous des formes peu spectaculaires :
  antispam, recommandation, tri, prevision.
- Les trois leviers d'un modèle sont le compute, les données et les paramètres ;
  le quatrième, souvent décisif en pratique, est la qualité de l'évaluation.

## Erreurs fréquentes / idées reçues

- « IA = LLM » -> faux : les LLM sont une famille de modèles parmi beaucoup d'autres,
  et l'essentiel du ML industriel n'est pas génératif.
- « Le modèle a appris par cœur ma question » -> il a souvent vu une formulation
  proche ; cela n'implique ni raisonnement ni recherche dans une base.
- « Le ML découvre la causalité » -> il découvre des corrélations dans une
  distribution donnée. Des que la distribution change, la performance s'effondre.
- « Un modèle plus gros résout le problème » -> passer d'un modèle à un autre ne
  remplace ni une bonne évaluation, ni de bonnes données, ni un bon découpage produit.
- « L'IA remplace les développeurs » -> elle déplace le travail : moins de code
  boilerplate, plus de spécification, de relecture et de tests.

## Pour aller plus loin

- [Chapitre 01 : fondamentaux du machine learning](../01-fondamentaux/README.md)
- [Chapitre 03 : réseaux de neurones](../03-reseaux-de-neurones/README.md)
- [Chapitre 04 : transformers](../04-transformers/README.md)
- [Chapitre 05 : grands modèles de langage](../05-llm/README.md)
- [Chapitre 17 : frontières et débats](../17-au-dela/README.md)
- Document fondateur : « Computing Machinery and Intelligence », Alan Turing, 1950.
- Papier fondateur du transformer : « Attention Is All You Need », 2017.
- Cours de référence en ligne : « Practical Deep Learning for Coders » (fast.ai) pour
  l'entrée en matière pratique.
