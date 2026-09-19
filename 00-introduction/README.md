# Introduction à l'intelligence artificielle

> Ce chapitre pose le vocabulaire commun et la carte du territoire : ce qu'on appelle
> IA, machine learning, deep learning et IA generative, d'ou vient tout ca, et comment
> fonctionne un grand modele de langage vu de loin. Il sert de socle aux dix-sept
> chapitres suivants.

## 1. Quatre mots qu'on confond tout le temps

« IA » n'est pas un synonyme de « ChatGPT », et « machine learning » n'est pas un
synonyme de « deep learning ». Ces termes s'emboitent comme des poupees russes :
chacun est un sous-ensemble du precedent.

```
+-----------------------------------------------------------+
|  INTELLIGENCE ARTIFICIELLE (IA)                           |
|  Toute technique faisant faire a une machine une tache    |
|  qui exige normalement de l'intelligence humaine.         |
|                                                           |
|   +---------------------------------------------------+   |
|   |  MACHINE LEARNING (ML)                            |   |
|   |  Sous-ensemble de l'IA ou la machine APPREND       |   |
|   |  a partir d'exemples au lieu de suivre des regles  |   |
|   |  ecrites a la main.                                |   |
|   |                                                    |   |
|   |   +------------------------------------------+     |   |
|   |   |  DEEP LEARNING (DL)                      |     |   |
|   |   |  ML fonde sur des reseaux de neurones     |     |   |
|   |   |  profonds, qui apprennent eux-memes       |     |   |
|   |   |  leurs representations.                   |     |   |
|   |   |                                          |     |   |
|   |   |   +----------------------------------+   |     |   |
|   |   |   |  IA GENERATIVE                   |   |     |   |
|   |   |   |  Produit du contenu neuf :       |   |     |   |
|   |   |   |  texte, image, audio, code.      |   |     |   |
|   |   |   +----------------------------------+   |     |   |
|   |   +------------------------------------------+     |   |
|   +---------------------------------------------------+   |
+-----------------------------------------------------------+
```

**Intelligence artificielle (IA)** : terme parapluie, apparu dans les annees 1950.
Il couvre aussi bien les systemes a base de regles logiques (IA symbolique : « si
temperature > 30 alors declencher l'alarme ») que les reseaux de neurones modernes.
Une grande partie de l'IA industrielle d'aujourd'hui n'est pas « intelligente » au
sens philosophique : ce sont des systemes experts, des planificateurs, des moteurs
de recherche de solution.

**Machine learning (ML, apprentissage automatique)** : au lieu d'ecrire les regles,
on fournit des exemples et l'algorithme ajuste ses parametres internes pour minimiser
une erreur. C'est un renversement de methode : le programmeur decrit *ce qu'il veut*,
pas *comment le faire*. Un modele de detection de spam n'a pas de regle « si le mot
'viagra' apparait... » : il a des poids numeriques appris sur des milliers d'emails.

**Deep learning (DL, apprentissage profond)** : ML base sur des reseaux de neurones
a nombreuses couches. La difference essentielle avec le ML classique : le modele
apprend lui-meme les variables intermediaires utiles (*representation learning*),
la ou le ML classique demande de les construire a la main (*feature engineering*).
Voir [chapitre 01](../01-fondamentaux/README.md) et [chapitre 03](../03-reseaux-de-neurones/README.md).

**IA generative** : sous-ensemble du DL qui produit du contenu nouveau plutot que de
classer ou de predire. Elle modelise la distribution statistique des donnees
d'entrainement et echantillonne dedans. Elle regroupe les grands modeles de langage
(texte, code), les modeles de diffusion (image, video) et les modeles audio.

Un point qui evite beaucoup de confusion : **l'IA generative n'est pas la seule IA
utile**. Un modele de detection de fraude, un systeme de recommandation ou un
algorithme de prevision de trafic ne generent rien et rendent pourtant des services
considerables.

## 2. Une histoire en cinq actes

L'histoire de l'IA n'est pas une progression continue : c'est une alternance de
vagues d'enthousiasme et de « hivers », quand les promesses depassent les capacites
reelles.

| Periode | Acte | Ce qui se passe | Ce qui casse |
|---|---|---|---|
| 1950-1970 | Naissance symbolique | Test de Turing (1950), conference de Dartmouth (1956) ou le terme « intelligence artificielle » est invente, premiers programmes de demonstration de theoremes, perceptron (1958) | Minsky et Papert montrent en 1969 les limites du perceptron simple (il ne resout pas XOR) |
| 1974-1980 | Premier hiver | Les promesses de traduction automatique et de robotique ne sont pas tenues, les financements s'arretent | Complexite de calcul insuffisante, donnees inexistantes |
| 1980-1987 | Systemes experts | Programmes a base de regles metier, tres efficaces dans des domaines etroits (diagnostic, configuration) et couteux a maintenir | Fragilite hors du domaine, cout de maintenance, deuxieme hiver |
| 1990-2010 | ML statistique | Les approches probabilistes et l'apprentissage statistique prennent le dessus : SVM, arbres, methodes ensemblistes. Succes industriels reels (spam, credit scoring, recommandation) | Les representations restent construites a la main |
| 2012-2017 | Revolution du deep learning | AlexNet (2012) ecrase la competition ImageNet avec un reseau convolutif et deux GPU. En 2016, AlphaGo bat un champion du monde de Go | Cout de calcul, besoin massif de donnees annotees |
| 2017-2022 | Ere du transformer | Le papier « Attention Is All You Need » (2017) remplace la recurrence par l'attention. GPT, BERT puis les premiers LLM suivent | Les modeles hallucinent, couts d'entrainement faramineux |
| 2022-2026 | Ere generative et agents | ChatGPT popularise l'usage grand public, les poids ouverts se multiplient, les modeles deviennent multimodaux, le RAG et les agents se banalisent | Fiabilite, securite, droit, cout energetique |

Deux lecons de cette chronologie :

1. **Le goulot d'etranglement a change trois fois.** D'abord les donnees (annees 1980),
   puis la puissance de calcul (annees 2000), puis l'architecture (2017). Aujourd'hui
   le goulot est surtout la fiabilite et le cout.
2. **Chaque vague a produit des resultats reels**, plus modestes que les annonces,
   mais persistants. Les systemes experts ne sont pas morts : ils tournent encore
   dans l'assurance et l'aeronautique.

## 3. Cartographie : a quoi sert chaque sous-domaine

| Sous-domaine | Entree | Sortie typique | Exemples concrets |
|---|---|---|---|
| Vision par ordinateur | Image, video | Classe, boites, masques, description | Controle qualite industriel, lecture de plaques, imagerie medicale |
| Traitement du langage (NLP) | Texte | Classe, entites, resume, traduction, reponse | Anti-spam, extraction d'information, assistants, traduction |
| Parole (speech) | Signal audio | Transcription, locuteur, commande | Sous-titrage automatique, assistants vocaux, transcription de reunions |
| Apprentissage par renforcement | Etat d'un environnement | Action | Jeux, robotique, optimisation de ressources, pilotage |
| Graphes | Noeud et aretes | Prediction sur les noeuds ou les aretes | Detection de fraude en reseau, decouverte de molecules |
| Series temporelles | Historique date | Valeur future, anomalie | Prevision de consommation, maintenance predictive, marche |
| Recommandation | Interactions utilisateur/objet | Liste ordonnee | Films, musique, produits, contenus |
| IA generative | Prompt, bruit, donnees | Nouveau contenu | LLM, images, voix, code |

Ces domaines partagent leurs briques : les memes optimiseurs, les memes GPU, souvent
le meme type d'architecture. Un progres sur les transformers se repercute en vision,
en audio et en biologie.

## 4. Comment fonctionne un LLM, en vue gros grain

Pas besoin de maths pour comprendre l'essentiel. Un grand modele de langage fait
une seule chose, en boucle : **predire le token suivant**.

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
  |   TRANSFORMER   |  des dizaines de couches qui melangent
  |   (N couches)   |  l'information entre tous les tokens
  +-----------------+
        |
        v
  distribution de probabilite sur tout le vocabulaire :
     " Paris"  0.87   " Lyon"  0.04   " une"  0.02   ...
        |
        v
  on tire un token (ou on prend le plus probable)
        |
        v
  le token est ajoute a la suite, et on recommence
```

Trois consequences directes de cette mecanique :

- **Le modele ne « sait » rien au sens d'une base de donnees.** Il a compresse des
  regularites statistiques du langage dans des milliards de nombres. Quand il repond
  juste, c'est que la suite etait statistiquement attendue dans son corpus.
- **Il ne peut pas verifier ce qu'il dit.** Rien dans la boucle ci-dessus ne compare
  la phrase produite a une source. C'est l'origine structurelle des hallucinations
  (voir [chapitre 05](../05-llm/README.md)) et la raison d'etre du RAG
  ([chapitre 07](../07-rag/README.md)).
- **Deux parametres pilotent le comportement** : la fenetre de contexte (ce qu'il
  peut lire) et la temperature (a quel point il prend des risques au tirage).

## 5. L'IA est deja partout (et pas sous forme de chatbot)

Lister ce qui utilise deja un modele est le meilleur moyen de sortir du fantasme :

- **Antispam et antifraude** : ML classique, souvent des arbres boosts, tres rapides.
- **Recommandation** : contenus, produits, films, personnes. Un des plus gros usages
  industriels, invisible et tres rentable.
- **Tri et priorisation** : files de support, tri de CV en amont, detection de
  documents anormaux.
- **Photographie computationnelle** : reduire le bruit, ameliorer la nettete, choisir
  la meilleure pose sur un smartphone.
- **Transports** : prevision de trafic, optimisation d'itineraires, aide a la conduite.
- **Sante** : aide a la lecture d'imagerie, priorisation de dossiers, aide a la
  redaction de comptes rendus (avec validation humaine).
- **Code** : completion, generation de tests, revue automatique, migration de code.
- **Recherche scientifique** : prediction de structures de proteines, criblage de
  molecules, acceleration de simulations.

Une remarque qui vaut son pesant de realisation : ces systemes sont souvent plus
performants, plus borels et moins « spectaculaires » que les LLM, parce qu'ils ont
ete evalues sur une tache precise et mesurable.

## 6. Cinq idees recues qui coutent cher

| Idee recue | Ce qui est vrai |
|---|---|
| « L'IA comprend ce qu'elle dit » | Un LLM manipule des regularites statistiques. Il produit du texte plausible, sans modele du monde verifie ni intention. Le mot « comprendre » est au mieux metaphorique. |
| « Si ca hallucine, c'est inutilisable » | L'hallucination est inherente a la generation probabiliste. On la reduit par l'ancrage documentaire (RAG), les outils, les citations obligatoires et la verification. Aucune de ces methodes ne la supprime a 100 pour cent. |
| « L'AGI arrive dans deux ans » | Personne n'a de definition operationnelle de l'AGI, donc personne ne peut mesurer son arrivee. Les systemes actuels echouent sur des taches elementaires des qu'on change de contexte. |
| « Plus de parametres = meilleur modele » | Vrai a budget de calcul egal, faux en absolu. La qualite des donnees, le post-entrainement et le budget de raisonnement comptent autant. Un petit modele bien affine bat un gros modele generique sur une tache etroite. |
| « Un modele, une reponse juste » | La meme question peut produire des reponses differentes : le decodage est stochastique. Un systeme fiable doit etre concu pour tolerer cette variabilite (tests, redondance, verification). |

## Ce qu'il faut retenir

- L'IA englobe le ML, qui englobe le deep learning, qui englobe l'IA generative : ce
  sont des sous-ensembles emboites, pas des synonymes.
- Le renversement du ML est methodologique : on fournit des exemples au lieu d'ecrire
  des regles, et le modele ajuste ses parametres pour minimiser une erreur.
- L'histoire de l'IA alterne vagues d'enthousiasme et hivers : chaque vague a laisse
  des resultats reels, plus modestes que les annonces.
- Un LLM ne fait qu'une chose : predire le token suivant, de facon autoregressive,
  a partir de tout ce qui se trouve dans son contexte.
- Cette mecanique explique a la fois la fluidite du texte produit et l'absence de
  garantie de verite : les hallucinations sont structurelles, pas un bug.
- L'IA est deja massivement deployee sous des formes peu spectaculaires :
  antispam, recommandation, tri, prevision.
- Les trois leviers d'un modele sont le compute, les donnees et les parametres ;
  le quatrieme, souvent decisif en pratique, est la qualite de l'evaluation.

## Erreurs frequentes / idees recues

- « IA = LLM » -> faux : les LLM sont une famille de modeles parmi beaucoup d'autres,
  et l'essentiel du ML industriel n'est pas generatif.
- « Le modele a appris par coeur ma question » -> il a souvent vu une formulation
  proche ; cela n'implique ni raisonnement ni recherche dans une base.
- « Le ML decouvre la causalite » -> il decouvre des correlations dans une
  distribution donnee. Des que la distribution change, la performance s'effondre.
- « Un modele plus gros resout le probleme » -> passer d'un modele a un autre ne
  remplace ni une bonne evaluation, ni de bonnes donnees, ni un bon decoupage produit.
- « L'IA remplace les developpeurs » -> elle deplace le travail : moins de code
  boilerplate, plus de specification, de relecture et de tests.

## Pour aller plus loin

- [Chapitre 01 : fondamentaux du machine learning](../01-fondamentaux/README.md)
- [Chapitre 03 : reseaux de neurones](../03-reseaux-de-neurones/README.md)
- [Chapitre 04 : transformers](../04-transformers/README.md)
- [Chapitre 05 : grands modeles de langage](../05-llm/README.md)
- [Chapitre 17 : frontieres et debats](../17-au-dela/README.md)
- Document fondateur : « Computing Machinery and Intelligence », Alan Turing, 1950.
- Papier fondateur du transformer : « Attention Is All You Need », 2017.
- Cours de reference en ligne : « Practical Deep Learning for Coders » (fast.ai) pour
  l'entree en matiere pratique.
