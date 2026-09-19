# Frontières et débats

> Explorer les sujets de recherche ouverts et les controverses de l'intelligence artificielle : quantique, neuro-symbolique, modèles du monde, débat sur l'AGI, alignement, interprétabilité et tendances de fond.

## 1. Introduction : distinguer faits, promesses et spéculations

Ce chapitre traite de sujets où l'incertitude est élevée.
Il faut distinguer trois registres :
*   **Les faits établis** : des résultats reproductibles, publiés et vérifiés indépendamment.
*   **Les promesses** : des démonstrations préliminaires, souvent spectaculaires, mais non encore industrialisables.
*   **Les spéculations** : des extrapolations sur ce que la technologie pourrait devenir, sans garantie.

La règle de lecture est simple : quand un sujet touche à l'AGI, à la conscience des machines ou à la supériorité imminente, cherche des preuves empiriques avant d'accepter une affirmation.
Le domaine est plein d'annonces tapageuses, rares sont celles qui survivent à un an de tests.

## 2. Informatique quantique et machine learning

L'informatique quantique exploite des qubits, des états physiques pouvant être en superposition (à la fois 0 et 1) et intriqués (corrélés à distance).
Elle promet, pour certaines tâches, une accélération théorique massive par rapport au calcul classique.

La réalité actuelle est plus nuancée :
*   Les ordinateurs quantiques existants comptent de quelques dizaines à quelques centaines de qubits physiques, très bruités.
*   La correction d'erreur quantique exige un grand nombre de qubits physiques par qubit logique, ce qui reste hors de portée pour des applications à grande échelle.
*   Les avantages pratiques ne sont démontrés que sur des problèmes très spécifiques (par exemple la simulation de systèmes quantiques ou certains algorithmes d'optimisation théoriques).

Sur le machine learning, le lien est double :
*   **Le quantique au service du ML** : accélérer l'entraînement ou l'inférence sur des problèmes particuliers (algèbre linéaire, échantillonnage).
*   **Le ML au service du quantique** : concevoir de meilleurs qubits, calibrer des portes, corriger les erreurs.

Il faut donc être honnête : il n'y a pas, à ce jour, de démonstration qu'un ordinateur quantique entraîne un LLM plus vite qu'un GPU de manière utile.
La quantique n'est pas un raccourci magique pour l'IA.

## 3. Neuro-symbolique et raisonnement formel

L'approche neuro-symbolique combine les réseaux de neurones (apprentissage par les données) et les systèmes symboliques (règles, logique, graphes de connaissances).

*   **Points forts des réseaux de neurones** : reconnaissance de motifs, robustesse au bruit, généralisation sur des données brutes.
*   **Points forts des systèmes symboliques** : raisonnement déductif exact, explicabilité, respect de règles formelles.

Leur combinaison vise un meilleur raisonnement : un réseau propose des hypothèses, un vérificateur formel les valide ou les rejette.
Une application emblématique est la preuve de théorèmes assistée ou automatique, où un modèle génère des étapes et un système vérifie leur validité.
Les vérificateurs de code (type checking, model checking) suivent la même logique.

L'arbitrage est clair : plus on met de symbolique, plus on gagne en exactitude et en explicabilité, mais moins on sait gérer des entrées bruitées.
La recherche actuelle cherche à brancher des vérificateurs externes sur des LLM pour filtrer leurs erreurs, par exemple en génération de code ou en arithmétique.

## 4. World models et IA incarnée

Un world model (modèle du monde) est une représentation interne qu'un agent apprend de son environnement pour prédire les conséquences de ses actions.
C'est une idée ancienne en robotique et en apprentissage par renforcement, remise au goût du jour par les modèles génératifs vidéo.

*   **IA incarnée** : intelligence qui apprend en interagissant physiquement avec le monde, par exemple via un robot.
*   **World models génératifs** : modèles capables de simuler des trajectoires futures plausibles (comme les modèles vidéo qui prédisent la suite d'une scène).

Les promesses sont importantes pour la robotique, où les données réelles sont rares et coûteuses à collecter.
L'idée est d'entraîner des agents dans un simulateur appris, puis de transférer les compétences au monde réel.
Les difficultés restent majeures : l'écart entre simulation et réalité (reality gap), la robustesse face à l'imprévu, et la sécurité des actions physiques.

## 5. Modèles de fondation pour la science

L'IA change la recherche scientifique en devenant un outil de prédiction et de découverte.

*   **AlphaFold et AlphaFold 3** (DeepMind) :
    Prédiction de la structure tridimensionnelle des protéines, puis de leurs interactions avec d'autres molécules.
    C'est un succès largement reconnu, qui a transformé la biologie structurale.
*   **GNoME** (DeepMind) :
    Découverte de matériaux inorganiques stables prédits par des modèles graphiques à grande échelle.
*   **Prévision météo (GraphCast, Pangu, FourCastNet)** :
    Des modèles de réseaux de neurones prédisent la météo à moyen terme, souvent plus vite que les modèles physiques classiques, à précision comparable.

La leçon de ces projets est que l'IA excelle quand les données sont abondantes et la physique partiellement connue.
Elle ne remplace pas la théorie, elle l'accélère en proposant des candidats à vérifier.

| Projet | Domaine | Apport principal | Statut |
|---|---|---|---|
| AlphaFold | Biologie structurale | Structure 3D des protéines | Déployé, largement utilisé |
| AlphaFold 3 | Interactions moléculaires | Complexes protéine-ligand, ADN | Recherche avancée |
| GNoME | Science des matériaux | Prédiction de matériaux stables | Résultats publiés, vérification en cours |
| GraphCast / Pangu | Météorologie | Prévision à moyen terme accélérée | Opérationnel chez plusieurs agences |

## 6. Le débat sur l'AGI

L'AGI (Artificial General Intelligence, intelligence artificielle générale) désigne une IA capable d'apprendre et d'accomplir des tâches cognitives variées au niveau humain, sans spécialisation préalable.

**Définitions concurrentes** :
*   Une IA qui réussit toute tâche intellectuelle humaine.
*   Une IA qui atteint un niveau de revenu économique comparable à un travailleur humain.
*   Une IA autonome capable de fixer ses propres objectifs.

**Échelles de niveaux** :
Plusieurs grilles de progression existent, proposant des paliers allant des systèmes à compétence nulle jusqu'à des systèmes surpassant l'humain sur tous les plans.
Ces échelles sont utiles pour discuter, mais elles reposent sur des critères arbitraires.

**Arguments pour une AGI proche** :
*   Les modèles progressent vite sur les benchmarks de raisonnement.
*   Les architectures sont générales et continuent de scaler.
*   Les agents multimodaux commencent à enchaîner des tâches longues.

**Arguments contre une AGI proche** :
*   Les benchmarks sont vite saturés et ne mesurent pas la généralisation réelle.
*   Les modèles échouent sur des changements mineurs de distribution.
*   Le raisonnement causal et la planification à long terme restent fragiles.
*   L'apprentissage continu et l'ancrage dans le monde physique sont encore absents.

Personne ne sait aujourd'hui trancher, et toute affirmation catégorique dans un sens ou dans l'autre relève de la croyance.

## 7. Alignement et risque existentiel

L'alignement désigne l'ensemble des techniques visant à faire en sorte qu'un système d'IA poursuive les objectifs que nous voulons, y compris dans des situations non prévues.

**Arguments des partisans d'un risque majeur** :
*   Un système très capable optimisant un objectif mal spécifié peut causer des dégâts par effet de bord (problème de la "clôture", tool use non contrôlé).
*   La difficulté de spécifier des préférences humaines complètes rend l'objectif final fragile.
*   La vitesse de progression dépasse celle de nos mécanismes de contrôle.

**Critiques de cette position** :
*   Les scénarios catastrophistes reposent souvent sur des hypothèses fortes peu étayées.
*   Le risque concret et immédiat (désinformation, biais, usage militaire, concentration de pouvoir) est plus mesurable que le risque existentiel hypothétique.
*   Un focus excessif sur le risque lointain détourne l'attention des dégâts présents.

Il est possible de tenir les deux à la fois : travailler sur les risques immédiats et la sûreté des systèmes, sans adopter une eschatologie sur l'AGI.

## 8. Interprétabilité mécaniste

L'interprétabilité mécaniste cherche à comprendre ce qui se passe à l'intérieur d'un réseau de neurones, en remontant des comportements aux mécanismes internes.

*   **Superposition** :
    Les réseaux représentent plus de concepts que le nombre de dimensions disponibles, en superposant les directions dans l'espace des activations.
    C'est ce qui rend l'interprétation difficile : une direction codée n'est pas un concept unique et propre.
*   **Circuits** (circuits d'attention, MLP) :
    Des groupes de neurones qui accomplissent ensemble une fonction identifiable, par exemple détecter un motif ou copier un token.
*   **Sondes (probing)** :
    On entraîne un classifieur léger sur les activations internes pour tester si une information (par exemple une notion de relation) est présente.

Les techniques de visualisation d'attention et d'activation aident à l'intuition, mais l'attention n'est pas une explication complète.
Ce champ est encore jeune et ses résultats, bien que prometteurs, restent partiels.

| Notion | Définition courte | Intérêt |
|---|---|---|
| Superposition | Plus de concepts que de dimensions | Explique la difficulte d'interprétation |
| Circuit | Groupe de neurones a fonction identifiable | Permet de suivre un calcul précis |
| Sonde (probe) | Classifieur leger sur les activations | Teste la presence d'une information |

## 9. Tendances de fond

Plusieurs directions structurent l'évolution actuelle du domaine.

*   **Agents** : des systèmes qui utilisent des outils, planifient et agissent de manière autonome.
*   **MoE (Mixture of Experts)** : des modèles dont seule une partie des paramètres est activée par token, ce qui réduit le coût de calcul à capacité totale égale.
*   **Long contexte** : des fenêtres de contexte de centaines de milliers à plusieurs millions de tokens.
*   **Multimodal** : traitement conjoint du texte, de l'image, de l'audio et de la vidéo dans un même modèle.
*   **Edge** : exécution de modèles sur des appareils à ressources limitées (téléphones, ordinateurs portables).
*   **Poids ouverts contre fermés** : un rapport de force qui évolue rapidement et détermine l'accès à la technique.
*   **Régulation** : le AI Act européen et d'autres cadres imposent des obligations de transparence et de sécurité.

Ces tendances se renforcent mutuellement : les agents et le long contexte, par exemple, se combinent pour des tâches de recherche complexe.

## 10. Ce qui reste difficile

Malgré les progrès, plusieurs obstacles de fond persistent.

*   **Raisonnement** : les modèles excellent en induction statistique mais peinent sur la déduction rigoureuse et les chaînes longues.
*   **Vérité** : les modèles génèrent des affirmations plausibles sans distinguer le vrai du faux, d'où les hallucinations.
*   **Causalité** : distinguer corrélation et causation reste un problème ouvert, car les données d'entraînement sont observationnelles.
*   **Apprentissage continu** : un modèle ne continue pas d'apprendre après son entraînement sans réentraînement ou fine-tuning coûteux.
*   **Énergie** : l'entraînement et le service de modèles massifs consomment des quantités considérables de ressource (GPU, électricité, eau).

Ces limites ne sont pas toutes près d'être levées, et certaines pourraient être structurelles.

## 11. Comment se tenir à jour sans se noyer

Le rythme des publications est tel qu'il faut une stratégie de filtrage raisonnée.

*   **Suivre la recherche par résumé** : sélectionner quelques sources fiables plutôt que tout suivre.
*   **Privilégier les évaluations indépendantes** : comparer les modèles sur des benchmarks éloignés de la publicité.
*   **Tester par soi-même** : un test direct sur son propre cas d'usage vaut mieux que dix avis de forum.
*   **Distinguer la démo de la production** : une démo convaincante n'implique pas une fiabilité industrielle.
*   **Limiter le temps de veille** : bloquer un créneau hebdomadaire plutôt que consommer en continu.

La compétence durable n'est pas connaître chaque modèle du mois, mais comprendre les mécanismes et les ordres de grandeur, ce que ces chapitres s'attachent à transmettre.

## 12. Cas pratique : vérifier une affirmation sur un modèle

Face à une annonce de performance, une méthode de vérification systématique.

```text
Protocole de vérification d'une affirmation (ex: "ce modèle surpasse l'humain")

1. Quelle est la source ?
   -> papier revu par les pairs, preprint, blog marketing, tweet ?

2. Quelles sont les données de test ?
   -> jeu public standardisé ?
   -> jeu prive non contamine ?
   -> contient-il des exemples d'entraînement ?

3. Comment le score est-il mesure ?
   -> métrique unique ? plusieurs ?
   -> intervals de confiance donnes ?

4. Sur quelles tâches ?
   -> une tâche étroite ou un ensemble varie ?
   -> les tâches faciles dominent-elles la moyenne ?

5. Qui a évalué ?
   -> l'auteur du modèle ou un tiers indépendant ?

6. Les résultats sont-ils reproductibles ?
   -> code et poids publiés ?
   -> quelqu'un d'autre a-t-il reproduit le score ?

Règle : une affirmation forte exige une preuve forte.
En cas de doute, considerer l'affirmation comme non établie.
```

Ce protocole évite de prendre des annonces pour des faits observés.

## Ce qu'il faut retenir

- La quantique n'offre aujourd'hui aucun avantage pratique démontré pour l'entraînement de grands modèles.
- L'approche neuro-symbolique vise à combiner l'apprentissage neuronal et des vérificateurs formels pour un raisonnement plus fiable.
- Les world models et l'IA incarnée progressent, mais l'écart entre simulation et réalité physique reste important.
- AlphaFold, GNoME et les modèles météo montrent que l'IA accélère la science là où les données sont abondantes.
- Le débat sur l'AGI repose sur des définitions instables et aucune réponse consensuelle n'existe.
- L'alignement combine un risque existentiel hypothétique et des risque concrets immédiats, tous deux à prendre au sérieux.
- L'interprétabilité mécaniste explore la superposition, les circuits et les sondes, avec des résultats encore partiels.
- Les tendances dominantes sont les agents, les MoE, le long contexte, le multimodal, l'edge et la régulation.
- Le raisonnement, la vérité, la causalité, l'apprentissage continu et l'énergie restent des défis ouverts.
- Se tenir à jour passe par le filtrage, l'évaluation indépendante et le test personnel.

## Erreurs fréquentes / idées reçues

- L'ordinateur quantique va entraîner les LLM plus vite -> Faux. Aucun avantage pratique n'est démontré pour cette application.
- Plus de neurones signifie automatiquement plus de raisonnement -> Faux. La structure et les données comptent autant que la taille.
- L'AGI est imminente -> Idée reçue. Les échelles annoncées reposent sur des critères arbitraires, et aucun consensus n'existe.
- Les modèles comprennent le sens des mots comme un humain -> Faux. Ils manipulent des régularités statistiques sans ancrage dans le monde.
- L'alignement est uniquement un problème futur -> Faux. Les dégâts actuels (biais, désinformation, usage abusif) sont mesurables dès maintenant.
- Une démo impressionnante suffit à prouver une capacité -> Faux. Une démo peut refléter une sur-spécialisation ou une fuite de données.

## Pour aller plus loin

- [chapitre 14](../14-ethique-societe/README.md) : Pour approfondir les enjeux éthiques, juridiques et l'AI Act.
- [chapitre 05](../05-llm/README.md) : Pour situer les capacités réelles des LLM avant d'aborder les débats sur l'AGI.
- [chapitre 08](../08-agents/README.md) : Pour comprendre les systèmes d'agents, une des tendances majeures.
- [chapitre 03](../03-reseaux-de-neurones/README.md) : Pour les fondations sur les réseaux de neurones et les architectures génératives.
- Site officiel du DeepMind AlphaFold (deepmind.google) : Documentation sur la prédiction de structures de protéines.
- Anthropic, "Core Views on AI Safety" : Une présentation argumentée des positions sur le risque et l'alignement.
- Anthropic, "Circuit Tracing" et travaux d'interprétabilité mécaniste : Exemples d'analyse des mécanismes internes des modèles.
- State of AI Report (stateof.ai) : Synthèse annuelle des avancées et des débats du domaine.
