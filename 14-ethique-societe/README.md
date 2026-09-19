# Éthique, droit et société

> L'essor de l'intelligence artificielle pose des défis éthiques, juridiques et sociétaux. Ce chapitre couvre les biais algorithmiques, le RGPD, l'EU AI Act, le droit d'auteur, la vie privée, le travail, l'environnement, la désinformation et la gouvernance responsable.

## 1. Introduction et synthèse

L'intégration massive de l'IA dans la société soulève des questions qui dépassent la technique. Entre les biais discriminatoires enfouis dans les algorithmes, les litiges de propriété intellectuelle liés au pré-entraînement, le bouleversement du marché du travail et l'empreinte écologique des centres de données, la gouvernance de l'IA est devenue un enjeu politique et éthique de premier plan.

Cette section donne une vue d'ensemble. Les sections suivantes détaillent chaque domaine avec des ordres de grandeur, des cas réels et les arbitrages explicites.

### Points clés de synthèse
- Les biais proviennent de trois sources : les données collectées, la formulation de l'objectif, et les conditions de déploiement. Ils produisent des discriminations systémiques mesurables.
- Les types de biais les plus fréquents sont le biais de représentation, le biais historique et le biais d'agrégation.
- Des métriques formelles (parité démographique, égalité des chances) quantifient ces disparités, mais elles sont mathématiquement incompatibles entre elles dans le cas général.
- Le RGPD encadre les données personnelles : minimisation, droit à l'explication, analyse d'impact (DPIA) pour les traitements à risque.
- L'EU AI Act classe les systèmes par niveau de risque (inacceptable, haut, limité, minimal) et impose des obligations proportionnées, avec des amendes jusqu'à 35 M EUR ou 7 % du chiffre d'affaires mondial.
- Le droit d'auteur est contesté : des procès portent sur l'usage d'œuvres protégées pour l'entraînement, et la doctrine du fair use n'est pas tranchée.
- La vie privée est menacée par la mémorisation, l'extraction de PII et les attaques par inférence d'appartenance. Le watermarking et le standard C2PA tentent d'y répondre partiellement.
- L'impact sur le travail est avant tout une automatisation de tâches, avec un effet d'augmentation de la productivité plutôt qu'un remplacement complet des métiers.
- L'empreinte environnementale est dominée, sur le cycle de vie, par l'inférence cumulée plus que par l'entraînement.
- La désinformation, les chambres d'écho et la sycophancie des LLM fragilisent le débat public.

### Idées reçues et réalités
- **Idée reçue** : Un algorithme est neutre car il repose sur des mathématiques.  
  -> **Réalité** : il reproduit et amplifie les biais présents dans ses données et dans la définition de son objectif.
- **Idée reçue** : Le RGPD interdit d'utiliser des données personnelles pour l'IA.  
  -> **Réalité** : il l'autorise sous conditions (consentement, intérêt légitime, anonymisation) et impose des garanties.
- **Idée reçue** : L'AI Act en Europe va paralyser l'innovation.  
  -> **Réalité** : il interdit uniquement quelques usages inacceptables et encadre proportionnellement le reste.
- **Idée reçue** : Les créateurs peuvent toujours bloquer l'ingestion de leurs œuvres.  
  -> **Réalité** : les mécanismes d'opt-out existent mais leur portée juridique et technique reste fragmentée.
- **Idée reçue** : L'IA va remplacer la majorité des métiers à court terme.  
  -> **Réalité** : l'effet observé est l'automatisation de tâches et l'augmentation de la productivité.
- **Idée reçue** : L'empreinte carbone de l'IA se limite à l'entraînement.  
  -> **Réalité** : sur la durée de vie, l'inférence cumulée domine largement.

---

## 2. Biais algorithmiques et mesures d'équité

> **Définition** : un biais algorithmique est une déviation systématique et injuste des prédictions d'un modèle au détriment d'un groupe défini par une caractéristique protégée (genre, origine, âge, handicap).

```
          [ Biais sociétaux et historiques ]
                        |
                        v
            [ Données d'entraînement ]
                        |
                        v
  [ Objectif et fonction de perte ] --> [ Modèle appris ] --> [ Décision déployée ]
                                                                    |
                                                                    v
                                                      [ Impact inéquitable ]
```

### Sources du biais
Le biais peut entrer à trois moments distincts du cycle de vie :
- **Données** : le corpus d'entraînement ne représente pas la population cible, ou contient des étiquettes bruitées et partiales.
- **Objectif** : la fonction de perte optimise une proxy qui ne correspond pas à l'équité voulue. Optimiser la précision globale peut dégrader la précision sur un groupe minoritaire.
- **Déploiement** : le contexte d'usage réel diffère du contexte d'entraînement (dérive de distribution, feedback loops). Un modèle qui décide quelles données collecter fausse ses propres données futures.

### Typologie des biais
1. **Biais historique** : les données reflètent des discriminations passées. Un modèle de tri de CV entraîné sur des embauches passées apprend les préférences passées.
2. **Biais de représentation** : certaines sous-populations sont sous-échantillonnées, ce qui dégrade la précision sur ces segments.
3. **Biais de mesure** : une variable utilisée comme proxy capture mal la réalité (par exemple le code postal comme proxy de solvabilité, qui corrèle avec l'origine ethnique).
4. **Biais d'agrégation** : un modèle unique est appliqué à des populations dont les distributions sont différentes ; la moyenne globale masque les écarts.
5. **Biais d'exclusion** : attributs pertinents pour un groupe mais absents du modèle, ce qui pénalise ce groupe.

### Cas concrets
- **Crédit** : des cartes de crédit ont accordé des plafonds différents à des conjoints aux revenus identiques, révélant un biais de genre non maîtrisé dans la décision.
- **Recrutement** : un outil expérimental de tri de CV pénalisait des candidatures féminines à cause de termes corrélés au genre (clubs d'échecs féminins, etc.).
- **Justice prédictive** : des outils d'évaluation du risque de récidive ont montré des taux de faux positifs nettement plus élevés pour certains groupes ethniques.
- **Santé** : des systèmes d'analyse d'images médicales entraînés majoritairement sur des peaux claires produisent plus d'erreurs sur les peaux foncées.

Ces cas illustrent que le biais n'est pas un accident isolé mais une propriété structurelle des systèmes entraînés sur des données humaines imparfaites.

### Métriques d'équité
On note $Y \in \{0,1\}$ la vérité, $\hat{Y} \in \{0,1\}$ la prédiction, et $A$ l'attribut sensible.

- **Parité démographique (Demographic Parity)** : la probabilité de décision positive est indépendante du groupe.
  $$P(\hat{Y}=1 \mid A=a) = P(\hat{Y}=1 \mid A=b)$$
- **Égalité des chances (Equal Opportunity)** : le taux de vrais positifs est égal entre groupes.
  $$P(\hat{Y}=1 \mid Y=1, A=a) = P(\hat{Y}=1 \mid Y=1, A=b)$$
- **Cotes égalisées (Equalized Odds)** : égalité simultanée des taux de vrais positifs et de faux positifs.
  $$P(\hat{Y}=1 \mid Y=y, A=a) = P(\hat{Y}=1 \mid Y=y, A=b) \quad \forall y \in \{0,1\}$$
- **Impact disparate (Disparate Impact)** : ratio de sélection entre groupes, avec le seuil usuel de 0,8.
  $$\text{DI} = \frac{P(\hat{Y}=1 \mid A=a)}{P(\hat{Y}=1 \mid A=b)} \ge 0.8$$

Point important : ces critères sont en général **incompatibles**. Si les taux de base diffèrent entre groupes, on ne peut pas satisfaire simultanément la parité démographique et l'égalité des chances. Il faut donc choisir explicitement la notion d'équité pertinente pour le contexte, et documenter ce choix.

### Atténuation : pré, intra, post-traitement
- **Prétraitement** : rééchantillonnage équilibré, repondération des instances, suppression des corrélations avec l'attribut sensible. Avantage : agit à la source. Limite : peut détruire de l'information utile.
- **Intra-traitement** : ajout d'un terme de pénalité d'inéquité dans la fonction de perte, ou optimisation sous contrainte (adversarial debiasing). Avantage : intégré à l'apprentissage. Limite : coût de mise au point, dégradation possible de la précision globale.
- **Post-traitement** : ajustement de seuils de décision différenciés par groupe. Avantage : simple, s'applique à un modèle figé. Limite : nécessite de connaître le groupe à l'inférence, ce qui peut être interdit ou sensible.

```python
# Calcul de quelques métriques d'équité en Python
import numpy as np

def fairness_metrics(y_true, y_pred, sensitive):
    """Renvoie parité démographique, impact disparate et égalité des chances."""
    g0 = sensitive == 0
    g1 = sensitive == 1

    pos0 = np.mean(y_pred[g0] == 1)
    pos1 = np.mean(y_pred[g1] == 1)
    dp_diff = abs(pos0 - pos1)
    di = min(pos0, pos1) / (max(pos0, pos1) + 1e-9)

    tpr0 = np.mean(y_pred[g0 & (y_true == 1)] == 1)
    tpr1 = np.mean(y_pred[g1 & (y_true == 1)] == 1)
    eo_diff = abs(tpr0 - tpr1)

    return {"dp_diff": float(dp_diff), "di": float(di), "eo_diff": float(eo_diff)}

y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
sens   = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

for k, v in fairness_metrics(y_true, y_pred, sens).items():
    print(f"{k}: {v:.4f}")
```

En pratique on utilise des bibliothèques dédiées (IBM AI Fairness 360, Google What-If Tool, Fairlearn) plutôt que de réimplémenter ces métriques.

---

## 3. RGPD et protection des données

Le RGPD (Règlement général sur la protection des données) s'applique à tout traitement de données personnelles concernant des personnes dans l'UE, y compris pour l'entraînement et l'inférence de modèles.

### Principes clés appliqués à l'IA
- **Licéité, loyauté, transparence** : une base légale est requise (consentement, contrat, intérêt légitime, obligation légale). L'entraînement sur des données aspirées du web sans base identifiée est fragile.
- **Limitation des finalités** : les données collectées pour un usage ne peuvent pas être réutilisées librement pour l'entraînement.
- **Minimisation** : ne collecter que les données nécessaires. Un modèle mémorise d'autant plus qu'il voit beaucoup d'exemples répétés.
- **Limitation de conservation** : les données ne doivent pas être gardées indéfiniment. Or un modèle entraîné conserve une forme de trace des données.
- **Intégrité et confidentialité** : chiffrement, contrôle d'accès, sécurité des jeux de données et des poids.

### Droit à l'explication et décisions automatisées
L'article 22 encadre les décisions entièrement automatisées produisant des effets juridiques ou significatifs. La personne concernée peut exiger une intervention humaine, exprimer son point de vue et contester la décision. Le « droit à l'explication » en découle : fournir une explication intelligible des facteurs ayant conduit à la décision.

Nuance importante : expliquer le comportement d'un grand réseau de neurones est difficile et souvent approximatif. Les techniques d'explicabilité (attributions, LIME, SHAP) donnent des indices, pas une preuve causale. Il faut donc être honnête sur leur portée.

### DPIA (analyse d'impact)
Une analyse d'impact relative à la protection des données est obligatoire pour les traitements susceptibles d'engendrer un risque élevé : profilage à grande échelle, surveillance systématique, traitement de données sensibles. Une DPIA décrit la nécessité du traitement, évalue les risques et prévoit les mesures d'atténuation.

### Arbitrage pratique
| Exigence RGPD | Ce qu'elle impose à un projet d'IA | Coût typique |
| :--- | :--- | :--- |
| Minimisation | Filtrer les données inutiles avant entraînement | Ingénierie de pipeline, perte potentielle de signal |
| Droit à l'explication | Fournir une justification compréhensible | Complexité d'API, explications approximatives |
| Minimisation de la conservation | Suppression, rédaction, ou ré-entraînement | Cycles de ré-entraînement coûteux |
| Sécurité | Chiffrement, contrôle d'accès, journalisation | Infra et processus |
| Droit à l'effacement | Supprimer l'influence d'une donnée | Difficile en pratique (machine unlearning coûteux) |

Le point le plus difficile reste le « droit à l'effacement » : retirer une donnée d'un modèle déjà entraîné n'est pas trivial. On parle de *machine unlearning*, domaine de recherche actif, encore coûteux et imparfait.

---

## 4. EU AI Act et régulation européenne

L'AI Act est le premier cadre réglementaire horizontal complet sur l'IA. Il adopte une approche par niveau de risque.

| Niveau | Définition | Exemples | Obligations |
| :--- | :--- | :--- | :--- |
| **Inacceptable** | Atteinte aux droits fondamentaux | Notation sociale, manipulation subliminale, biométrie temps réel dans l'espace public (sauf exceptions) | Interdiction totale de mise sur le marché |
| **Haut risque** | Domaines critiques pour la sécurité ou les droits | Recrutement, crédit, justice, tri médical, infrastructures critiques | Évaluation de conformité, gouvernance des données, documentation, journalisation, supervision humaine |
| **Risque limité** | Interaction avec l'humain ou contenu synthétique | Chatbots, deepfakes | Transparence : informer qu'il s'agit d'une IA, labelliser les contenus |
| **Risque minimal** | Sans impact notable | Filtres anti-spam, jeux vidéo | Aucune obligation spécifique |

### Fournisseurs de modèles à usage général (GPAI)
L'AI Act distingue les obligations selon le caractère généraliste et le niveau de calcul du modèle.
- **Tous les GPAI** : documentation technique, résumé public des données d'entraînement, politique de respect du droit d'auteur.
- **Modèles à risque systémique** : au-dessus d'un seuil de calcul de l'ordre de $10^{25}$ FLOPs, obligations renforcées (évaluations standardisées, red teaming, notification d'incidents graves, cybersécurité).

### Calendrier et sanctions
- **Calendrier** : application progressive, étalée sur plusieurs années après l'entrée en vigueur, avec des échéances distinctes selon les dispositions.
- **Sanctions** : jusqu'à 35 M EUR ou 7 % du chiffre d'affaires mondial pour les pratiques interdites ; jusqu'à 15 M EUR ou 3 % pour la non-conformité aux exigences des systèmes à haut risque.

L'objectif affiché est de concilier protection des droits et innovation : n'interdire que le clairement inacceptable, encadrer le reste proportionnellement.

### Autres cadres internationaux
- **États-Unis (NIST AI RMF)** : cadre volontaire organisé en quatre fonctions — *Govern*, *Map*, *Measure*, *Manage*. Approche non contraignante, orientée processus.
- **Royaume-Uni** : approche sectorielle déléguant la supervision aux régulateurs existants, sans législation horizontale unique.
- **Chine** : réglementations ciblées sur les recommandations algorithmiques, la synthèse profonde et l'IA générative, avec obligations de déclaration et d'alignement sur des valeurs définies.
- **UNESCO** : recommandation sur l'éthique de l'IA, adoptée par un grand nombre d'États, posant des principes universels (inclusion, diversité, protection de l'environnement).

| Juridiction | Type d'instrument | Contrainte | Orientation |
| :--- | :--- | :--- | :--- |
| Union européenne | Règlement | Contraignant | Approche par risque |
| États-Unis | Cadre volontaire | Non contraignant | Gestion de risque interne |
| Royaume-Uni | Orientations sectorielles | Souple | Régulateurs existants |
| Chine | Règlements ciblés | Contraignant | Contrôle des services et contenus |
| UNESCO | Recommandation | Non contraignant | Principes éthiques universels |

---

## 5. Droit d'auteur et propriété intellectuelle

### L'entraînement comme cas juridique
Les modèles de fondation sont entraînés sur des volumes massifs de textes, images et code aspirés du web. Deux thèses s'opposent :
- **Thèse de la violation** : la copie massive d'œuvres protégées pour constituer un corpus d'entraînement porte atteinte aux droits patrimoniaux des titulaires.
- **Thèse du fair use / usage transformateur** : le modèle n'archive pas les œuvres mais en extrait des régularités statistiques, ce qui relèverait d'un usage transformateur.

Ces thèses ne sont pas tranchées et les décisions varient selon les juridictions.

### Affaires en cours
- Des éditeurs de presse poursuivent des fournisseurs de modèles pour la restitution verbatim de contenus protégés par certaines requêtes.
- Des agences de presse visuelles contestent l'utilisation massive d'images protégées, avec des preuves de reproduction de filigranes.
- Ces affaires se concentrent sur l'argument de la mémorisation : si un modèle peut restituer un texte ou une image précise, l'argument transformateur est fragilisé sur ce point précis.

### Sorties, deepfakes et opt-out
- **Sorties et reproductions** : la génération de contenu très proche d'une œuvre existante peut constituer une contrefaçon, indépendamment de la question de l'entraînement.
- **Deepfakes** : la génération d'images ou voix imitant des personnes pose des questions de droit à l'image et de réputation, souvent distinctes du droit d'auteur.
- **Opt-out** : des mécanismes volontaires (standard de réservation de droits, `robots.txt`, politiques de crawler) permettent aux ayants droit de signaler leur refus de voir leurs œuvres ingérées. Leur efficacité dépend de l'adhésion des aspirateurs de données.

### Licences des jeux de données
| Type de source | Risque juridique typique |
| :--- | :--- |
| Contenu sous droit d'auteur | Reproduction sans licence, faute de base claire |
| Licences ouvertes permissives (MIT, CC-BY) | Attribution à respecter |
| Licences restrictives (CC-NC, GPL) | Usage commercial interdit ou contaminant |
| Données personnelles | Conformité RGPD, base légale requise |
| Code source | Respect des licences logicielles (copyleft, attribution) |

---

## 6. Vie privée, mémorisation et traçabilité

### Mémorisation et fuite de données
- **Mémorisation** : les modèles entraînés sur beaucoup d'exemples répétés peuvent restituer des fragments exacts du corpus. Le risque augmente avec la duplication et la taille des modèles.
- **PII (informations personnelles identifiantes)** : noms, adresses, numéros, identifiants. Une extraction ciblée peut les faire ressortir.
- **Attaques par inférence d'appartenance** : déterminer si une donnée précise figurait dans le jeu d'entraînement, à partir de la réponse du modèle. Cela peut révéler des informations sensibles (présence dans un jeu médical).

### Protection et atténuation
- **Déduplication** : réduire la duplication des données limite fortement la mémorisation.
- **Filtrage** : supprimer les PII directement identifiables du corpus.
- **Confidentialité différentielle (DP-SGD)** : tronquer les gradients et ajouter un bruit calibré pour garantir une borne de fuite. Coût : baisse de qualité et surcoût de calcul.
- **Évaluation par extraction** : red teamer le modèle pour mesurer la quantité de données extractibles.

### Deepfakes, watermarking et C2PA
- **Watermarking** : insertion d'un signal (dans les tokens pour le texte, dans les fréquences pour l'image) servant à identifier une provenance générée. Un watermark textuel peut être effacé par paraphrase.
- **C2PA** : standard de provenance qui attache des métadonnées signées aux médias, traçant l'origine et les modifications. C'est une signature de provenance, pas une garantie contre la création d'un faux hors de la chaîne de confiance.
- **Détection** : les classifieurs de contenu généré sont imparfaits et leur performance se dégrade dès que le générateur évolue ou que le contenu est retouché.

---

## 7. Travail, économie et emploi

### Automatisation de tâches plutôt que de métiers
L'effet dominant de l'IA est l'automatisation de tâches spécifiques, pas la suppression de métiers entiers. Un métier est un ensemble de tâches ; l'IA en remplace certaines et augmente la productivité sur d'autres.

```
[ Tâches routinières et standardisées ]  --> forte automatisation
[ Tâches cognitives et créatives ]       --> augmentation (productivité accrue)
[ Tâches physiques complexes ]           --> robotique encore limitée
[ Tâches relationnelles et éthiques ]    --> résilience humaine
```

### Métiers exposés et résilients
- **Fortement exposés** : traduction standard, rédaction de contenu basique, support de premier niveau, saisie, codage répétitif.
- **Partiellement exposés** : analyse de données, rédaction juridique ou médicale assistée.
- **Résilients** : métiers nécessitant interaction physique fine, jugement éthique, relation humaine de confiance.

### Augmentation contre remplacement
L'arbitrage central : l'IA peut être déployée pour **augmenter** un travailleur (assistant, suggestions, gain de temps) ou pour le **remplacer** (automatisation complète). Le choix est organisationnel et politique, pas purement technique. Les gains de productivité se répartissent différemment selon qu'ils reviennent aux salariés, aux clients ou aux détenteurs de capital.

Le risque de fracture : si les gains sont concentrés chez quelques plateformes, les inégalités peuvent se creuser malgré une productivité globale en hausse.

---

## 8. Environnement

### Énergie de l'entraînement
Un grand modèle pré-entraîné sur des semaines de calcul distribué consomme de l'ordre de centaines de mégawattheures. La quantité dépend de la taille du modèle, du nombre de tokens et du matériel.

### L'inférence domine sur le cycle de vie
C'est le point souvent mal compris : une requête unique coûte peu (de l'ordre de quelques watt-heures), mais des milliards de requêtes cumulées dépassent en général l'empreinte de l'entraînement initial. La bonne métrique est donc le coût cumulé par requête multiplié par le volume de trafic sur la durée de vie du modèle.

### Eau et minerais
- **Eau** : le refroidissement des centres de données évapore de grands volumes d'eau douce, avec des tensions locales dans les régions stressées hydriquement.
- **Minerais** : la fabrication des accélérateurs dépend de métaux (lithium, cobalt, terres rares) dont l'extraction a des impacts environnementaux et humains.

### Leviers de réduction
- Efficacité algorithmique : modèles plus petits, distillation, quantification.
- Efficacité d'inférence : batching, cache KV, decoding spéculatif.
- Choix du mix énergétique et localisation des centres de données.
- Sobriété d'usage : éviter les appels inutiles, cacher les résultats, dimensionner le modèle à la tâche.

---

## 9. Désinformation, chambres d'écho et sycophancie

### Chambres d'écho
Les systèmes de recommandation optimisés pour l'engagement tendent à renforcer les croyances préexistantes en réduisant l'exposition aux points de vue divergents. La polarisation s'auto-entretient.

### Sycophancie
> **Définition** : la sycophancie est la tendance d'un modèle entraîné par retour humain à confirmer ou flatter l'opinion de l'utilisateur plutôt qu'à corriger factuellement son erreur.

C'est un effet secondaire du RLHF : si les annotateurs préfèrent des réponses agréables, le modèle apprend à être agréable plutôt que juste. Cela dégrade la fiabilité et renforce les biais de confirmation.

### Manipulation à grande échelle
La génération automatique permet de produire des milliers de variantes d'un narratif, adaptées à des audiences ciblées, à faible coût. La détection est difficile car le contenu peut être grammaticalement irréprochable.

### Contre-mesures
- Éducation aux médias et à l'information.
- Transparence sur la provenance (C2PA), labellisation des contenus synthétiques.
- Conception de systèmes de recommandation qui réintroduisent de la diversité.
- Entraînement des modèles à résister à la complaisance plutôt qu'à flatter.

---

## 10. Concentration industrielle, modèles ouverts et gouvernance

### Concentration du calcul
L'entraînement des modèles frontière exige des moyens de calcul et de données hors de portée de la plupart des acteurs. Cela concentre le pouvoir entre quelques grandes entreprises et crée des dépendances critiques.

### Modèles ouverts contre modèles propriétaires
- **Modèles aux poids ouverts** : transparence, auditabilité, souveraineté, innovation décentralisée. Contrepartie : risque d'usages non encadrés, absence de garde-fous par défaut.
- **Modèles propriétaires** : contrôlés par API, plus faciles à filtrer, mais opaques et dépendants d'un fournisseur.

L'arbitrage n'a pas de réponse unique : selon les usages, l'ouverture est un atout (recherche, audit, souveraineté) ou un risque (diffusion de capacités dangereuses).

### Gouvernance et audit
```
[ Politique et rôles ] -> [ Documentation ] -> [ Évaluation ] -> [ Surveillance ] -> [ Recours ]
        |                       |                   |                   |                |
   Qui décide            Model cards         Biais, sécurité      Monitoring       Voie de contestation
```

### Ce qu'un développeur peut faire concrètement
1. Rédiger une **Model Card** : capacités, limites, contextes d'usage valides, métriques de biais, résultats d'évaluation.
2. Rédiger une **Data Sheet** : provenance des données, licences, méthodes de nettoyage, éventuelles exclusions.
3. Évaluer les biais avec des métriques explicites et documenter la notion d'équité choisie et pourquoi.
4. Mettre en place un **recours humain** pour toute personne affectée par une décision automatisée significative.
5. Journaliser les décisions et permettre l'audit.
6. Prévoir la réversibilité : pouvoir désactiver ou remplacer le modèle.

---

## Ce qu'il faut retenir
- Les biais algorithmiques proviennent des données, de l'objectif et du déploiement ; ils sont mesurables par des métriques d'équité souvent incompatibles entre elles.
- Le RGPD impose minimisation, droit à l'explication et DPIA, avec le droit à l'effacement comme point difficile (machine unlearning).
- L'EU AI Act régule par niveau de risque et impose des obligations renforcées aux modèles à usage général à risque systémique.
- D'autres cadres existent : NIST AI RMF aux États-Unis, approche sectorielle au Royaume-Uni, règlements ciblés en Chine, principes UNESCO.
- Le droit d'auteur sur l'entraînement n'est pas tranché ; mémorisation et reproduction de sorties sont les points sensibles.
- La vie privée est menacée par la mémorisation et l'inférence d'appartenance ; déduplication, filtrage et confidentialité différentielle atténuent partiellement.
- L'IA automatise des tâches plus qu'elle ne remplace des métiers ; le partage des gains de productivité est l'enjeu central.
- L'empreinte environnementale est dominée par l'inférence cumulée plus que par l'entraînement.
- La sycophancie et les chambres d'écho fragilisent la fiabilité et le débat public.
- La gouvernance responsable repose sur la documentation, l'audit, la surveillance et des mécanismes de recours.

## Erreurs fréquentes / idées reçues
- "Un modèle mathématique est neutre." -> Faux : il reproduit les biais de ses données et de son objectif.
- "Le RGPD interdit l'IA sur données personnelles." -> Faux : il l'encadre sous conditions et garanties.
- "L'AI Act bloque la recherche." -> Faux : il cible les usages à risque, avec des traitements spécifiques pour la recherche.
- "On peut satisfaire toutes les métriques d'équité." -> Faux : elles sont en général incompatibles ; il faut choisir et documenter.
- "Supprimer une donnée du jeu d'entraînement suffit à l'effacer du modèle." -> Faux : l'influence persiste sans machine unlearning.
- "L'empreinte de l'IA s'arrête à l'entraînement." -> Faux : l'inférence cumulée domine sur la durée de vie.
- "La sycophancie est une qualité." -> Faux : c'est une faille de fiabilité qui valide les erreurs.
- "Un watermark rend une détection fiable." -> Faux : il peut être retiré par transformation du contenu.

## Pour aller plus loin
- Liens internes : [chapitre 12 sur les données](../12-donnees/README.md), [chapitre 08 sur les agents](../08-agents/README.md).
- Ressource externe : texte officiel de l'EU AI Act, Commission européenne.
- Ressource externe : RGPD (règlement UE 2016/679), texte officiel.
- Ressource externe : NIST AI Risk Management Framework (AI RMF 1.0).
- Ressource externe : IBM AI Fairness 360 et Fairlearn pour l'évaluation d'équité.
- Ressource externe : recommandation de l'UNESCO sur l'éthique de l'IA.
