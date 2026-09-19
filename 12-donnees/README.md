# Données : la matière première

> Comprendre l'importance cruciale de la qualité, de la diversité et du traitement des données dans la performance des systèmes d'intelligence artificielle modernes.

## 1. L'importance de la qualité des données

Dans le domaine de l'intelligence artificielle, l'adage "Garbage In, Garbage Out" (GIGO) est plus vrai que jamais.
La performance d'un modèle est directement limitée par la qualité des données utilisées lors de son entraînement.
Qu'il s'agisse d'un classifieur d'images ou d'un grand modèle de langage, les données sont le socle de toute compétence apprise.
Si les données sont biaisées, incomplètes ou de mauvaise qualité, le modèle reproduira et amplifiera ces défauts.

La qualité d'un jeu de données se mesure selon plusieurs critères fondamentaux :
*   **La pertinence** : Les données doivent être représentatives de la tâche réelle que le modèle devra accomplir.
*   **La diversité** : Un modèle entraîné sur un échantillon trop restreint ne saura pas généraliser à de nouveaux cas de figure.
*   **L'exactitude** : Les étiquettes (labels) ou le contenu textuel doivent être véridiques et exempts d'erreurs factuelles.
*   **La propreté** : Absence de bruit, de doublons, de données corrompues ou de contenu indésirable (spam, toxicité).

Historiquement, l'accent a souvent été mis sur la quantité brute de données ("Big Data").
Cependant, on observe aujourd'hui un basculement vers l'approche "Data-Centric AI".
Cette philosophie privilégie l'amélioration itérative du jeu de données plutôt que la complexification de l'architecture du modèle.
Des études récentes sur les modèles comme Llama 3 ou Mistral ont montré des résultats probants.
Un filtrage drastique et une sélection rigoureuse permettent d'obtenir de meilleures performances avec moins de tokens d'entraînement.

## 2. Collecte et sources des données

Pour entraîner des modèles massifs, les chercheurs utilisent des sources de données variées, souvent à l'échelle du pétaoctet.
L'aspiration du Web est la méthode dominante pour constituer les socles de connaissances.

Les principales sources pour les modèles de langage (LLM) incluent :
*   **Common Crawl** :
    C'est un dépôt massif de pages Web aspirées depuis 2008.
    Il constitue la base de la plupart des jeux de données ouverts comme C4 ou FineWeb.
*   **Wikipedia** :
    C'est une source de très haute qualité, structurée et multilingue.
    Elle couvre une vaste gamme de connaissances factuelles validées par une communauté.
*   **Dépôts de code (GitHub)** :
    Ces données sont essentielles pour apprendre aux modèles la logique de programmation.
    Elles aident également à développer des capacités de raisonnement structuré et de résolution de problèmes.
*   **Livres (Project Gutenberg, Bibliothèques numériques)** :
    Ils apportent la richesse du vocabulaire et la cohérence narrative.
    La profondeur des arguments y est souvent bien supérieure à celle trouvée sur les forums Web.
*   **Articles scientifiques (arXiv, PubMed)** :
    Ils fournissent la précision technique et le vocabulaire académique spécialisé.

Le tableau suivant présente une comparaison de quelques jeux de données célèbres utilisés pour l'entraînement :

| Jeu de données | Source principale | Taille (Tokens / Volume) | Qualité perçue | Usage principal |
|---|---|---|---|---|
| C4 (Colossal Clean Crawled Corpus) | Common Crawl filtré | ~150-800 milliards | Moyenne | Pré-entraînement LLM (T5, PaLM) |
| FineWeb-Edu | Web (Hugging Face) | ~1 300 milliards | Très Haute | Pré-entraînement focalisé éducation |
| The Stack | GitHub / Gitlab | ~3 To de code | Haute | Modèles de code (StarCoder) |
| Wikipedia (FR) | Encyclopédie en ligne | ~1-2 milliards | Très Haute | Connaissances factuelles, alignement |
| LAION-5B | Web (Images + Légendes) | 5,8 milliards d'images | Variable | Modèles de diffusion (Stable Diffusion) |
| Books3 | Livres numérisés | ~100 milliards | Très Haute | Capacité narrative et raisonnement long |

La collecte massive pose des défis techniques majeurs en termes de stockage et de bande passante.
La gestion des licences (Open Source, Creative Commons, Copyright) devient également un enjeu central pour les entreprises.

## 3. Nettoyage et filtrage des données

Le passage d'un crawl Web brut à un jeu de données d'entraînement propre nécessite un pipeline de traitement complexe.
Ce processus peut éliminer jusqu'à 90 % de l'information initiale pour ne garder que la "substantifique moelle".

Les étapes clés du nettoyage sont les suivantes :
1.  **Détection de la langue** :
    On utilise des classifieurs très rapides comme `fastText` de Meta.
    Cela permet de ne conserver que les langues ciblées et d'éliminer les pages "malades" (mélanges de langues).
2.  **Filtrage par heuristiques** :
    Suppression des pages ayant un ratio de ponctuation trop élevé ou trop de mots répétés.
    On élimine aussi les documents ayant trop peu de texte par rapport au code HTML (boilerplate).
3.  **Filtrage de contenu toxique** :
    Retrait du contenu haineux, sexuellement explicite ou violent.
    On s'appuie sur des listes de mots-clés interdits et des modèles de classification de toxicité.
4.  **Déduplication** :
    C'est une étape cruciale pour éviter que le modèle ne mémorise des passages répétés.
    Une mémorisation excessive dégrade la capacité de généralisation et favorise le plagiat involontaire.

La déduplication se fait souvent à deux niveaux d'abstraction :
*   **Déduplication exacte** :
    Comparaison de hachages (MD5 ou SHA-256) pour supprimer les fichiers identiques au bit près.
*   **Déduplication approximative (Fuzzy Deduplication)** :
    On utilise des algorithmes comme **MinHash** et **LSH** (Locality Sensitive Hashing).
    Ils permettent de détecter des documents presque identiques.
    Par exemple, une même dépêche Reuters publiée sur plusieurs sites avec des en-têtes différents sera détectée comme doublon.

Le processus MinHash suit généralement ces étapes :
*   **Shingling** : Découpage du texte en petits morceaux de $k$ mots consécutifs.
*   **MinHashing** : Création d'une signature numérique compacte pour chaque document.
*   **LSH** : Regroupement des documents ayant des signatures similaires dans des "seaux" (buckets) pour limiter les comparaisons.

## 4. Annotation et curation des données

Une fois le pré-entraînement terminé sur des données brutes, le modèle doit être affiné.
L'objectif est qu'il apprenne à suivre des instructions et à se comporter de manière utile et sûre.
C'est ici qu'intervient l'annotation humaine de haute précision.

L'annotation peut prendre plusieurs formes complémentaires :
*   **SFT (Supervised Fine-Tuning)** :
    Des humains rédigent des paires de questions et réponses idéales.
    Le modèle apprend par imitation de ces exemples "en or".
*   **RLHF (Reinforcement Learning from Human Feedback)** :
    Des humains classent plusieurs réponses générées par le modèle de la meilleure à la moins bonne.
    Ce feedback permet d'entraîner un "modèle de récompense" (Reward Model).
    Le modèle principal est ensuite optimisé pour maximiser cette récompense via des algorithmes comme PPO ou DPO.
*   **Correction de biais et Red Teaming** :
    Identification active des réponses contenant des stéréotypes ou des instructions dangereuses.

Le coût de l'annotation humaine est extrêmement élevé.
Il peut atteindre plusieurs dizaines de dollars par exemple complexe nécessitant une expertise (maths, code, droit).
Pour réduire ces coûts, on utilise désormais l'annotation assistée par IA (RLAIF).
Ici, un modèle très puissant (comme GPT-4o) aide à critiquer ou à classer les sorties de modèles plus petits.

## 5. Données synthétiques et effondrement

Les données synthétiques sont produites par un autre modèle d'IA au lieu d'être collectées dans le monde réel.

Elles présentent des avantages indéniables :
*   **Coût marginal quasi nul** : Générer des millions de lignes de données ne coûte que du temps de calcul.
*   **Passage à l'échelle (Scalability)** : On peut créer des données sur des sujets très pointus ou rares.
*   **Fiabilité logique** : On peut forcer la génération de raisonnements mathématiques étape par étape (Chain of Thought).

Cependant, un danger majeur guette cette approche : l'**effondrement du modèle** (Model Collapse).
Si un modèle est entraîné majoritairement sur des données synthétiques issues de ses propres versions précédentes :
1.  Les erreurs de compréhension s'accumulent à chaque génération.
2.  La diversité sémantique s'appauvrit progressivement.
3.  Le modèle finit par oublier les cas rares ("long tail") de la distribution réelle.

L'arbitrage actuel consiste à utiliser un mélange équilibré.
Les données réelles servent de socle de connaissances et de diversité.
Les données synthétiques servent à améliorer le raisonnement logique et le respect des formats de sortie.

## 6. Aspects juridiques et propriété intellectuelle

L'utilisation de données Web massives soulève des questions juridiques mondiales sans précédent.

Les points de tension principaux sont les suivants :
*   **Le Droit d'auteur (Copyright)** :
    Les auteurs d'œuvres contestent l'aspiration de leur travail sans consentement ni rémunération.
    Des procès majeurs opposent des médias (New York Times) aux créateurs de modèles (OpenAI).
*   **Le Fair Use (Usage Loyal)** :
    Aux États-Unis, les entreprises d'IA soutiennent que l'entraînement est une opération "transformative".
    Selon elles, cela ne constitue pas une violation directe du copyright.
*   **Le RGPD et la Vie Privée** :
    En Europe, l'IA ne doit pas mémoriser d'informations personnelles identifiables (PII).
    Le droit à l'effacement est techniquement difficile à appliquer une fois le modèle entraîné.
    On ne peut pas "extraire" facilement une donnée spécifique des milliards de poids du réseau.
*   **Les mécanismes d'Opt-out** :
    Le standard `robots.txt` a été étendu pour permettre aux sites de bloquer les agents de collecte (ex: GPTBot).
    De plus en plus de sites ferment leurs accès API ou facturent l'accès aux données.

## 7. Cas pratique : Pipeline de nettoyage de texte

Voici un exemple en Python montrant comment implémenter un pipeline simple de nettoyage et de filtrage.
Ce script illustre les étapes de base pour préparer un petit dataset avant l'entraînement.

```python
import re
import unicodedata

def clean_text(text):
    """
    Pipeline de base pour nettoyer du texte brut extrait du Web.
    """
    # 1. Normalisation Unicode (NFKC) pour gérer les variantes de caractères
    # Cela permet d'unifier les accents et les symboles spéciaux.
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Suppression des balises HTML résiduelles via une expression régulière
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Normalisation des espaces (suppression des tabulations, retours chariot)
    # On remplace toutes les séquences d'espaces par un espace unique.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_quality_content(text):
    """
    Heuristiques simples pour filtrer le bruit et le contenu non informatif.
    """
    # Filtre 1 : Longueur minimale (éliminer les fragments de phrases)
    if len(text) < 100:
        return False, "Trop court"
        
    # Filtre 2 : Ratio de mots/caractères (éliminer le code ou les listes)
    words = text.split()
    if len(words) == 0:
        return False, "Vide"
    
    avg_word_length = len(text) / len(words)
    # Un mot moyen en français fait entre 4 et 12 caractères.
    if avg_word_length < 3 or avg_word_length > 15:
        return False, "Structure non naturelle (code ou spam)"
        
    # Filtre 3 : Présence de ponctuation finale
    # On s'assure que le texte se termine par un signe de ponctuation fort.
    if not re.search(r'[.!?]$', text.strip()):
        return False, "Phrase incomplète ou fragment"
        
    return True, "Ok"

# Exemple d'utilisation sur un échantillon de données brutes
raw_data = [
    "Bonjour ! Ceci est un exemple de texte de qualité. Il contient plusieurs phrases complètes.",
    "Click here now!!! buy cheap stuff <html> <body> 12345 </body> </html>",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "Ce texte est beaucoup trop court pour être utile."
]

print("Analyse du pipeline de nettoyage :\n")
for doc in raw_data:
    cleaned = clean_text(doc)
    is_ok, reason = is_quality_content(cleaned)
    status = "[ok]" if is_ok else f"[ko] ({reason})"
    print(f"{status} -> {cleaned[:60]}...")
```

Ce script montre que le nettoyage est une étape de décision binaire : on garde ou on jette.
L'application de ces filtres garantit la propreté du corpus d'entraînement final.
Chaque document est ainsi validé avant d'être tokenisé.

## Ce qu'il faut retenir

- La qualité des données est le facteur limitant numéro 1 de la performance finale d'un modèle d'IA.
- Le filtrage des données (déduplication, détection de langue, retrait de toxicité) est indispensable.
- Il permet de réduire le volume de stockage tout en augmentant la précision du modèle.
- La déduplication approximative via MinHash/LSH permet d'améliorer la capacité de généralisation.
- L'aspiration de données Web (Web scraping) est la source principale de connaissances pour les LLM.
- L'annotation humaine (SFT/RLHF) est nécessaire pour aligner le modèle sur les besoins humains.
- Les données synthétiques offrent un passage à l'échelle rapide mais comportent un risque d'effondrement.
- La régulation européenne (AI Act et RGPD) impose des contraintes de transparence et de protection.
- Une approche Data-Centric privilégie l'amélioration constante du dataset sur celle de l'architecture.
- Le standard robots.txt permet désormais aux propriétaires de sites de s'opposer à la collecte.

## Erreurs fréquentes / idées reçues

- Plus on a de données, mieux c'est -> Faux. L'ajout de données bruitées peut ruiner un bon modèle.
- Les données synthétiques vont remplacer totalement les humains -> Idée reçue. Les données humaines sont la "vérité terrain" indispensable pour éviter la dérive sémantique.
- Une fois la donnée apprise, on peut la supprimer facilement -> Faux. Les informations sont diluées dans des milliards de paramètres. Le "unlearning" est un sujet complexe.
- Le Web est une source de vérité absolue -> Faux. Le Web est saturé de biais, de désinformation et de contenus générés par IA. Le filtrage est donc de plus en plus difficile.
- Le nettoyage des données est une tâche triviale -> Faux. C'est l'étape la plus chronophage et coûteuse en expertise dans tout projet d'intelligence artificielle sérieux.

## Pour aller plus loin

- [chapitre 01](../01-fondamentaux/README.md) : Pour comprendre l'usage des données dans l'apprentissage supervisé et non supervisé.
- [chapitre 14](../14-ethique-societe/README.md) : Pour approfondir les enjeux de biais, de vie privée et la régulation européenne.
- [chapitre 15](../15-ecosysteme/README.md) : Pour découvrir les outils comme Hugging Face `datasets` qui automatisent ces pipelines.
- "Data-centric AI" (Andrew Ng) : Le mouvement prônant la priorité systématique à la qualité des données.
- "Textbooks Are All You Need" (Gunasekar et al., 2023) : Étude montrant l'efficacité d'un entraînement sur des données synthétiques de très haute qualité (modèle phi-1).
- Common Crawl (commoncrawl.org) : Accès aux données de crawl Web en accès libre pour la recherche.
- Hugging Face FineWeb : Documentation détaillée sur la constitution d'un dataset de 15 billions de tokens.
