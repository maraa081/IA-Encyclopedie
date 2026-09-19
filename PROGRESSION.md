# Parcours d'apprentissage

Ce parcours d'apprentissage est conçu pour vous guider de manière méthodique à travers l'encyclopédie, depuis les fondations mathématiques et conceptuelles jusqu'aux architectures avancées de LLM, d'agents autonomes et d'optimisation matérielle.

Le programme est structuré en **3 niveaux** répartis sur **15 semaines** (environ 10 à 15 heures d'effort par semaine).

---

## Niveau 1 : Fondations (Semaines 1 à 5)

L'objectif de ce premier niveau est de maîtriser les bases théoriques et pratiques du Machine Learning et des réseaux de neurones.

### Semaine 1 : Introduction, Histoire et Cartographie de l'IA
- **Objectifs** : Comprendre l'évolution historique de l'IA, distinguer les différentes sous-disciplines et identifier les mythes et réalités.
- **Chapitres à lire** : 
  - [[00-introduction/README.md](00-introduction/README.md)]
  - [[01-fondamentaux/README.md](01-fondamentaux/README.md)]
- **Exercices concrets** : Installer l'environnement Python et PyTorch, écrire un script simple de régression linéaire avec `scikit-learn`.
- **Projet / Livrable** : Un rapport de synthèse d'une page sur l'évolution des paradigmes de l'IA depuis les systèmes experts jusqu'aux LLM.
- **Critère d'auto-validation** : Capacité à expliquer la différence entre apprentissage supervisé et non supervisé avec des exemples concrets.

### Semaine 2 : Mathématiques pour l'IA : Algèbre linéaire
- **Objectifs** : Maîtriser les vecteurs, matrices, multiplications matricielles, valeurs propres et vecteurs propres indispensables au Deep Learning.
- **Chapitres à lire** : [[02-mathematiques/README.md](02-mathematiques/README.md)]
- **Exercices concrets** : Implémenter une multiplication matricielle en pur Python sans NumPy, puis comparer les performances avec NumPy.
- **Projet / Livrable** : Un notebook Jupyter démontrant la décomposition en valeurs singulières (SVD) sur une image en noir et blanc.
- **Critère d'auto-validation** : Comprendre géométriquement l'effet d'une transformation linéaire représentée par une matrice 2x2.

### Semaine 3 : Probabilités, statistiques et information
- **Objectifs** : Assimiler les lois de probabilité, le théorème de Bayes, l'entropie et la divergence KL.
- **Chapitres à lire** : [[02-mathematiques/README.md](02-mathematiques/README.md)]
- **Exercices concrets** : Calculer l'entropie croisée entre deux distributions de probabilité discrètes avec Python.
- **Projet / Livrable** : Un script simulant le théorème de Bayes pour un filtre anti-spam naïf bayésien.
- **Critère d'auto-validation** : Expliquer intuitivement pourquoi la divergence KL n'est pas symétrique.

### Semaine 4 : Réseaux de neurones : du Perceptron au MLP
- **Objectifs** : Comprendre le fonctionnement d'un neurone artificiel, les fonctions d'activation et le Perceptron Multicouche (MLP).
- **Chapitres à lire** : [[03-reseaux-de-neurones/README.md](03-reseaux-de-neurones/README.md)]
- **Exercices concrets** : Implémenter un MLP de zéro en PyTorch pour résoudre le problème logique XOR.
- **Projet / Livrable** : Un script d'entraînement d'un réseau de neurones sur le dataset MNIST de reconnaissance de chiffres manuscrits.
- **Critère d'auto-validation** : Obtenir plus de 95% d'exactitude sur le test MNIST et expliquer le rôle de la rétropropagation (backpropagation).

### Semaine 5 : Architectures classiques : CNN et RNN
- **Objectifs** : Découvrir les convolutions pour le traitement d'images (CNN) et les réseaux récurrents (RNN, LSTM) pour les séquences.
- **Chapitres à lire** : [[03-reseaux-de-neurones/README.md](03-reseaux-de-neurones/README.md)]
- **Exercices concrets** : Construire un CNN simple pour classifier des images de chats et de chiens.
- **Projet / Livrable** : Un modèle de classification d'images fonctionnel avec rapport de performance (précision/rappel).
- **Critère d'auto-validation** : Comprendre pourquoi les RNN souffrent du problème de disparition du gradient (vanishing gradient).

---

## Niveau 2 : Praticien (Semaines 6 à 10)

L'objectif de ce deuxième niveau est de maîtriser l'architecture Transformer, les grands modèles de langage, le fine-tuning et les pipelines RAG.

### Semaine 6 : L'architecture Transformer et le mécanisme d'attention
- **Objectifs** : Décortiquer le papier fondateur "Attention Is All You Need", comprendre l'auto-attention et l'attention multi-têtes.
- **Chapitres à lire** : [[04-transformers/README.md](04-transformers/README.md)]
- **Exercices concrets** : Coder le mécanisme d'attention à partir de zéro avec PyTorch en utilisant des opérations matricielles.
- **Projet / Livrable** : Un script visualisant les poids d'attention d'une phrase simple.
- **Critère d'auto-validation** : Expliquer la différence mathématique et conceptuelle entre attention croisée et auto-attention.

### Semaine 7 : Tokenisation et embeddings
- **Objectifs** : Étudier les algorithmes de tokenisation (BPE, WordPiece, SentencePiece) et la géométrie des espaces d'embeddings.
- **Chapitres à lire** : [[04-transformers/README.md](04-transformers/README.md)]
- **Exercices concrets** : Utiliser la bibliothèque `tokenizers` de Hugging Face pour entraîner un tokenizer sur un corpus personnalisé.
- **Projet / Livrable** : Un script mesurant la similarité cosinus entre différents mots et phrases à l'aide d'un modèle d'embedding.
- **Critère d'auto-validation** : Comprendre pourquoi le choix du tokenizer impacte directement la taille du contexte et la langue du modèle.

### Semaine 8 : Grands modèles de langage (LLM)
- **Objectifs** : Explorer le cycle de vie d'un LLM : pré-entraînement sur corpus massif, apprentissage par renforcement (RLHF, DPO) et alignement.
- **Chapitres à lire** : [[05-llm/README.md](05-llm/README.md)]
- **Exercices concrets** : Utiliser l'API Hugging Face Transformers pour générer du texte avec un modèle ouvert (ex: Llama 3 ou Mistral) en ajustant la température et le top-p.
- **Projet / Livrable** : Un comparateur de prompts évaluant les réponses de deux modèles différents sur un jeu de questions standardisées.
- **Critère d'auto-validation** : Savoir configurer les paramètres de génération (repetition penalty, temperature, top_k, top_p).

### Semaine 9 : Fine-tuning et adaptation efficace (PEFT/LoRA)
- **Objectifs** : Apprendre à adapter un modèle pré-entraîné à une tâche spécifique en minimisant l'empreinte mémoire grâce à LoRA et QLoRA.
- **Chapitres à lire** : [[06-fine-tuning/README.md](06-fine-tuning/README.md)]
- **Exercices concrets** : Réaliser un fine-tuning LoRA sur un petit modèle avec Hugging Face `peft` et `trl`.
- **Projet / Livrable** : Un modèle spécialisé dans la classification de sentiments ou la génération de FAQ sectorielles.
- **Critère d'auto-validation** : Expliquer comment LoRA réduit le nombre de paramètres entraînables par rapport au fine-tuning complet.

### Semaine 10 : Systèmes RAG et bases vectorielles
- **Objectifs** : Concevoir un pipeline de Retrieval-Augmented Generation complet : découpage (chunking), indexation vectorielle, recherche et génération.
- **Chapitres à lire** : [[07-rag/README.md](07-rag/README.md)]
- **Exercices concrets** : Construire un système RAG local avec FAISS ou Chroma et un modèle d'embedding open source.
- **Projet / Livrable** : Un assistant documentaire interactif capable de répondre à des questions sur un dossier de fichiers PDF.
- **Critère d'auto-validation** : Évaluer la pertinence du contexte récupéré et la fidélité de la réponse du LLM.

---

## Niveau 3 : Avancé (Semaines 11 à 15)

L'objectif de ce troisième niveau aborde les agents autonomes, l'optimisation des performances d'inférence, l'infrastructure matérielle et la sécurité.

### Semaine 11 : Agents autonomes, tool calling et MCP
- **Objectifs** : Implémenter des agents capables d'utiliser des outils externes, de planifier des tâches et de communiquer via le Model Context Protocol (MCP).
- **Chapitres à lire** : [[08-agents/README.md](08-agents/README.md)]
- **Exercices concrets** : Créer un agent ReAct capable d'interroger une API météo et une calculatrice.
- **Projet / Livrable** : Un assistant de recherche web autonome capable de synthétiser des articles en ligne.
- **Critère d'auto-validation** : Gérer les boucles d'erreurs d'un agent lorsque l'outil appelé renvoie un résultat inattendu.

### Semaine 12 : Optimisation de l'inférence et KV cache
- **Objectifs** : Comprendre la gestion de la mémoire KV cache, le batching continu et le décodage spéculatif.
- **Chapitres à lire** : [[09-inference-optimisation/README.md](09-inference-optimisation/README.md)]
- **Exercices concrets** : Déployer un modèle avec vLLM et mesurer le débit (throughput) et la latence.
- **Projet / Livrable** : Un banc de test comparant les performances d'inférence entre Hugging Face Transformers standard et vLLM.
- **Critère d'auto-validation** : Expliquer comment le PagedAttention de vLLM élimine la fragmentation de la mémoire.

### Semaine 13 : Infrastructure matérielle, VRAM et quantification
- **Objectifs** : Dimensionner une infrastructure GPU, calculer l'empreinte mémoire et appliquer des méthodes de quantification (GGUF, AWQ, GPTQ).
- **Chapitres à lire** : [[10-infrastructure/README.md](10-infrastructure/README.md)]
- **Exercices concrets** : Quantifier un modèle au format GGUF et le faire tourner localement avec llama.cpp.
- **Projet / Livrable** : Un rapport de dimensionnement matériel pour le déploiement d'un modèle de 70B en entreprise.
- **Critère d'auto-validation** : Calculer précisément la VRAM requise pour un modèle donné selon sa précision et la longueur du contexte.

### Semaine 14 : Multimodalité et données synthétiques
- **Objectifs** : Étudier les modèles multimodaux (vision, audio, texte) et les techniques de génération et de filtrage de données synthétiques.
- **Chapitres à lire** : 
  - [[11-multimodal/README.md](11-multimodal/README.md)]
  - [[12-donnees/README.md](12-donnees/README.md)]
- **Exercices concrets** : Utiliser un Vision Language Model (VLM) open source pour analyser et décrire des graphiques.
- **Projet / Livrable** : Un pipeline de génération de dataset synthétique validé par un LLM enseignant.
- **Critère d'auto-validation** : Identifier les risques de collapsus de modèle (model collapse) lors de l'entraînement sur des données synthétiques.

### Semaine 15 : Sécurité, éthique, alignement et gouvernance
- **Objectifs** : Maîtriser les techniques de sécurité (red teaming, protection contre les injections de prompts), l'alignement et la conformité réglementaire (AI Act).
- **Chapitres à lire** : 
  - [[13-securite/README.md](13-securite/README.md)]
  - [[14-ethique-societe/README.md](14-ethique-societe/README.md)]
  - [[15-ecosysteme/README.md](15-ecosysteme/README.md)]
- **Exercices concrets** : Mettre en place des barrières de sécurité (guardrails) sur un chatbot pour bloquer les contenus toxiques et les injections de prompts.
- **Projet / Livrable** : Un audit de sécurité (red teaming simulé) sur une application RAG.
- **Critère d'auto-validation** : Connaître les principales obligations de l'AI Act européen pour les systèmes d'IA à haut risque.

---

## Projets de portfolio progressifs

Pour valider vos compétences et constituer un portfolio attractif, réalisez 8 projets progressifs :

1. **Classificateur de texte simple (Niveau Débutant)** : Un script de classification de spams avec scikit-learn et régression logistique.
2. **Réseau de neurones from scratch (Niveau Débutant)** : Un MLP en pur NumPy pour reconnaître les chiffres MNIST.
3. **Générateur de texte de type Bigramme (Niveau Intermédiaire)** : Un modèle de langage simple basé sur des tables de transition de caractères en PyTorch.
4. **Assistant RAG local pour documents PDF (Niveau Intermédiaire)** : Une application Streamlit avec FAISS et un petit modèle open source pour interroger des contrats.
5. **Fine-tuning LoRA d'un petit LLM (Niveau Intermédiaire)** : Adaptation d'un modèle de 1B ou 3B sur un format de questions-réponses métier spécifique.
6. **Agent autonome multi-outils (Niveau Avancé)** : Un agent ReAct capable de récupérer des données financières en temps réel et de tracer des graphiques.
7. **Serveur d'inférence optimisé vLLM (Niveau Avancé)** : Déploiement d'un modèle avec quantification AWQ et test de charge concurrent.
8. **Pipeline d'évaluation et de Red Teaming (Niveau Avancé)** : Un banc d'audit automatisé pour tester la robustesse d'un LLM face aux injections de prompts.

---

## Ressources externes de référence

Pour aller plus loin, consultez ces ressources réelles et reconnues :
- **Cours et Tutoriels** : 
  - *CS231n : Convolutional Neural Networks for Visual Recognition* (Stanford)
  - *CS224n : Natural Language Processing with Deep Learning* (Stanford)
  - *DeepLearning.AI* (Cours de Andrew Ng sur les LLM et le RAG)
- **Livres fondateurs** :
  - *Deep Learning* par Ian Goodfellow, Yoshua Bengio et Aaron Courville.
  - *Speech and Language Processing* par Dan Jurafsky et James H. Martin.
- **Papiers fondateurs indispensables** :
  - *Attention Is All You Need* (Vaswani et al., 2017)
  - *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
  - *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
- **Blogs et actualités** :
  - *Hugging Face Blog* (tutoriels techniques de pointe)
  - *The Gradient* et *Distill.pub* (explications visuelles et articles de recherche)

---
*Lien connexe : Pour démarrer vos premières lignes de code pratique, consultez le chapitre dédié [[16-pratique/README.md](16-pratique/README.md)].*