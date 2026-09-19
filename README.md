# Encyclopédie de l'Intelligence Artificielle

> Une référence complète, en français, pour comprendre l'IA de A à Z : le vocabulaire,
> les mécanismes, l'infrastructure, les usages, les risques et les débats.
> Aucun prérequis en mathématiques avancées : chaque notion est introduite par une
> intuition concrète avant la formule.

Dernière mise à jour : 2026-09-19.

---

## À qui s'adresse ce dépôt

- À un étudiant en informatique ou en cybersécurité qui veut **comprendre** l'IA
  plutôt que d'en consommer les buzzwords.
- À un développeur qui veut savoir **quoi faire** concrètement : RAG ou fine-tuning,
  quel matériel, quel coût, quel risque.
- À toute personne qui veut un **glossaire fiable** pour décoder les articles,
  les docs d'API et les annonces des fournisseurs.

Ce dépôt **n'entraîne pas** de modèle (pour ça, voir un projet de type « LLM from
scratch »). Il explique **comment ça marche, pourquoi, et avec quels ordres de grandeur**.

## Comment lire ce dépôt

Trois façons d'entrer :

1. **Lecture linéaire** : suivre les chapitres 00 -> 17 dans l'ordre. Chaque chapitre
   se termine par « Ce qu'il faut retenir », « Erreurs fréquentes » et des liens vers
   les chapitres voisins.
2. **Besoin ponctuel** : ouvrir [GLOSSAIRE.md](GLOSSAIRE.md) et partir du terme qui
   te bloque. Chaque entrée renvoie au chapitre qui développe le sujet.
3. **Objectif pratique** : aller directement au [chapitre 16](16-pratique/README.md)
   (tutoriels) avec le [CHEATSHEET.md](CHEATSHEET.md) à côté, et revenir à la théorie
   quand un blocage apparaît.

Conventions de rédaction : mots techniques conservés en anglais quand c'est l'usage
(`embedding`, `token`, `fine-tuning`) mais toujours définis à la première occurrence.
Les ordres de grandeur sont donnés en unités explicites (VRAM en Go, latence en ms,
débit en tokens/s, coût en euros ou dollars par million de tokens).

## Sommaire

| # | Chapitre | Ce que tu y trouves |
|---|---|---|
| 00 | [Introduction à l'IA](00-introduction/README.md) | Définitions, histoire, cartographie, mythes |
| 01 | [Fondamentaux du machine learning](01-fondamentaux/README.md) | Données, apprentissage, métriques, sur-apprentissage, optimisation |
| 02 | [Les maths de l'IA, sans douleur](02-mathematiques/README.md) | Algèbre linéaire, gradient, probabilités, entropie croisée, KL |
| 03 | [Réseaux de neurones](03-reseaux-de-neurones/README.md) | Perceptron, MLP, activations, rétropropagation, CNN/RNN/GAN/diffusion |
| 04 | [Transformers](04-transformers/README.md) | Attention, tokenisation, embeddings, positionnel, variantes modernes |
| 05 | [Grands modèles de langage (LLM)](05-llm/README.md) | Pré-entraînement, scaling laws, RLHF/DPO, raisonnement, évaluation |
| 06 | [Fine-tuning et adaptation](06-fine-tuning/README.md) | LoRA, QLoRA, PEFT, distillation, datasets, quand fine-tuner |
| 07 | [RAG : Retrieval-Augmented Generation](07-rag/README.md) | Pipeline, chunking, embeddings, bases vectorielles, reranking, éval |
| 08 | [Agents IA](08-agents/README.md) | Tool calling, MCP, mémoire, planification, multi-agents, fiabilité |
| 09 | [Inférence : optimiser et servir](09-inference-optimisation/README.md) | Prefill/decode, KV cache, vLLM, quantization, speculative decoding |
| 10 | [Infrastructure : GPU, VRAM, cloud](10-infrastructure/README.md) | Pourquoi des GPU, CUDA, précision numérique, parallélisme, énergie, coûts |
| 11 | [Multimodalité](11-multimodal/README.md) | Vision, audio, vidéo, diffusion d'image, VLM |
| 12 | [Données : la matière première](12-donnees/README.md) | Collecte, nettoyage, annotation, données synthétiques, droit |
| 13 | [Sécurité des systèmes d'IA](13-securite/README.md) | Prompt injection, jailbreak, poisoning, supply chain, guardrails |
| 14 | [Éthique, droit et société](14-ethique-societe/README.md) | Biais, RGPD, AI Act, droit d'auteur, travail, environnement |
| 15 | [Écosystème, outils et acteurs](15-ecosysteme/README.md) | Frameworks, libs, Hugging Face, fournisseurs, modèles ouverts |
| 16 | [Mise en pratique : tutoriels](16-pratique/README.md) | 8 tutoriels concrets, du premier réseau au RAG et à l'agent |
| 17 | [Frontières et débats](17-au-dela/README.md) | Quantique, world models, AGI, alignement, tendances |

Annexes :

- [GLOSSAIRE.md](GLOSSAIRE.md) — plus de 300 termes définis, A à Z, plus les acronymes.
- [FAQ.md](FAQ.md) — les questions qui reviennent, réponses courtes.
- [CHEATSHEET.md](CHEATSHEET.md) — formules, commandes, ordres de grandeur, arbres de décision.
- [PROGRESSION.md](PROGRESSION.md) — un parcours d'apprentissage structuré en 3 niveaux.

## Par où commencer selon ton objectif

| Ton objectif | Parcours conseillé |
|---|---|
| Comprendre les mots qu'on lit partout | 00 -> GLOSSAIRE -> 05 |
| Construire un chatbot sur mes documents | 07 -> 16(c, g) -> 09 |
| Entraîner ou adapter un modèle | 01 -> 03 -> 04 -> 06 -> 16(e) |
| Faire tourner un modèle chez moi | 10 -> 09 -> 16(d) |
| Sécuriser un système qui utilise un LLM | 13 -> 08 -> 12 |
| Comprendre les enjeux et la régulation | 14 -> 17 |
| Réviser pour un entretien technique IA | CHEATSHEET -> 04 -> 05 -> 09 |

## Repères d'ordre de grandeur

Quelques chiffres utiles pour calibrer l'intuition (détails et contexte dans les
chapitres concernés) :

| Élément | Ordre de grandeur |
|---|---|
| Un token | environ 0,75 mot en anglais, souvent moins en français |
| Taille d'un modèle courant | 7 à 70 milliards de paramètres pour l'usage local, 400+ pour les frontières |
| Mémoire des poids en FP16 | environ 2 Go par milliard de paramètres |
| Mémoire des poids en 4 bits | environ 0,5 Go par milliard de paramètres |
| VRAM d'une carte grand public | 8 à 24 Go |
| VRAM d'un accélérateur de datacenter | 40 à 192 Go |
| Latence perçue acceptable en chat | premier token sous 1 s, puis 20+ tokens/s |
| Coût d'un million de tokens via API | de quelques centimes à quelques dizaines de dollars selon le modèle |
| Contexte d'un modèle moderne | de 8 000 à plus d'un million de tokens |

## État de l'art et fraîcheur

L'IA évolue vite : ce dépôt privilégie les **mécanismes** (stables) sur les
**noms de produits** (périssables). Les chiffres et les modèles cités sont datés
(septembre 2026) et servent d'illustration, pas de vérité absolue. Si un chiffre
te semble décisif, vérifie-le à la source : les fiches modèles des fournisseurs et
les papiers cités en fin de chapitre.

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md). En résumé : une correction = une pull
request claire, avec la source. Pas d'emoji dans les fichiers, pas d'affirmation
sans source pour les chiffres.

## Licence

Contenu publié sous licence MIT, voir [LICENSE](LICENSE). Tu peux le réutiliser,
le modifier et le redistribuer en citant la source.
