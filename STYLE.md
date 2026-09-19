# STYLE.md — Charte de rédaction de l'encyclopédie IA

Document de travail. Il fixe les règles communes à
tous les chapitres pour garantir un ensemble homogène.

## 1. Public visé

Un lecteur curieux et technique : étudiant en informatique, profil dev / cybersécurité,
qui veut **comprendre** l'IA (vocabulaire, mécanismes, ordres de grandeur, arbitrages)
sans forcément entraîner un modèle géant. Il code un peu (Python), il connaît Linux.
Il ne connaît PAS les maths du ML par coeur. On explique tout ce qu'on emploie.

## 2. Langue et ton

- **Français** integral, sauf les **termes techniques consacrés** qui restent en anglais
  quand c'est l'usage : `fine-tuning`, `embedding`, `chunking`, `prompt`, `throughput`,
  `attention`, `token`, `checkpoint`, `scaling laws`...
  Regle : on garde le mot anglais, mais **on le definit en francais a la premiere
  occurrence**.
- Tutoiement ("tu") ou "on" impersonnel. Vouvoiement interdit (c'est un guide perso).
- Ton direct, precis, sans hype. Pas de "revolutionnaire", "incroyable", "game changer".
  Si un point est incertain ou debattu, on le dit explicitement.
- Pas de langue de bois commerciale sur les fournisseurs. On nomme les acteurs
  (OpenAI, Anthropic, Google, Meta, Mistral, NVIDIA, Hugging Face...) factuellement.

## 3. Regles de format (STRICTES)

- **AUCUN emoji, aucune icone, aucun unicode decoratif.** Interdits : checkmarks, croix,
  fleches Unicode, traits de boite, puces exotiques, drapeaux, etc.
  Autorisé : ASCII pur (`->`, `|--`, `[ok]`, `[ko]`, `+/-`) et **les accents francais**
  (é è à ç ù ô î) ainsi que les symboles mathematiques courants (θ, ε, ∇, Σ, √, ≈, ≤, ≥, →).
  En cas de doute, ASCII.
- Markdown standard GitHub : titres `#`/`##`/`###`, **gras**, `code inline`,
  blocs ```clos```, tableaux, listes, citations `>`.
- **Pas de HTML brut**, pas de `[embed]`, pas de badges images.
- Les **tableaux** sont encourages (comparaisons, ordres de grandeur, glossaires locaux).
- Les **blocs de code** sont encourages : schemas ASCII, extraits Python, commandes shell.
- Les liens : chemins **relatifs**. Depuis `05-llm/README.md` vers RAG :
  `[chapitre 07](07-rag/README.md)`. Tous les liens doivent resoudre (jamais de lien mort).
- Longueur cible par chapitre : **300 a 700 lignes** (sois dense, pas de remplissage).
  Chaque paragraphe doit apprendre quelque chose.

## 4. Structure obligatoire d'un chapitre

Chaque fichier de chapitre s'ouvre et se ferme toujours de la meme facon :

```
# <Titre du chapitre>

> Resume en une ou deux phrases : de quoi ca parle et pourquoi ca compte.

## 1. <Premiere section>
...
## n. <Derniere section>

## Ce qu'il faut retenir
- 5 a 10 puces, chacune une phrase autoportante et verifiable.

## Erreurs frequentes / idees recues
- 3 a 8 puces du type "idee recue -> ce qui est vrai".

## Pour aller plus loin
- Liens internes vers les autres chapitres concernes (chemins relatifs).
- 2 a 5 ressources externes reelles (papiers, docs officielles, cours).
```

Regles de fond :

1. **Toujours definir un terme avant de l'utiliser.** Premier emploi -> definition courte,
   eventuellement en bloc `>`.
2. **Concret d'abord.** Une analogie ou un chiffre reel avant la formule. La formule
   vient apres, et elle est expliquee terme a terme.
3. **Ordres de grandeur systematiques** : tailles de modeles, nombre de tokens, VRAM,
   couts, latences, debits, energies. Toujours avec l'unite et le contexte.
4. **Arbitrages explicites** : ne jamais dire "il faut faire X" sans dire quand et
   pourquoi, et ce qu'on perd en echange.
5. **Pas de chiffre invente.** Si tu n'es pas sur d'une valeur precise, donne un
   ordre de grandeur et presente-le comme tel ("de l'ordre de", "typiquement").
   Ne fabrique jamais de citation, de nom de papier ou d'URL precise : en cas de doute,
   cite l'institution ou le titre generique plutot qu'une fausse reference.
6. **Nommer les modeles et versions** quand c'est utile (GPT-4o, Claude Sonnet,
   Llama 3.1, Mistral Large, Qwen 2.5, DeepSeek-V3...), mais sans supposer que le
   lecteur connait le paysage : le chapitre 05 et le chapitre 15 posent les bases.

## 5. Cartographie des chapitres (pour les liens croises)

| Dossier | Sujet | Chapitre |
|---|---|---|
| `00-introduction` | Qu'est-ce que l'IA, histoire, cartographie, mythes | 00 |
| `01-fondamentaux` | Donnees, ML, DL, types d'apprentissage, metriques, overfitting, optimisation | 01 |
| `02-mathematiques` | Algebre lineaire, calcul differentiel, probabilites, stats, information | 02 |
| `03-reseaux-de-neurones` | Perceptron, MLP, activations, retropropagation, CNN/RNN/GAN/VAE/diffusion | 03 |
| `04-transformers` | Attention, architecture, tokenisation, embeddings, positionnel, variantes | 04 |
| `05-llm` | Pre-entrainement, scaling laws, post-training, RLHF/DPO, raisonnement, evaluation | 05 |
| `06-fine-tuning` | Quand fine-tuner, LoRA/QLoRA/PEFT, distillation, quantization | 06 |
| `07-rag` | Pipeline RAG, chunking, embeddings, bases vectorielles, reranking, eval | 07 |
| `08-agents` | Agents, tool calling, MCP, memoire, planification, multi-agents | 08 |
| `09-inference-optimisation` | KV cache, vLLM, batching continu, speculative decoding, latence/cout | 09 |
| `10-infrastructure` | GPU vs CPU vs TPU, CUDA, VRAM, precision, parallelisme, energie, cloud | 10 |
| `11-multimodal` | Vision, audio/speech, video, generation d'image, VLM | 11 |
| `12-donnees` | Collecte, nettoyage, annotation, donnees synthetiques, droit | 12 |
| `13-securite` | Prompt injection, jailbreak, fuites, adversarial, guardrails, red teaming | 13 |
| `14-ethique-societe` | Biais, AI Act, emploi, environnement, alignement, droit d'auteur | 14 |
| `15-ecosysteme` | Frameworks, libs, Hugging Face, fournisseurs API, modeles ouverts | 15 |
| `16-pratique` | Tutoriels : premier modele, RAG, serving, QLoRA, agent | 16 |
| `17-au-dela` | Quantique, neuro-symbolique, modeles du monde, debat AGI, tendances | 17 |
| `GLOSSAIRE.md` | 300+ termes, A-Z | — |
| `FAQ.md` | Questions frequentes, reponses courtes | — |
| `CHEATSHEET.md` | Formules, commandes, ordres de grandeur, arbres de decision | — |
| `PROGRESSION.md` | Parcours d'apprentissage en semaines | — |

Tous les chapitres citent au moins 2 liens internes. Utilise cette table pour les
chemins exacts.

## 6. Checklist avant de rendre

- [ ] Mon fichier respecte la structure obligatoire (resume, sections numerotees,
      "Ce qu'il faut retenir", "Erreurs frequentes", "Pour aller plus loin").
- [ ] Zero emoji, zero unicode decoratif.
- [ ] Tous les liens internes pointent vers des fichiers qui existent vraiment
      (verifie le nom du dossier et le nom de fichier).
- [ ] Les termes techniques sont definis a la premiere occurrence.
- [ ] Il y a au moins un tableau et au moins un bloc de code/schema.
- [ ] Aucune reference inventee.
