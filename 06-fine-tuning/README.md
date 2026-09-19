# Fine-tuning et adaptation

> Ce chapitre explique quand et comment adapter un modèle pré-entraîné à un usage
> précis : le fine-tuning complet, les méthodes paramétriques efficaces (PEFT, LoRA,
> QLoRA), la distillation et la préparation des jeux de données. C'est le guide
> pratique pour transformer un modèle générique en modèle spécialisé sans casser la
> banque.

## 1. Faut-il vraiment fine-tuner ? L'arbre de décision

Avant d'ouvrir un GPU, il faut identifier le vrai problème. Le fine-tuning est souvent
la mauvaise réponse à un problème de contexte.

```
Question : le modèle échoue sur ma tâche. Pourquoi ?

A. Il ne connait pas mes informations (documents prives, actualite) ?
   -> RAG (chapitre 07). Ne pas fine-tuner pour injecter des faits.

B. Il ne respecte pas le format / le style / la tache precise ?
   -> D'abord un meilleur prompt (few-shot, sortie structuree).
   -> Si insuffisant : fine-tuning d'instruction (LoRA, section 5).

C. Il connait la tache mais se trompe souvent sur un domaine technique pointu ?
   -> Fine-tuning sur des donnees de domaine (paires verifiees).

D. Le cout ou la latence par requete est trop eleve ?
   -> Distillation (section 8) ou modele plus petit specialise.

E. Le modele frontier marche deja bien ?
   -> Ne rien faire. Complexite ajoutee = maintenance, regressions, dette.

F. Besoin de confidentialite stricte (donnees qui ne peuvent pas sortir) ?
   -> Modele ouvert + LoRA en local (chapitres 10 et 15).
```

Ordre de coût croissant : prompt engineering < RAG < LoRA/QLoRA < full fine-tuning <
pré-entraînement. Chaque étape demande plus de données, plus de calcul et plus de
rigueur d'évaluation.

| Approche | Modifie les poids | Coût typique | Convient pour |
|---|---|---|---|
| Prompt engineering | Non | quasi nul | Format, ton, tâches simples |
| RAG | Non | faible à moyen | Connaissances factuelles, fraîcheur |
| LoRA/PEFT | Oui (partiel) | faible à moyen | Style, format, domaine étroit |
| Full fine-tuning | Oui (total) | élevé | Nouveau comportement profond, gros volume |
| Pré-entraînement | Oui (total, from scratch) | très élevé | Nouvelle langue, nouveau domaine massif |

## 2. Full fine-tuning : la version lourde

> Full fine-tuning (fine-tuning complet) : on ré-entraîne tous les poids du modèle sur
> le nouveau jeu de données.

Pour un modèle de 7B en précision 16-bit, il faut déjà environ 14 Go rien que pour les
poids, plus les gradients (14 Go), plus l'état de l'optimiseur Adam (deux moments par
paramètre, soit environ 56 Go en 32-bit), plus les activations. On dépasse facilement
les 100 Go de VRAM. Un GPU 80 Go ne suffit pas toujours : il faut du parallélisme
(sharding, offload), ce qui complique l'infrastructure
([chapitre 10](../10-infrastructure/README.md)).

### L'oubli catastrophique

> Oubli catastrophique (catastrophic forgetting) : quand un modèle ré-entraîné sur une
> tâche étroite perd ses compétences générales.

Mécanisme : les poids qui portaient la connaissance générale sont écrasés par la
nouvelle distribution de données. Un modèle fine-tuné sans précaution sur un corpus
médical peut devenir incapable de tenir une conversation banale ou de coder.

Remèdes :

- Mélanger des données générales avec les données cibles (replay).
- Baisser le learning rate.
- Utiliser PEFT plutôt que full fine-tuning : on gèle la majorité des poids, donc on
  protège la connaissance générale.
- Limiter le nombre d'époques.

## 3. Le paradigme PEFT

> PEFT (Parameter-Efficient Fine-Tuning, fine-tuning efficace en paramètres) : famille
> de méthodes qui n'entraînent qu'une petite fraction des poids, en gelant le reste.

Avantages : VRAM réduite, checkpoints légers (quelques dizaines de Mo au lieu de
plusieurs Go), risque d'oubli catastrophique réduit, plusieurs adaptateurs
interchangeables pour un même modèle de base.

| Méthode | Idée | Paramètres entraînés |
|---|---|---|
| Adapters | Insertion de petites couches MLP entre les blocs | 1 à 5 pour cent |
| Prompt tuning | Vecteurs de prompt appris en entrée | < 1 pour cent |
| Prefix tuning | Préfixes appris dans chaque couche d'attention | < 1 pour cent |
| LoRA | Décomposition de rang faible des mises à jour | < 1 pour cent |
| IA3 | Recalibrage de vecteurs clés/valeurs | très faible |

## 4. LoRA : décomposition de rang faible

> LoRA (Low-Rank Adaptation) : au lieu d'apprendre la matrice complète de mise à jour
> d'une couche, on l'apprend comme le produit de deux petites matrices de rang r.

Soit une couche linéaire de poids W (dimensions d x k). Une mise à jour complète serait
Delta_W, de même taille que W. LoRA pose :

```
Delta_W = B x A

A : matrice r x k   (initialisee aleatoirement, souvent gaussienne)
B : matrice d x r   (initialisee a zero)
r : rang, typiquement 8 a 64

Le poids effectif devient W + alpha/r * (B x A).
alpha : facteur d'echelle du LoRA.
```

Comme B commence à zéro, la mise à jour est nulle au départ : le modèle démarre
identique à sa version pré-entraînée. On n'entraîne que A et B, ce qui réduit le nombre
de paramètres d'un facteur considérable.

Pour r=16 sur un modèle 7B, on entraîne typiquement quelques dizaines de millions de
paramètres au lieu de 7 milliards, soit moins de 1 pour cent.

### Rang, alpha et modules cibles

- **r (rang)** : capacité de l'adaptation. Petit (4-8) suffit pour le style ou le
  format. Plus grand (32-64) pour un domaine complexe. Au-delà, gain marginal et
  sur-apprentissage.
- **alpha** : facteur d'échelle appliqué à B x A. Une convention courante est de fixer
  alpha = 2r. Le facteur effectif est alpha/r.
- **modules cibles** : les couches q_proj et v_proj (projections query et value de
  l'attention) sont le minimum historique. Ajouter k_proj, o_proj et les couches
  feed-forward (up_proj, down_proj, gate_proj) améliore souvent la qualité au prix
  d'un peu plus de paramètres.

## 5. QLoRA, DoRA et autres variantes

### QLoRA

> QLoRA : LoRA appliqué au-dessus d'un modèle de base quantifié en 4 bits (NF4), avec
> déquantification à la volée pour les calculs.

Techniques internes :

- **NF4 (4-bit NormalFloat)** : format de quantification optimisé pour des poids
  suivant une distribution approximativement normale.
- **Double quantization** : on quantifie aussi les constantes de quantification, ce qui
  économise de la mémoire supplémentaire.
- **Paged optimizers** : l'état de l'optimiseur peut être déchargé vers la RAM CPU
  pour éviter le pic hors mémoire (out-of-memory).

Résultat : on peut fine-tuner un modèle de 7B sur un GPU grand public de 12-16 Go de
VRAM. Le coût : un peu plus de temps de calcul (déquantification) et une qualité
légèrement inférieure à un LoRA sur modèle 16-bit, souvent négligeable en pratique.

### Autres méthodes

- **DoRA (Weight-Decomposed Low-Rank Adaptation)** : sépare magnitude et direction de
  la mise à jour, se rapprochant du full fine-tuning à coût comparable à LoRA.
- **Adapters** : petites couches insérées dans le réseau (Houlsby et al.). Simples mais
  latence légèrement accrue à l'inférence.
- **Prompt / prefix tuning** : on n'entraîne que des vecteurs de contexte. Très léger,
  mais moins expressif et parfois fragile.
- **IA3** : apprend trois vecteurs de recalibrage (key, value, feed-forward), encore
  plus économe que LoRA.

## 6. Hyperparamètres LoRA en pratique

Valeurs typiques observées dans la communauté (à adapter, ce ne sont pas des lois) :

| Hyperparamètre | Valeur courante | Plage raisonnable | Commentaire |
|---|---|---|---|
| Rang r | 16 | 8 à 64 | 8 pour le style, 32+ pour un domaine |
| Alpha | 32 | r à 2r | Souvent fixé à 2r |
| Dropout LoRA | 0.05 | 0 à 0.1 | Réduit le sur-apprentissage |
| Learning rate | 2e-4 | 1e-4 à 3e-4 | Bien plus élevé qu'en full fine-tuning |
| Scheduler | cosine | linear, cosine | Cosine couramment utilisé |
| Warmup | 3 pour cent | 0 à 10 pour cent | Stabilise le début |
| Batch effectif | 16 à 128 | selon VRAM | Via accumulation de gradients |
| Époques | 2 à 3 | 1 à 5 | Au-delà, risque de surapprentissage |
| Modules cibles | q,k,v,o + FFN | attention seule ou tout | Plus de modules = plus de capacité |

```python
# Exemple conceptuel avec PEFT (pseudo-code, adapter a votre version)
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# attendu : quelques dizaines de millions sur plusieurs milliards
```

Signal d'alarme : si la loss d'entraînement descend très bas (< 0.3) sur un petit jeu
de données, le modèle récite ses exemples au lieu de généraliser.

## 7. Jeux de données de fine-tuning

La qualité des données domine tout le reste. 1 000 exemples propres et cohérents
valent souvent mieux que 100 000 exemples bruités.

### Formats courants

- **Alpaca** : JSON avec les champs instruction, input, output. Simple, historique.
- **ShareGPT** : conversations multi-tours au format role/content (system, user,
  assistant). Adapté au dialogue.
- **Chat template** : chaque modèle a son gabarit (Llama, Mistral, ChatML, Qwen). Il
  faut utiliser le template exact du modèle, sinon les performances s'effondrent.

```
Exemple Alpaca (JSON) :
{
  "instruction": "Classe ce mail comme spam ou non.",
  "input": "Gagnez 1 million en 3 jours !!!",
  "output": "spam"
}
```

### Règles de qualité

1. **Cohérence** : même format, même ton, même langue sur tout le jeu.
2. **Vérité** : les sorties doivent être correctes. Propager une erreur en la répétant
   l'ancre dans le modèle.
3. **Diversité** : couvrir les cas limites, pas seulement les cas faciles.
4. **Taille** : pour du style ou du format, quelques centaines à quelques milliers
   d'exemples suffisent. Pour un nouveau comportement, dizaines de milliers.
5. **Déduplication** : retirer les quasi-doublons qui favorisent la mémorisation.

### Données synthétiques et distillation de modèles frontières

Générer des exemples avec un modèle fort (GPT-4o, Claude) pour entraîner un modèle
plus petit est courant. Attention :

- Vérifier les licences d'utilisation des sorties (les conditions d'usage des API
  encadrent souvent l'entraînement de modèles concurrents).
- Filtrer et corriger : les sorties générées contiennent des erreurs.
- Risque de "mode collapse" si toutes les données viennent du même modèle.

### Licence et droit

Les jeux de données ont chacun leur licence (Apache, MIT, CC-BY, CC-BY-NC,
propriétaire). Un usage commercial du modèle fine-tuné peut être interdit selon la
licence des données ou du modèle de base. C'est un point juridique réel, à vérifier
avant tout déploiement ([chapitre 12](../12-donnees/README.md) et
[chapitre 14](../14-ethique-societe/README.md)).

## 8. Distillation

> Distillation : entraîner un petit modèle (student, élève) à imiter un grand modèle
> (teacher, enseignant).

Deux variantes :

- **Distillation de logits** : le student apprend à reproduire la distribution complète
  du teacher (soft labels), pas seulement la bonne réponse. Les soft labels portent une
  information riche (les relations entre classes).
- **Distillation dure** : le student s'entraîne simplement sur les sorties textuelles du
  teacher (hard labels, données synthétiques).

```
Teacher (70B) : distribution P_t sur le vocabulaire
Student (7B)  : distribution P_s

Perte = alpha * CE(y, P_s) + (1 - alpha) * KL(P_t || P_s)
CE : cross-entropy sur les vraies etiquettes
KL : divergence entre distributions teacher et student
alpha : ponderation (souvent 0.5)
```

**Distillation de raisonnement** : le teacher produit des chaînes de pensée détaillées,
le student apprend à les reproduire. C'est la recette qui a permis à des petits modèles
de raisonner correctement sur des tâches mathématiques.

Compromis : un student distillé est plus rapide et moins cher, mais plafonne sous son
teacher et hérite de ses biais.

## 9. Quantization : entraînement vs inférence

> Quantization (quantification) : réduire la précision numérique des poids (16-bit vers
> 8-bit, 4-bit) pour économiser de la mémoire.

- **Côté entraînement** : QLoRA quantifie le modèle gelé en 4-bit pour tenir en VRAM,
  pendant que les adaptateurs restent en précision supérieure.
- **Côté inférence** : on quantifie le modèle final pour servir plus de requêtes avec
  moins de mémoire. Formats courants : GGUF (llama.cpp), AWQ, GPTQ, bitsandbytes.

La quantification d'inférence est traitée en détail au
[chapitre 09](../09-inference-optimisation/README.md). Point important : quantifier
puis fine-tuner ou fine-tuner puis quantifier ne donnent pas les mêmes résultats ;
l'ordre compte et les deux étapes s'évaluent séparément.

## 10. Fusion de LoRA (merge)

Un adaptateur LoRA peut être **fusionné** dans le modèle de base : on calcule
W + alpha/r * (B x A) et on stocke le résultat comme un modèle classique. Avantages :
aucune latence supplémentaire à l'inférence, un seul artefact à déployer. Inconvénient :
on perd la modularité (on ne peut plus désactiver l'adaptateur).

La fusion de plusieurs adaptateurs (LoRA soupe, TIES, DARE) tente de combiner des
compétences distinctes dans un seul modèle. C'est un domaine actif, avec des résultats
variables : les adaptateurs fusionnés peuvent interférer et dégrader certaines
capacités.

## 11. RLHF et DPO en pratique

Le chapitre [05](../05-llm/README.md) présente la théorie. En pratique, pour aligner un
modèle sur des préférences :

1. Collecter des paires (réponse préférée, réponse rejetée) sur ta tâche.
2. Lancer DPO avec un learning rate très faible (de l'ordre de 1e-6 à 5e-7) et peu
   d'époques (1 à 3), car DPO déstabilise vite.
3. Surveiller la dérive : le modèle peut devenir répétitif ou perdre en généralité.
4. Évaluer à la fois sur la tâche cible et sur des tâches générales de contrôle.

Le RLHF complet (reward model + PPO) est rarement nécessaire pour un cas d'usage privé :
DPO offre un rapport bénéfice/complexité bien meilleur, à moins d'avoir des moyens
d'infrastructure importants.

## 12. Évaluer un fine-tuning

Sans évaluation rigoureuse, le fine-tuning est de la loterie.

| Dimension | Ce qu'on mesure | Risque |
|---|---|---|
| Tâche cible | Exactitude sur un jeu de test tenu à l'écart | Sur-apprentissage sur le format |
| Généralité | Performance sur des tâches hors domaine | Oubli catastrophique |
| Robustesse | Variations de formulation du prompt | Fragilité inattendue |
| Sécurité | Comportement sur entrées nuisibles | Perte des garde-fous |

Bonnes pratiques :

- Toujours garder un jeu de validation séparé, jamais vu à l'entraînement.
- Comparer au modèle de base sur les mêmes échantillons.
- Tester des prompts différents du format d'entraînement (le modèle doit généraliser,
  pas mémoriser le template).
- Vérifier les capacités hors domaine (généralité) avant/après.

Signal typique de sur-apprentissage : la loss de validation remonte alors que celle
d'entraînement descend, et les réponses deviennent rigides ou répétitives.

## 13. Apprentissage continu

> Apprentissage continu (continual learning) : mettre à jour un modèle déjà déployé avec
> de nouvelles données, sans tout réentraîner.

Situation fréquente : le monde change, les données évoluent, mais le modèle reste figé.
Options :

- **Réentraîner périodiquement** sur un mélange ancien + nouveau (replay) : simple et
  efficace.
- **Ajouter un adaptateur LoRA** par période et les empiler : modulaire mais les
  adaptateurs peuvent entrer en conflit.
- **Éviter l'oubli catastrophique** en conservant un noyau de données historiques.

En pratique, pour la plupart des applications, la mise à jour des connaissances
factuelles passe par le RAG, et le fine-tuning ne sert qu'aux changements de
comportement ([chapitre 07](../07-rag/README.md)).

## Ce qu'il faut retenir
- Le fine-tuning est rarement le premier outil : prompt engineering et RAG passent avant.
- Le full fine-tuning coûte cher en VRAM et risque l'oubli catastrophique.
- PEFT et LoRA n'entraînent qu'une fraction des paramètres en gelant le modèle de base.
- QLoRA permet de fine-tuner un modèle 7B sur un GPU grand public grâce à la quantification 4-bit.
- Le rang r et alpha contrôlent la capacité de l'adaptateur ; la convention alpha = 2r est courante.
- La qualité des données prime largement sur leur quantité.
- La distillation transfère les compétences d'un grand modèle vers un petit.
- DPO est plus simple et stable que le RLHF complet pour la plupart des cas.
- Évaluer sur des données non vues et sur des tâches hors domaine est indispensable.

## Erreurs frequentes / idees recues
- "Fine-tuner va ajouter des connaissances au modèle" -> faux : le RAG est plus adapté pour injecter des faits.
- "Il faut des millions d'exemples" -> faux : quelques milliers bien choisis suffisent souvent pour un style ou un format.
- "LoRA donne toujours un résultat moins bon que le full fine-tuning" -> souvent proche, parfois meilleur en généralité grâce au gel des poids.
- "Un learning rate élevé accélère tout" -> faux : en fine-tuning, un LR trop haut détruit le modèle.
- "On peut ignorer le chat template" -> faux : un mauvais template dégrade fortement les performances.
- "Un score de validation bas signifie un bon modèle" -> faux : c'est souvent du sur-apprentissage au format.

## Pour aller plus loin
- [Chapitre 05-llm](../05-llm/README.md) : post-training, RLHF/DPO, modèles de raisonnement.
- [Chapitre 07-rag](../07-rag/README.md) : alternative au fine-tuning pour les connaissances.
- [Chapitre 09-inference-optimisation](../09-inference-optimisation/README.md) : quantification d'inférence et formats.
- [Chapitre 10-infrastructure](../10-infrastructure/README.md) : VRAM, GPU, parallélisme.
- [Chapitre 12-donnees](../12-donnees/README.md) : collecte, nettoyage et synthèse de données.
- [Chapitre 16-pratique](../16-pratique/README.md) : tutoriel QLoRA pas à pas.
- Ressources externes :
  - Article *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021).
  - Article *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023).
  - Article *Direct Preference Optimization* (Rafailov et al., 2023).
  - Documentation Hugging Face PEFT et TRL.
