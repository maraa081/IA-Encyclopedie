# Multimodalité

> Comprendre comment les systèmes d'intelligence artificielle intègrent et croisent différents types d'informations (texte, image, audio, vidéo) au sein d'une architecture unifiée pour percevoir et générer du contenu riche.

## 1. Concepts de base de la multimodalité

La multimodalité désigne la capacité d'un modèle d'intelligence artificielle à traiter, aligner et générer des données provenant de différentes modalités.
Ces modalités peuvent être du texte, de l'image, du son, de la vidéo ou encore des données capteurs (Lidar, thermiques).
Dans les architectures traditionnelles, chaque modalité était traitée par un modèle spécialisé.
Par exemple, un réseau de neurones convolutif (CNN) gérait les images, tandis qu'un réseau récurrent (RNN) s'occupait du texte.
La multimodalité moderne vise à fusionner ces flux hétérogènes au sein d'un espace de représentation commun et partagé.

On distingue historiquement deux approches principales pour fusionner les modalités :
*   **La fusion tardive (Late Fusion)** :
    Chaque modalité est d'abord traitée de manière indépendante par son propre encodeur spécialisé.
    Ces encodeurs produisent des prédictions ou des vecteurs de caractéristiques de haut niveau.
    Ces vecteurs sont ensuite combinés en fin de réseau (par concaténation, somme pondérée ou via un classifieur final).
    Cette méthode est simple à mettre en œuvre.
    Cependant, elle limite la capacité du modèle à apprendre des interactions complexes et précoces entre les modalités.
*   **La fusion précoce (Early Fusion)** :
    Les données brutes ou les représentations de bas niveau de chaque modalité sont combinées dès les premières étapes du réseau.
    Les interactions croisées sont ainsi apprises à travers toutes les couches de calcul.
    C'est l'approche privilégiée par les architectures modernes basées sur les Transformers.
    Ici, les différents types de données sont convertis en jetons (tokens) et projetés dans un espace de dimension identique pour être traités ensemble par les mêmes couches d'attention.

L'alignement multimodal repose sur la projection de caractéristiques sémantiques hétérogènes dans un espace latent partagé.
Par exemple, pour l'alignement texte-image, le modèle doit apprendre qu'un vecteur représentant l'image d'un chat noir et un vecteur représentant le texte "un chat noir" doivent être très proches.
Cette proximité s'évalue par une faible distance cosinus au sein de l'espace commun.

L'une des innovations majeures de cet alignement est l'apprentissage contrastif, popularisé par CLIP (Contrastive Language-Image Pre-training) développé par OpenAI en 2021.
CLIP utilise deux encodeurs distincts (un pour les images, un pour le texte) entraînés simultanément sur des centaines de millions de paires image-texte du Web.
L'objectif est de maximiser la similarité cosinus entre l'image et sa légende correspondante au sein d'un lot (batch) de taille $N$.
En même temps, le modèle minimise la similarité avec toutes les autres légendes non correspondantes de ce lot.

Mathématiquement, pour un lot de $N$ paires d'images et de textes, CLIP génère un ensemble de représentations visuelles $I = \{i_1, i_2, ..., i_N\}$ et textuelles $T = \{t_1, t_2, ..., t_N\}$.
La fonction d'optimisation (perte contrastive symétrique) calcule les scores de similarité :

$$S_{j,k} = \cos(i_j, t_k) \cdot e^{\tau}$$

où $\tau$ est un paramètre de température appris.
Le modèle est entraîné en optimisant la perte d'entropie croisée le long des lignes et des colonnes de cette matrice de similarité $N \times N$, poussant les paires correctes vers la diagonale.

## 2. Vision-Language Models (VLM)

Les modèles de vision-langage (VLM) sont des systèmes capables de comprendre simultanément des images et du texte.
Ils accomplissent des tâches telles que la description d'image (captioning), la réponse à des questions visuelles (Visual Question Answering, VQA) ou la détection d'objets par commande textuelle.

L'architecture standard d'un VLM moderne (comme LLaVA, LLaVA-NeXT ou Qwen2-VL) se compose de trois éléments principaux :

1.  **Un encodeur visuel** :
    Typiquement un Vision Transformer (ViT).
    Contrairement aux réseaux convolutifs, un ViT découpe l'image d'entrée en une grille de patchs carrés (par exemple, de taille 14x14 pixels).
    Chaque patch est aplati en un vecteur, projeté linéairement, puis traité comme un jeton textuel classique à travers des couches d'attention.
    Pour une image de 224x224 pixels découpée en patchs de 14x14, on obtient (224/14) * (224/14) = 256 tokens visuels.
2.  **Un connecteur (ou couche de projection)** :
    Les embeddings produits par l'encodeur visuel n'ont généralement ni la même dimension physique, ni le même espace sémantique que les embeddings textuels du modèle de langage.
    Le connecteur projette les tokens visuels de la dimension d'origine de l'encodeur (par exemple, 1024) vers la dimension d'entrée du LLM (par exemple, 4096).
    Ce connecteur peut être une simple matrice de projection linéaire ou un petit perceptron multicouche (MLP) à deux ou trois couches linéaires séparées par une activation non linéaire.
3.  **Un grand modèle de langage (LLM)** :
    Le LLM reçoit une séquence mixte contenant à la fois les tokens de l'image projetés et les tokens du texte d'instruction.
    Le LLM traite cette séquence hybride de manière autorégressive pour générer la réponse textuelle.

```
[ Image brute ] -> [ Encodeur ViT ] -> [ Tokens visuels (D=1024) ]
                                                   |
                                           [ Connecteur MLP ]
                                                   |
                                                   v
[ Texte prompt ] -> [ Tokenizer ]   -> [ Tokens textuels (D=4096) ] -> [ LLM Décodeur ] -> [ Réponse ]
```

Cette architecture permet de réutiliser des modèles pré-entraînés puissants (comme un encodeur CLIP pour la vision et un modèle Llama pour le texte).
On peut ainsi n'entraîner au départ que le connecteur, ce qui réduit considérablement le coût de calcul.
Lors de phases d'entraînement ultérieures (le fine-tuning supervisé ou le RLHF multimodal), les poids du LLM peuvent également être ajustés pour améliorer la précision sémantique globale et la cohérence de la réponse visuelle.

Certains VLMs avancés comme Qwen2-VL introduisent des mécanismes de traitement multi-échelle (Native Dynamic Resolution).
Ils permettent de traiter des images de n'importe quelle taille sans distorsion.
Pour ce faire, ils remplacent les embeddings positionnels de grille fixes par des embeddings positionnels 3D (prenant en compte la hauteur, la largeur et le temps pour les flux vidéo).

## 3. Traitement audio et parole

Le domaine de l'audio multimodal englobe la reconnaissance automatique de la parole (Speech-to-Text), la synthèse vocale (Text-to-Speech) et l'analyse de signaux sonores non verbaux (bruitages, musique, émotions).

Pour traiter l'audio avec des neurones, le signal analogique continu doit d'abord être converti sous forme numérique.
Cette étape requiert un échantillonnage (généralement à 16 kHz ou 24 kHz) puis l'application d'une transformée de Fourier à court terme (STFT).
On obtient ainsi une représentation temps-fréquence appelée spectrogramme de Mel.
Ce spectrogramme est une matrice en deux dimensions représentant l'évolution de l'énergie du signal sur différentes bandes de fréquences au cours du temps, adaptées à la sensibilité de l'oreille humaine.

Le modèle Whisper d'OpenAI utilise cette approche de bout en bout :
1.  Le signal audio brut est découpé en segments de 30 secondes.
2.  Chaque segment est transformé en un spectrogramme de Mel à 80 ou 128 canaux avec des fenêtres glissantes.
3.  Ce spectrogramme est traité par un petit réseau de neurones convolutif (CNN) qui extrait des caractéristiques locales et réduit la résolution temporelle d'un facteur 2.
4.  Les caractéristiques extraites sont transmises à un encodeur Transformer standard.
5.  Un décodeur Transformer génère ensuite les jetons textuels correspondants de manière autorégressive.

Whisper intègre également des jetons spéciaux dans son dictionnaire pour contrôler le comportement de la génération :
*   Identifiant de langue : `<|fr|>` pour le français, `<|en|>` pour l'anglais.
*   Tâche requise : transcription `<|transcribe|>` ou traduction directe vers l'anglais `<|translate|>`.
*   Repères temporels : des tokens de temps spéciaux (par exemple `<|0.00|>` à `<|30.00|>`) sont intercalés entre les mots transcrits pour caler l'affichage du texte sur le flux audio de manière millimétrée.

Dans les architectures de parole de pointe (comme Gemini 1.5 Pro ou GPT-4o), on observe une transition importante vers des modèles d'audio natifs.
Au lieu d'employer des blocs distincts de transcription textuelle intermédiaire, le signal audio brut est directement tokenisé sous forme de jetons discrets (à l'aide d'outils de compression neuronale comme EnCodec ou SoundStream).
Ces jetons audio se comportent exactement comme des mots.
Ils permettent au décodeur du LLM d'apprendre à écouter et à répondre à la voix humaine de manière transparente.
Cela élimine les latences de transcription et capture fidèlement les intonations, les hésitations et les bruits ambiants.

## 4. Génération d'images et modèles de diffusion

La génération d'images à partir de descriptions textuelles (Text-to-Image) a été transformée par les modèles de diffusion.
Le principe fondamental de la diffusion consiste à détruire progressivement l'information d'une image en y ajoutant du bruit blanc gaussien (processus direct, "forward process").
Ensuite, on entraîne un réseau de neurones à inverser ce processus pour reconstruire pas à pas l'image d'origine à partir du bruit pur (processus inverse, "backward process").

Pour éviter d'effectuer ces calculs sur des images de haute résolution (ce qui exigerait une VRAM phénoménale), les modèles de diffusion latente (Latent Diffusion Models, LDM), comme Stable Diffusion, effectuent ce processus dans un espace compressé :

1.  **L'Auto-encodeur variationnel (VAE)** :
    Un encodeur compresse une image de taille $512 \times 512 \times 3$ (pixels RGB) en une représentation latente de taille $64 \times 64 \times 4$.
    Cela réduit la quantité d'éléments à traiter d'un facteur 48.
    Le décodeur effectue l'opération inverse à la toute fin pour restituer l'image finale sous forme de pixels RGB nets.
2.  **Le réseau de débruitage** :
    Un réseau (historiquement un U-Net, de plus en plus remplacé par des structures de type Diffusion Transformer ou DiT) est entraîné à prédire la quantité exacte de bruit ajoutée à un instant $t$ de la diffusion.
3.  **Le conditionnement textuel** :
    La description textuelle saisie par l'utilisateur (le prompt) est encodée par un modèle de texte (comme CLIP ou un modèle de langage massif comme T5-XXL).
    Cet encodeur produit des embeddings de contexte.
    Ces embeddings sont injectés dans le réseau de débruitage via des couches d'attention croisée (Cross-Attention).
    Elles modifient le comportement du modèle à chaque étape de débruitage pour qu'il s'aligne sur la description sémantique fournie.

Le processus d'échantillonnage (sampling) consiste à exécuter de manière répétée (généralement entre 20 et 50 étapes) le réseau de débruitage.
À chaque étape, le modèle prédit le bruit, le soustrait partiellement, et réinjecte une fraction du signal nettoyé.
Ce calcul suit une équation différentielle résolue par des samplers mathématiques sophistiqués (comme DDIM, Euler Ancestral, ou DPM-Solver).

Le tableau suivant résume les principaux modèles de génération d'images et leurs caractéristiques clés :

| Modèle | Développeur | Architecture principale | Taille du modèle (Paramètres) | Espace de diffusion | Description & Caractéristiques |
|---|---|---|---|---|---|
| Stable Diffusion 1.5 | Stability AI | U-Net + CLIP Text Encoder | ~1,0 milliard | Espace latent (VAE) | Modèle de référence open-source, très flexible pour le contrôle de pose (ControlNet) et la personnalisation (LoRA). |
| Stable Diffusion XL | Stability AI | U-Net + Double CLIP (ViT-L & OpenCLIP) | ~2,3 milliards | Espace latent (VAE) | Génération native en résolution 1024x1024, meilleure qualité anatomique et rendu textuel amélioré. |
| PixArt-alpha | Huawei / Académique | Diffusion Transformer (DiT) + T5-XXL | ~600 millions | Espace latent (VAE) | Utilise un Transformer à la place du U-Net classique. Très efficace en calcul avec une excellente cohérence textuelle. |
| Flux.1 Schnell | Black Forest Labs | Rectified Flow Transformer + T5 & CLIP | ~12 milliards | Espace latent (VAE) | Modèle haut de gamme, excellent rendu des visages, des mains et du texte lisible dans l'image en seulement 4 étapes. |
| Imagen 3 | Google | U-Net + T5-XXL | non communiquée | Espace pixel direct | Génération directe de haute qualité, alignement sémantique exceptionnel sans passer par une compression de type VAE. |

## 5. Vidéo et génération 3D

La génération de vidéos et d'objets 3D constitue la frontière actuelle de la multimodalité.
Elle nécessite la prise en compte de contraintes de cohérence temporelle et spatiale extrêmement complexes.

La génération vidéo s'appuie sur deux types d'extensions des modèles d'images :
*   **La diffusion spatio-temporelle** :
    Les convolutions et les blocs d'attention des U-Net d'images sont dotés d'une dimension temporelle supplémentaire (attention temporelle).
    Le modèle traite des séquences de trames et apprend à maintenir la cohérence des objets et des mouvements d'une image à l'autre.
*   **Les Transformers de diffusion vidéo (Video DiT)** :
    Des modèles comme Sora (OpenAI) ou Runway Gen-3 découpent les vidéos en "patchs spatio-temporels" (cubes de pixels s'étendant sur plusieurs images).
    Ces patchs sont aplatis, projetés et traités par un gigantesque Transformer.
    Cela permet de modéliser des interactions physiques plus réalistes sur des durées plus longues (jusqu'à une minute).

Pour la génération 3D, les deux principales approches sont :
1.  **La distillation de score (Score Distillation Sampling, SDS)** :
    Elle utilise un modèle de diffusion 2D pré-entraîné pour guider l'optimisation d'une représentation 3D sous différents angles de caméra.
    Cette représentation peut être un champ de radiance de neurones (NeRF).
    L'optimisation se poursuit jusqu'à ce que les rendus 2D générés soient réalistes sous tous les angles.
2.  **La génération directe par diffusion 3D** :
    Des modèles récents sont entraînés directement sur des bases de données d'objets 3D.
    Ils produisent des nuages de points, des maillages ou des représentations par splatting gaussien 3D (3D Gaussian Splatting) en une seule étape d'inférence.

## 6. Évaluation des modèles multimodaux

Évaluer un modèle multimodal est complexe car il faut mesurer à la fois la fidélité textuelle et la pertinence perceptuelle.
Plusieurs métriques et benchmarks sont utilisés par la communauté pour comparer les performances :
*   **CIDEr et BLEU** : Historiquement utilisés pour le captioning d'images en comparant les n-grammes de la description générée par rapport à des légendes de référence humaines.
*   **CLIP Score** : Mesure la similarité cosinus entre les embeddings de l'image générée (ou analysée) et le texte associé, évaluant l'alignement sémantique global.
*   **MMBench et MME** : Benchmarks standardisés sous forme de questionnaires à choix multiples couvrant des centaines de sous-tâches visuelles (raisonnement spatial, lecture de graphiques, reconnaissance de texte dans l'image, etc.).
*   **Évaluation humaine et RLHF** : Jugement direct par des experts ou des utilisateurs finaux pour évaluer l'utilité, la sécurité et le réalisme esthétique des contenus générés.
*   **Tests de robustesse contradictoire** : Introduction d'artefacts visuels ou d'ambiguïtés textuelles pour mesurer la résistance des VLM face aux hallucinations.

## 7. Cas pratique d'intégration multimodale

Voici un exemple concret en Python utilisant la bibliothèque Hugging Face `transformers` pour charger et exécuter un modèle de vision-langage open-source (LLaVA-1.5) afin d'analyser le contenu d'une image.
Ce script illustre comment l'image et le texte d'instruction sont préparés conjointement avant d'être envoyés au décodeur.

```python
import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 1. Chargement du modèle de vision-langage et de son processeur
# Llava-1.5 associe un encodeur ViT-L-14 à un LLM Vicuna de 7 milliards de paramètres.
model_id = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Chargement du processeur et du modèle sur le périphérique : {device}")
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

# 2. Récupération d'une image exemple
url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1"  # Image d'un chien
image = Image.open(requests.get(url, stream=True).raw)

# 3. Définition du prompt avec le tag obligatoire <image> qui indique l'emplacement des tokens visuels
prompt = "USER: <image>\nDécris précisément ce que tu vois sur cette image, ainsi que l'arrière-plan. ASSISTANT:"

# 4. Prétraitement conjoint de l'image et du prompt
# Le processeur gère la redimension de l'image pour le ViT et la tokenisation du texte pour le LLM.
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)

# 5. Génération de la réponse textuelle par le LLM
print("Génération de la description...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.2,
        top_p=0.9
    )

# 6. Décodage et affichage du résultat
response = processor.decode(output_ids[0], skip_special_tokens=True)
print("\nRésultat de l'analyse :")
print(response)
```

Ce flux montre que l'image n'est pas "traduite" en texte avant d'être lue.
Au contraire, ses caractéristiques denses sont injectées directement dans l'espace de calcul du décodeur.

## Ce qu'il faut retenir

- La multimodalité unifie le traitement de types de données hétérogènes (texte, image, son, vidéo) au sein d'un même modèle ou d'un espace de représentation partagé.
- Les modèles de vision-langage (VLM) modernes reposent sur un encodeur d'image (comme un ViT), un connecteur sémantique (MLP) et un grand modèle de langage (LLM) décodeur.
- Pour être traitée par un Transformer, une image est découpée en patchs bidimensionnels convertis en vecteurs (tokens visuels), mimant la structure linéaire du langage.
- Les signaux audio sont généralement convertis en spectrogrammes de Mel pour être traités comme des images, ou tokenisés directement sous forme de codes audio discrets.
- Les modèles de diffusion latente effectuent le processus de génération d'images au sein d'un espace compressé par un auto-encodeur (VAE) pour économiser la mémoire de calcul.
- La génération de vidéo et de 3D utilise désormais massivement les Transformers de diffusion (DiT) qui améliorent le passage à l'échelle par rapport aux anciens U-Net.
- Le choix de la métrique d'évaluation est crucial, car la fidélité sémantique diffère souvent de la perception esthétique humaine.

## Erreurs fréquentes / idées reçues

- L'IA "voit" les images pixel par pixel -> Faux. Les encodeurs de vision modernes découpent l'image en patchs de taille fixe (par exemple, 14x14 ou 16x16 pixels) et analysent les relations spatiales globales via des mécanismes d'attention, ou extraisent des caractéristiques d'échelle intermédiaire.
- On a besoin de réentraîner entièrement un LLM pour lui ajouter la vision -> Faux. On peut geler les paramètres de l'encodeur d'image et du LLM pré-entraînés, et n'entraîner que le connecteur (quelques millions de paramètres seulement) lors d'une première phase d'alignement.
- Les modèles d'images génèrent des pixels directement un par un -> Faux. La plupart des modèles modernes (comme Stable Diffusion ou Flux) travaillent dans un espace de caractéristiques latentes réduites, et le rendu final en pixels est assuré par un décodeur de type VAE.
- Un modèle de transcription comme Whisper comprend la sémantique de l'audio -> Faux. Même s'il est très performant, Whisper transcrit en se basant sur la reconstruction de séquences de jetons textuels conditionnés par l'audio. Les modèles d'interaction vocale en temps réel natifs doivent utiliser des tokens audio bidirectionnels pour capter l'intonation, le rythme et l'émotion de manière continue.

## Pour aller plus loin

- [chapitre 04](../04-transformers/README.md) : Pour réviser le fonctionnement de l'attention et des architectures de Transformers appliquées aux séquences.
- [chapitre 05](../05-llm/README.md) : Pour approfondir l'apprentissage autorégressif des grands modèles de langage servant de décodeurs multimodaux.
- [chapitre 10](../10-infrastructure/README.md) : Pour analyser l'impact du format de précision (FP16, INT8, BF16) sur l'utilisation de la VRAM lors de la manipulation de gros modèles.
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Alexey Dosovitskiy et al., 2020) : Le papier fondateur du Vision Transformer (ViT) montrant comment appliquer l'attention aux patchs d'images.
- "Generative Learning Trilemma" (Aditya Ramesh et al., 2022) : Une explication approfondie des compromis de conception entre la vitesse d'inférence, la qualité d'image et la couverture de mode dans les modèles génératifs.
- "Understanding Diffusion Models: A Unified Perspective" (Calvin Luo, 2022) : Un excellent guide théorique sur le fonctionnement mathématique des modèles de diffusion.
