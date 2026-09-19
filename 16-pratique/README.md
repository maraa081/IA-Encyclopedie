# Mise en pratique : tutoriels

Ce chapitre propose une série de tutoriels autonomes pour mettre en pratique les concepts abordés dans l'encyclopédie. Chaque tutoriel est conçu pour être réalisé étape par étape, avec un objectif clair, des prérequis, le code complet, les commandes à exécuter, le résultat attendu et les pièges courants à éviter.

## Tutoriel 1 : Premier modèle de langage avec PyTorch

**Objectif** : Entraîner un petit modèle de langage générateur de caractères sur un texte simple.

**Prérequis** :
- Python 3.8+
- PyTorch installé (`pip install torch`)
- Un corpus de texte (exemple : une courte histoire)

**Code complet** :
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.data = [self.stoi[c] for c in text]
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Hyperparamètres
block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Lecture du corpus
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

train_size = int(0.9 * len(text))
train_data = text[:train_size]
val_data = text[train_size:]

train_dataset = CharDataset(train_data, block_size)
val_dataset = CharDataset(val_data, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = BigramLanguageModel(len(train_dataset.stoi))
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for k, (x, y) in enumerate(loader):
            if k >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# Boucle d'entraînement
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Génération
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = m.generate(context, max_new_tokens=500)[0].tolist()
generated_text = ''.join([train_dataset.itos[i] for i in generated_chars])
print(generated_text)
```

**Commandes** :
```bash
# Créer un fichier d'exemple
echo "hello world. hello AI. hello encyclopedia." > input.txt
# Exécuter le script
python train_char.py
```

**Résultat attendu** : Le modèle apprend à reproduire des séquences de caractères du corpus. Après entraînement, la génération produit du texte ressemblant au langage d'entrée (ex: "hello world. hello AI...").

**Pièges** :
- Utiliser un taux d'apprentissage trop élevé peut provoquer une divergence.
- Un `block_size` trop petit limite la capacité du modèle à apprendre des dépendances longues.
- Négliger de mettre le modèle en mode évaluation lors du calcul de la perte de validation.

---

## Tutoriel 2 : Fine-tuning d'un modèle LLM avec LoRA

**Objectif** : Adapter un modèle pré-entraîné à une tâche spécifique (classification de sentiments) en utilisant LoRA pour réduire les coûts de calcul.

**Prérequis** :
- Python 3.8+
- Bibliothèques : `torch`, `transformers`, `peft`, `datasets`
- Un GPU avec au moins 8 Go de VRAM (ou utiliser CPU avec longueur de séquence réduite)

**Code complet** :
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset

# Charger le modèle et le tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Préparer le modèle pour l'entraînement en 8-bit (optionnel, économise de la VRAM)
model = prepare_model_for_int8_training(model)

# Configuration LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_lin", "v_lin"],  # spécifique à DistilBERT
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

# Appliquer LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Charger un dataset (exemple : SST-2)
dataset = load_dataset("glue", "sst2")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

# Sauvegarde
model.save_pretrained("./lora-sst2")
tokenizer.save_pretrained("./lora-sst2")
```

**Commandes** :
```bash
# Installer les dépendances
pip install torch transformers peft datasets
# Exécuter le script
python fine_tune_lora.py
```

**Résultat attendu** : Le modèle atteint une précision d'environ 90% sur le jeu de validation SST-2 après quelques époques, avec seulement une fraction des paramètres entraînés grâce à LoRA.

**Pièges** :
- Choisir les bons `target_modules` selon l'architecture du modèle (voir documentation PEFT).
- Un taux de dropout trop élevé peut sous-ajuster l'adaptation.
- Négliger de sauvegarder le tokenizer avec le modèle adapté.

**Lien relatif** : Voir le chapitre sur le fine-tuning [06-fine-tuning/README.md](../06-fine-tuning/README.md) pour plus de détails sur LoRA et ses variantes.

---

## Tutoriel 3 : Construire un pipeline RAG simple avec FAISS

**Objectif** : Mettre en place un système de Retrieval-Augmented Génération (RAG) pour répondre à des questions basé sur un corpus de documents.

**Prérequis** :
- Python 3.8+
- Bibliothèques : `sentence-transformers`, `faiss-cpu`, `transformers`
- Un corpus de documents texte (exemple : articles Wikipédia)

**Code complet** :
```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Charger le modèle d'embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Charger le corpus (liste de strings)
corpus = [
    "La tour Eiffel est située à Paris.",
    "Elle a été construite pour l'Exposition universelle de 1889.",
    "Son concepteur était Gustave Eiffel.",
    "La tour mesure 330 mètres de hauteur.",
    "Elle est peinte en trois tons différents."
]

# Créer les embeddings
corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True)

# Construire l'index FAISS
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# Fonction de recherche
def retrieve(query, k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [corpus[i] for i in indices[0]]

# Charger un modèle de génération (exemple : petit modèle de texte)
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def rag_answer(question):
    context = retrieve(question)
    prompt = f"Question: {question}\nContexte: {' '.join(context)}\nRéponse:"
    result = generator(prompt, max_length=100)
    return result[0]['generated_text']

# Exemple d'utilisation
if __name__ == "__main__":
    question = "Qui a conçu la tour Eiffel ?"
    print(rag_answer(question))
```

**Commandes** :
```bash
# Installer les dépendances
pip install sentence-transformers faiss-cpu transformers torch
# Exécuter le script
python rag_simple.py
```

**Résultat attendu** : Le système retourne une réponse basée sur le contexte récupéré, par exemple : "Gustave Eiffel a conçu la tour Eiffel."

**Pièges** :
- Utiliser un index FAISS inadapté à la taille du corpus (IndexFlatL2 est adapté pour de petits corpus seulement).
- Négliger de normaliser les embeddings si l'on utilise une métrique de produit scalaire.
- Le modèle de génération peut ignorer le contexte s'il n'est pas bien conditionné.

**Lien relatif** : Voir le chapitre sur le RAG [07-rag/README.md](../07-rag/README.md) pour des techniques avancées de chunking et de reranking.

---

## Tutoriel 4 : Servir un modèle avec vLLM

**Objectif** : Déployer un modèle de langage pour l'inférence à haute performance en utilisant le moteur vLLM.

**Prérequis** :
- Python 3.8+
- Bibliotheque : `vllm`
- Un modèle compatible (exemple : `facebook/opt-125m`)
- Un GPU avec suffisamment de VRAM (ou utiliser le mode CPU pour les tests)

**Code complet** :
```python
# Aucun script Python nécessaire pour le lancement du serveur, mais voici un exemple d'utilisation client
from vllm import LLM, SamplingParams

# Charger le modèle
llm = LLM(model="facebook/opt-125m")

# Définir les paramètres d'échantillonnage
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

# Générer du texte
prompts = [
    "Bonjour, comment ça va ?",
    "Explique la théorie de la relativité en deux phrases :"
]
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt!r}")
    print(f"Génération: {output.outputs[0].text!r}")
    print()
```

**Commandes** :
```bash
# Installer vLLM
pip install vllm
# Lancer le serveur OpenAI-compatible (optionnel)
# vllm serve facebook/opt-125m --port 8000
# Exécuter le script client
python vllm_example.py
```

**Résultat attendu** : Le modèle génère du texte cohérent en réponse aux prompts, avec une latence réduite grâce à la gestion efficace du KV cache et du batching continu.

**Pièges** :
- S'assurer que le modèle est bien pris en charge par vLLM (consulter la liste des modèles supportés).
- Ajuster la taille du bloc de tokens selon la VRAM disponible pour éviter les erreurs d'allocation de mémoire.
- En mode serveur, vérifier que le port est libre et que le pare-feu autorise les connexions.

**Lien relatif** : Voir le chapitre sur l'optimisation de l'inférence [09-inference-optimisation/README.md](../09-inference-optimisation/README.md) pour des détails sur vLLM et le KV cache.

---

## Tutoriel 5 : Créer un agent qui utilise des outils (ex: calculatrice)

**Objectif** : Implémenter un agent basé sur un LLM capable d'utiliser des outils externes comme une calculatrice pour résoudre des problèmes mathématiques.

**Prérequis** :
- Python 3.8+
- Bibliothèques : `transformers`, `torch`
- Un modèle de langage instructif (exemple : `google/flan-t5-base`)

**Code complet** :
```python
import re
from transformers import pipeline

# Charger un modèle de suivi d'instructions
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Définir un outil simple : calculatrice
def calculator(expression):
    try:
        # Évaluer uniquement des expressions mathématiques sûres
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Erreur : expression non autorisée"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Erreur : {e}"

# Fonction pour extraire une expression mathématique du texte
def extract_expression(text):
    # Recherche de motifs comme "calculer 2+2" ou "quelle est 5*3 ?"
    match = re.search(r'[\d+\-*/().\s]+', text)
    if match:
        return match.group(0).strip()
    return None

# Agent principal
def agent(question):
    # Étape 1 : Déterminer si l'on doit utiliser un outil
    if any(keyword in question.lower() for keyword in ["calculer", "quelle est", "combien font"]):
        expr = extract_expression(question)
        if expr:
            result = calculator(expr)
            return f"Le résultat de {expr} est {result}."
    # Étape 2 : Sinon, utiliser le modèle de langage directement
    prompt = f"Question: {question}\nRéponse:"
    output = generator(prompt, max_length=100)
    return output[0]['generated_text']

# Exemple d'utilisation
if __name__ == "__main__":
    print(agent("Quelle est la racine carrée de 144 ?"))  # Ne contient pas d'opérateurs de base, ira au LLM
    print(agent("Calculer 12 * 7"))
    print(agent("Combien font 100 + 250 ?"))
```

**Commandes** :
```bash
# Installer les dépendances
pip install transformers torch
# Exécuter le script
python tool_agent.py
```

**Résultat attendu** : L'agent utilise la calculatrice pour les expressions mathématiques explicites et recourt au LLM pour les questions nécessitant du raisonnement ou des connaissances externes.

**Pièges** :
- L'utilisation de `eval` présente un risque de sécurité ; dans un environnement de production, utiliser une bibliothèque d'évaluation sûre comme `ast.literal_eval` avec des opérations limitées.
- L'agent peut échouer à détecter l'intention d'utiliser un outil si la formulation est trop variée.
- Le modèle de langage peut fournir des réponses incorrectes même lorsqu'un outil est disponible ; il faut parfois forcer l'utilisation de l'outil via le prompt.

**Lien relatif** : Voir le chapitre sur les agents [08-agents/README.md](../08-agents/README.md) pour des architectures d'agents plus avancées avec mémoire et planification.

---

## Tutoriel 6 : Quantifier un modèle avec GGUF

**Objectif** : Réduire la taille d'un modèle de langage et accélérer son chargement en le convertissant au format GGUF utilisé par llama.cpp.

**Prérequis** :
- Python 3.8+
- Bibliothèques : `torch`, `transformers`, `sentencepiece` (si nécessaire)
- Outils : `llama.cpp` (cloner depuis GitHub et compiler)
- Un modèle de langage compatible (exemple : `TheBloke/Llama-2-7B-chat-GGUF` ou convertir un modèle HuggingFace)

**Code complet** (script de conversion) :
```python
# Ce script montre comment convertir un modèle HuggingFace en GGUF en utilisant les outils de llama.cpp
# Étape 1 : Cloner et compiler llama.cpp (à faire une fois)
# git clone https://github.com/ggerganov/llama.cpp
# cd llama.cpp
# make

# Étape 2 : Convertir le modèle (exemple avec un modèle de 7B)
# Depuis le dossier llama.cpp :
# python convert_hf_to_gguf.py /chemin/vers/le/modelle/huggingface --outtype f16
# Où --outtype peut être f16, q4_0, q4_1, q5_0, q5_1, q8_0, etc.

# Étape 3 : Utiliser le modèle quantifié avec llama.cpp ou un serveur compatible
# ./main -m /chemin/vers/modèle.gguf -p "Bonjour, comment ça va ?" -n 128
```

**Commandes** :
```bash
# Installer les dépendances Python pour la conversion
pip install torch transformers sentencepiece
# Cloner llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# Compiler
make
# Convertir un modèle (exemple : utiliser un petit modèle pour tester)
# Télécharger un modèle de test
git lfs install
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Convertir
python convert_hf_to_gguf.py TinyLlama-1.1B-Chat-v1.0 --outtype q4_0
# Exécuter
./main -m TinyLlama-1.1B-Chat-v1.0-ggml-model-q4_0.gguf -p "Dis-moi une blague sur les IA." -n 100
```

**Résultat attendu** : Le modèle quantifié occupe moins d'espace disque et de VRAM, permettant son exécution sur du matériel plus modeste tout en conservant une qualité de génération acceptable.

**Pièges** :
- La quantification introduit des erreurs d'approximation ; choisir un niveau de quantification adapté au cas d'usage (q4_0 pour un bon compromis, q8_0 pour préserver davantage la qualité).
- Certains modèles nécessitent une adaptation spécifique du script de conversion (vérifier la compatibilité avec `convert_hf_to_gguf.py`).
- Lors de l'exécution avec `llama.cpp`, ajuster les paramètres de contexte (`-c`) et de lot (`-b`) selon les capacités du matériel.

**Lien relatif** : Voir le chapitre sur l'écosystème [15-ecosysteme/README.md](../15-ecosysteme/README.md) pour une présentation de llama.cpp et d'autres cadres d'inférence légère.

---

## Tutoriel 7 : Entraîner un modèle de classification de texte avec HuggingFace Transformers

**Objectif** : Classer des textes en catégories prédéfinies (exemple : détection de spam) en utilisant un modèle pré-entraîné et la bibliothèque Transformers de Hugging Face.

**Prérequis** :
- Python 3.8+
- Bibliothèques : `transformers`, `datasets`, `torch`
- Un dataset de textes étiquetés (exemple : SMS Spam Collection)

**Code complet** :
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np

# Charger le tokenizer et le modèle
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Charger le dataset (exemple : version simplifiée de SMS Spam)
# Ici, nous créons un petit dataset factice pour l'exemple
data = {
    "text": [
        "Gagnez 1000 euros maintenant ! Cliquez ici",
        "Rappel : réunion demain à 10h",
        "Offre limitée ! Répondre STOP pour annuler",
        "Bonjour, comment allez-vous ?",
        "URGENT : Votre compte sera suspendu",
        "Hier j'ai acheté du pain",
        "Félicitations ! Vous avez gagné un voyage",
        "Le projet avance comme prévu"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = pas spam
}
dataset = load_dataset("dict", data["text"], data["label"])  # Cette ligne est illustrative ; en pratique, utiliser load_dataset réel

# Pour un dataset réel, on ferait quelque chose comme :
# dataset = load_dataset("csv", data_files="spam.csv")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Diviser en train/test
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Métrique
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Évaluation
eval_results = trainer.evaluate()
print(f"Précision : {eval_results['eval_accuracy']:.4f}")

# Sauvegarde
model.save_pretrained("./spam-classifier")
tokenizer.save_pretrained("./spam-classifier")
```

**Commandes** :
```bash
# Installer les dépendances
pip install transformers datasets
# Exécuter le script
python text_classification.py
```

**Résultat attendu** : Le modèle atteint une précision élevée sur le jeu de validation (au-dessus de 90% sur un dataset équilibré), démontrant sa capacité à distinguer les spams des messages légitimes.

**Pièges** :
- Un déséquilibre des classes peut biaiser le modèle ; envisager l'utilisation de `class_weight` ou de suréchantillonnage.
- Une longueur de séquence trop courte peut tronquer des informations importantes ; ajuster `max_length` selon la longueur moyenne des textes.
- Négliger d'évaluer sur un ensemble de test totalement inconnu peut conduire à une surestimation des performances.

**Lien relatif** : Voir le chapitre sur le pré-entraînement et le fine-tuning [05-llm/README.md](../05-llm/README.md) et [06-fine-tuning/README.md](../06-fine-tuning/README.md) pour le choix du modèle et les stratégies d'entraînement.

---

## Tutoriel 8 : Évaluer un modèle avec les métriques de perplexité et exactitude

**Objectif** : Calculer la perplexité d'un modèle de langage génératif et l'exactitude d'un classificateur pour évaluer leurs performances.

**Prérequis** :
- Python 3.8+
- Bibliothèques : `torch`, `transformers`, `datasets`
- Un modèle de langage (exemple : `gpt2`) et un dataset de texte (exemple : Wikitext-2)
- Un classificateur entraîné (exemple : celui du tutoriel 7)

**Code complet** :
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# --- Évaluation de la perplexité pour un modèle de langage ---
print("Évaluation de la perplexité :")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Charger un petit extrait de Wikitext-2 pour la démonstration
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"][:5])  # Prendre les 5 premiers articles
encodings = tokenizer(text, return_tensors="pt")

max_length = model.config.n_positions
stride = 512
lls = []
for i in range(0, encodings.input_ids.size(1), stride):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100  # ignore index
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    lls.append(neg_log_likelihood * trg_len)

ppl = torch.exp(torch.stack(lls).sum() / end_loc)
print(f"Perplexité sur Wikitext-2 (extrait) : {ppl.item():.2f}")

# --- Évaluation de l'exactitude pour un classificateur ---
print("\nÉvaluation de l'exactitude :")
# Charger le classificateur sauvegardé du tutoriel 7 (à adapter selon le chemin)
classifier_model = AutoModelForSequenceClassification.from_pretrained("./spam-classifier")
classifier_tokenizer = AutoTokenizer.from_pretrained("./spam-classifier")
classifier_model.eval()

# Quelques exemples de test
test_texts = [
    "Gagnez de l'argent facilement !",
    "Bonjour, voici le rapport demandé.",
    "Cliquez ici pour gagner un prix !",
    "On se voit demain pour déjeuner."
]
test_labels = [1, 0, 1, 0]  # 1 = spam, 0 = pas spam

inputs = classifier_tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = classifier_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

correct = (predictions == torch.tensor(test_labels)).sum().item()
accuracy = correct / len(test_texts)
print(f"Exactitude sur les exemples de test : {accuracy:.2f}")
```

**Commandes** :
```bash
# Installer les dépendances
pip install transformers datasets torch
# Exécuter le script
python evaluate_model.py
```

**Résultat attendu** : La perplexité du modèle GPT-2 sur un extrait de Wikitext-2 devrait être dans les dizaines (valeur inférieure indique une meilleure modélisation). L'exactitude du classificateur sur les exemples de test devrait être élevée si le modèle a bien appris.

**Pièges** :
- Le calcul de la perplexité nécessite une fenêtre glissante pour gérer les longues séquences ; une mauvaise gestion du recouvrement peut biaiser le résultat.
- Lors de l'évaluation d'un classificateur, s'assurer que les exemples de test sont représentatifs et ne proviennent pas du même ensemble que l'entraînement.
- La perplexité est sensible à la tokenizer utilisée ; comparer des modèles nécessite d'utiliser la même tokenizer ou de renormaliser.

**Lien relatif** : Voir le [chapitre 05 sur les LLM](../05-llm/README.md) pour une discussion détaillée sur les métriques d'évaluation et leurs limites.

## Ce qu'il faut retenir

- Un premier réseau de neurones en PyTorch tient en une trentaine de lignes : tenseurs,
  `nn.Module`, boucle d'entraînement, rétropropagation, pas d'optimiseur.
- Charger un modèle Hugging Face se fait en trois objets : le tokenizer, le modèle, et
  la génération ; le template de chat doit correspondre exactement à celui du modèle.
- Un RAG minimal se monte avec trois briques : un modèle d'embedding, un index FAISS,
  et un LLM pour la synthèse. Aucun framework n'est nécessaire pour commencer.
- Servir un modèle local passe par `vllm serve` (GPU, haut débit) ou `ollama run`
  (poste de travail, quantifié), les deux exposant une API compatible OpenAI.
- Le fine-tuning QLoRA rend l'adaptation d'un modèle 7B possible sur une seule carte
  de 16 Go : quantification NF4 du modèle de base, LoRA sur les projections attention
  et MLP, puis fusion des poids.
- Un agent se résume à une boucle : le modèle propose un appel d'outil, le code
  l'exécute, le résultat est réinjecté dans la conversation, jusqu'à une réponse finale.
- L'évaluation d'un RAG repose sur un petit jeu de questions dorées : sans jeu de test,
  aucune optimisation n'est mesurable.
- Mesurer un serveur d'inférence signifie mesurer quatre nombres : temps jusqu'au
  premier token, temps par token, débit global en tokens par seconde, et taux d'erreur.

## Erreurs fréquentes / idées reçues

- « Je commence par installer LangChain » -> commence par les bibliothèques de base ;
  un RAG de vingt lignes se debugge infiniment plus facilement qu'un graphe d'abstractions.
- « Le modèle quantifié sera identique » -> la quantification dégrade d'autant plus que
  le modèle est petit et la quantification agressive ; mesure avant de déployer.
- « LoRA et fine-tuning complet donnent le même résultat » -> LoRA est excellent pour
  un style ou un format, plus limité pour ajouter des connaissances factuelles massives.
- « Une réponse lente signifie que le serveur est saturé » -> le temps jusqu'au premier
  token dépend de la longueur du prompt, pas du débit de génération.
- « Je testerai à la fin » -> sans jeu de test doré dès le début, tu ne sauras pas si
  tes modifications améliorent ou dégradent le système.

## Pour aller plus loin

- [Chapitre 06 : fine-tuning et adaptation](../06-fine-tuning/README.md)
- [Chapitre 07 : RAG](../07-rag/README.md)
- [Chapitre 08 : agents](../08-agents/README.md)
- [Chapitre 09 : optimisation de l'inférence](../09-inference-optimisation/README.md)
- [Aide-mémoire](../CHEATSHEET.md) pour les commandes et les ordres de grandeur.
- Documentation officielle : PyTorch, Hugging Face Transformers, PEFT, TRL, FAISS, vLLM.
