# JVLIVS

> **Non pas un assistant IA de plus. Un Personal AI Runtime.**

**JVLIVS** (prononcé _"Julius"_, commande CLI `jvl`) est un environnement d'exécution IA personnel (_Personal AI Runtime_) conçu pour orchestrer vos modèles (locaux et distants), appliquer vos propres règles de développement, mémoriser votre contexte d'un projet à l'autre et exécuter des workflows automatisés complexes.

Développé en mode _CLI-first_, JVLIVS propose une alternative locale, configurable, persistante et transparente aux assistants IA génériques et amnésiques.

---

## Vision Produit & Positionnement

Les assistants IA traditionnels (Claude Code, Cursor, Aider) sont pensés pour le grand public. Ils souffrent d'amnésie entre deux sessions, ignorent vos contextes métier et n'offrent que peu de contrôle. JVLIVS résout cela :

| Fonctionnalité            | Assistants IA Génériques        | JVLIVS                                             |
| :------------------------ | :------------------------------ | :------------------------------------------------- |
| **Nature**                | Assistants de codage ponctuels  | **Personal AI Runtime**                            |
| **Mémoire**               | Par session uniquement          | **Persistante & structurée** (global + par projet) |
| **Modèles**               | Monoproviders / APIs uniquement | **Multi-backend** (Local + API)                    |
| **Customisation**         | Limitée (ex: `.claudecode`)     | **Configuration complète** (YAML + Markdown)       |
| **Workflows**             | Non supportés                   | **Moteur déclaratif** de workflows YAML            |
| **Base de Connaissances** | Non intégrée                    | **RAG local intégré** (PDF, Markdown, URL...)      |

---

## Stack Technique

JVLIVS s'appuie sur une stack robuste, moderne et performante :

- **Gestion de projet & Dépendances** : [`uv`](https://github.com/astral-sh/uv) (remplaçant ultra-rapide de pip, venv et poetry).
- **CLI Framework** : [`Typer`](https://typer.tiangolo.com/) pour la structure des commandes et le typage automatique.
- **Rendu Terminal** : [`Rich`](https://github.com/Textualize/rich) (Markdown, streaming fluide sans flickering) & [`prompt_toolkit`](https://github.com/prompt-toolkit/python-prompt-toolkit) (REPL interactif).
- **Abstraction LLM** : Implémentations maison par provider ([`openai`](https://github.com/openai/openai-python), [`anthropic`](https://github.com/anthropics/anthropic-sdk-python), [`httpx`](https://github.com/encode/httpx) direct pour Ollama) pour une maîtrise totale du streaming et des API.
- **Base de Données** : SQLite avec [`SQLAlchemy`](https://www.sqlalchemy.org/) pour la persistance locale des sessions.
- **Validation Config** : YAML avec [`Pydantic`](https://docs.pydantic.dev/) pour une validation stricte dès le lancement.
- **Qualité de code** : [`ruff`](https://github.com/astral-sh/ruff) pour le formatage et le linting rapides, [`pytest`](https://docs.pytest.org/) pour la suite de tests.

---

## Installation & Lancement (v0.1)

Le projet utilise **uv** pour simplifier le développement local.

### Prérequis

- Python 3.11+
- [Ollama](https://ollama.com/) (pour les modèles locaux) et/ou des clés d'API (OpenAI, Anthropic)

### 1. Cloner et installer les dépendances

```bash
git clone <url-du-repo>
cd jvlivs
uv sync
```

### 2. Configurer l'environnement

Copiez le fichier d'exemple et remplissez vos variables d'environnement (clés d'API, etc.) :

```bash
cp .env.example .env
```

### 3. Exécuter en mode développement

Vous pouvez lancer le CLI directement via `uv` :

```bash
uv run jvl --help
```

Ou activer l'environnement virtuel pour utiliser la commande `jvl` directement :

```bash
source .venv/bin/activate
jvl --help
```


---

## Configuration & Commandes CLI

JVLIVS utilise un système de configuration à deux niveaux :
1. **Repository Config** : Le fichier `config/config.yaml` sert de base et de configuration par défaut pour le développement.
2. **User Config** : Persistée localement dans `~/.jvl/config.yaml`, elle permet à chaque utilisateur de configurer ses propres clés API, endpoints personnalisés et modèles préférés.

### Le groupe de commandes `jvl config`

Gérez votre configuration utilisateur directement en ligne de commande :

#### 🔄 Migration initiale
```bash
jvl config migrate
```
Copie automatiquement la configuration du repo de développement dans votre configuration utilisateur `~/.jvl/config.yaml` pour démarrer instantanément.

#### 🔌 Gestion des Providers (`provider`)
Configurez n'importe quel provider natif (`ollama`, `openai`, `anthropic`, `azure`, `gemini`) :
```bash
# Ajouter ou modifier un provider
jvl config provider add ollama --base-url http://localhost:11434 --default-model "llama3.2:3b"
jvl config provider add openai --default-model "gpt-4o"

# Lister les providers configurés et voir le statut
jvl config provider list

# Supprimer un provider
jvl config provider remove anthropic
```

#### 🤖 Gestion des Modèles (`model`)
```bash
# Définir le modèle par défaut d'un provider
jvl config model set openai gpt-4o-mini

# Définir de façon permanente le provider et le modèle actif par défaut
jvl config model use openai gpt-4o

# Afficher le provider et modèle actif résolu
jvl config model show
```

#### 🔑 Gestion des Clés API (`key`)
```bash
# Définir une clé API en clair
jvl config key set openai sk-proj-...

# Lier une clé API à une variable d'environnement (recommandé)
jvl config key set openai --env OPENAI_API_KEY

# Lister les clés configurées (masquées)
jvl config key show

# Supprimer la clé d'un provider
jvl config key remove openai
```

---

## Utilisation & Changement dynamique de Modèle

### Flags CLI
Vous pouvez écraser temporairement le modèle ou le backend actif lors d'une commande grâce aux options `--backend` (`-b`) et `--model` (`-m`) :
```bash
jvl ask -b openai -m gpt-4o "Explique-moi la relativité en 3 phrases."
jvl chat -b anthropic -m claude-sonnet-4-5
```

### Picker interactif
Si vous lancez la commande `jvl model` sans argument, ou si vous utilisez la commande `/model` dans le REPL de chat sans paramètre, JVLIVS lance un picker interactif en deux étapes :
1. **Choix du provider** parmi ceux disponibles.
2. **Choix du modèle** :
   - Pour **Ollama**, JVLIVS interroge l'API locale pour lister les modèles réellement installés sur votre machine.
   - Pour les **providers Cloud**, JVLIVS propose une liste des modèles de référence ainsi qu'une option de saisie libre.

---


## Auteur

**Ethan Benchetrit** — AI Engineer @ WeDR  
Lyon / Paris — 2026
