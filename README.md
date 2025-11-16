# README

## Overview

This project is a CLI-based Retrieval-Augmented Generation (RAG) application. It loads a text document, builds embeddings using Sentence Transformers, stores them in ChromaDB, and answers queries using the Mistral model running locally via Ollama.

## Requirements

* Python 3.9 or later
* Ollama installed locally
* Mistral model pulled in Ollama
* Git

## Setup

### 1. Clone the repository

```
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create and activate a virtual environment

```
python -m venv venv
```

Windows:

```
venv\Scripts\activate
```

macOS/Linux:

```
source venv/bin/activate
```

### 3. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and set up Ollama

Install Ollama from [https://ollama.ai](https://ollama.ai)

Pull the Mistral model:

```
ollama pull mistral
```

Verify:

```
ollama list
```

## Run

```
python main.py
```

## Files

* main.py
* speech.txt
* requirements.txt
