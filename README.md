# Reconnaissance de Chiffres Manuscrits

Ce projet utilise un modèle de réseau de neurones convolutionnel (CNN) pour reconnaître les chiffres manuscrits. L'application est divisée en un backend (API) développé avec FastAPI et un frontend développé avec Streamlit.

## Table des matières

- [Structure du Projet](#structure-du-projet)

## Structure du Projet

```plaintext
.
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── main.py
├── model
│   └── mnist-0.0.1.pt
├── src
│   ├── app
│   │   ├── back.py
│   │   └── front.py
│   └── model
│       ├── mlp.py
│       └── convnet.py
└── README.md
