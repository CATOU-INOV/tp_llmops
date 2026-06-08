"""
Script de création des 5 prompts dans Langfuse avec le label 'production'.
Usage : python scripts/seed_langfuse.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not _PUBLIC_KEY or not _SECRET_KEY:
    raise SystemExit("Erreur : LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY doivent être définis dans .env")

from langfuse import Langfuse

_client = Langfuse(public_key=_PUBLIC_KEY, secret_key=_SECRET_KEY, host=_HOST)

_PROMPTS = [
    {
        "name": "service_client",
        "prompt": (
            "Tu es un assistant service client professionnel et empathique.\n"
            "Réponds en français de façon concise et utile\n"
            "Client : {{_message}}\n"
            "Assistant :"
        ),
    },
    {
        "name": "filtre_politique",
        "prompt": (
            "Analyse le message suivant et détermine s'il contient du contenu à caractère politique.\n"
            "Réponds uniquement par OUI ou NON.\n"
            "Message : {{_message}}\n"
            "Réponse :"
        ),
    },
    {
        "name": "juge_coherence",
        "prompt": (
            "Évalue si la réponse suivante est cohérente avec l'intention du message utilisateur.\n"
            "Retourne un score décimal entre 0 et 1 (0 = incohérent, 1 = parfaitement cohérent).\n"
            "Message : {{_message}}\n"
            "Réponse : {{_reponse}}\n"
            "Score :"
        ),
    },
    {
        "name": "juge_ton",
        "prompt": (
            "Évalue si le ton de la réponse est approprié au contexte attendu (professionnel, empathique).\n"
            "Retourne un score décimal entre 0 et 1 (0 = ton inapproprié, 1 = ton parfait).\n"
            "Contexte : {{_contexte}}\n"
            "Réponse : {{_reponse}}\n"
            "Score :"
        ),
    },
    {
        "name": "juge_hallucination",
        "prompt": (
            "Évalue si la réponse contient des informations absentes ou contredites par le contexte de référence.\n"
            "Retourne un score décimal entre 0 et 1 (0 = hallucination totale, 1 = aucune hallucination).\n"
            "Contexte de référence : {{_contexte_ref}}\n"
            "Réponse : {{_reponse}}\n"
            "Score :"
        ),
    },
]

for _p in _PROMPTS:
    _client.create_prompt(
        name=_p["name"],
        prompt=_p["prompt"],
        labels=["production"],
    )
    print(f"✓ Prompt '{_p['name']}' créé avec le label 'production'")

print("\nTous les prompts ont été créés avec succès.")
