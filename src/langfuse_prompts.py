import os
from dotenv import load_dotenv

load_dotenv()

_LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
_LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
_LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Langfuse v3 : observe est dans le module racine
try:
    from langfuse import observe
except ImportError:
    def observe(_func=None, **kwargs):
        if _func is not None:
            return _func
        def _decorator(_f):
            return _f
        return _decorator

_client_langfuse = None

if _LANGFUSE_PUBLIC_KEY and _LANGFUSE_SECRET_KEY:
    try:
        from langfuse import Langfuse
        _client_langfuse = Langfuse(
            public_key=_LANGFUSE_PUBLIC_KEY,
            secret_key=_LANGFUSE_SECRET_KEY,
            host=_LANGFUSE_HOST,
        )
    except Exception:
        _client_langfuse = None

_PROMPTS_FALLBACK = {
    "service_client": (
        "Tu es un assistant service client professionnel et empathique.\n"
        "Réponds en français de façon concise et utile\n"
        "Client : {{_message}}\n"
        "Assistant :"
    ),
    "filtre_politique": (
        "Analyse le message suivant et détermine s'il contient du contenu à caractère politique.\n"
        "Réponds uniquement par OUI ou NON.\n"
        "Message : {{_message}}\n"
        "Réponse :"
    ),
    "juge_coherence": (
        "Évalue si la réponse suivante est cohérente avec l'intention du message utilisateur.\n"
        "Retourne un score décimal entre 0 et 1 (0 = incohérent, 1 = parfaitement cohérent).\n"
        "Message : {{_message}}\n"
        "Réponse : {{_reponse}}\n"
        "Score :"
    ),
    "juge_ton": (
        "Évalue si le ton de la réponse est approprié au contexte attendu (professionnel, empathique).\n"
        "Retourne un score décimal entre 0 et 1 (0 = ton inapproprié, 1 = ton parfait).\n"
        "Contexte : {{_contexte}}\n"
        "Réponse : {{_reponse}}\n"
        "Score :"
    ),
    "juge_hallucination": (
        "Évalue si la réponse contient des informations absentes ou contredites par le contexte de référence.\n"
        "Retourne un score décimal entre 0 et 1 (0 = hallucination totale, 1 = aucune hallucination).\n"
        "Contexte de référence : {{_contexte_ref}}\n"
        "Réponse : {{_reponse}}\n"
        "Score :"
    ),
}


def _compiler_template(_template: str, **kwargs) -> str:
    _resultat = _template
    for _cle, _valeur in kwargs.items():
        _resultat = _resultat.replace("{{" + _cle + "}}", str(_valeur))
    return _resultat


def obtenir_prompt(_nom: str, **kwargs) -> str:
    if _client_langfuse is not None:
        try:
            _prompt_obj = _client_langfuse.get_prompt(_nom)
            return _prompt_obj.compile(**kwargs)
        except Exception:
            pass
    _template = _PROMPTS_FALLBACK.get(_nom, "")
    return _compiler_template(_template, **kwargs)


def obtenir_template(_nom: str) -> str:
    if _client_langfuse is not None:
        try:
            _prompt_obj = _client_langfuse.get_prompt(_nom)
            return _prompt_obj.prompt
        except Exception:
            pass
    return _PROMPTS_FALLBACK.get(_nom, "")
