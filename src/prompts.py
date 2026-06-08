from src.langfuse_prompts import observe, obtenir_prompt
from src.modele import generer

_modele_instance = None


@observe()
def _appeler_modele(_prompt: str) -> str:
    return generer(_prompt)


@observe()
def construire_prompt_service_client(_message: str) -> str:
    if not _message or not _message.strip():
        raise ValueError("Le message ne peut pas être vide (non vide)")
    if len(_message) > 1000:
        raise ValueError("Le message est trop long (trop long)")
    return obtenir_prompt("service_client", _message=_message)


@observe()
def repondre_service_client(_message: str) -> str:
    _prompt = construire_prompt_service_client(_message)
    return _appeler_modele(_prompt)


@observe()
def construire_prompt_filtre_politique(_message: str) -> str:
    if not _message or not _message.strip():
        raise ValueError("Le message ne peut pas être vide (non vide)")
    if len(_message) > 1000:
        raise ValueError("Le message est trop long (trop long)")
    return obtenir_prompt("filtre_politique", _message=_message)


@observe()
def filtrer_contenu_politique(_message: str) -> str:
    _prompt = construire_prompt_filtre_politique(_message)
    return _appeler_modele(_prompt)
