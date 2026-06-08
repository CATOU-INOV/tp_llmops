import re
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tests.dataset import (
    _cas_service_client,
    _cas_hors_sujet,
    _cas_contenu_sensible,
    _prompt_juge_coherence,
    _prompt_juge_ton,
    _prompt_juge_hallucination,
)
from src.prompts import (
    construire_prompt_service_client,
    repondre_service_client,
    filtrer_contenu_politique,
)

_NOM_MODELE = "HuggingFaceTB/SmolLM2-135M"

_tok_juge = None
_mod_juge = None


def _obtenir_modele_juge():
    global _tok_juge, _mod_juge
    if _tok_juge is None:
        _tok_juge = AutoTokenizer.from_pretrained(_NOM_MODELE)
        if _tok_juge.pad_token is None:
            _tok_juge.pad_token = _tok_juge.eos_token
        _mod_juge = AutoModelForCausalLM.from_pretrained(_NOM_MODELE)
    return _tok_juge, _mod_juge


# Exercice 8 — LLM juge
def _appeler_juge(_juge, _prompt_formate: str) -> float:
    """
    Appelle le modele avec un prompt juge deja formate.
    Retourne le score numerique extrait de la reponse
    (float entre 0 et 1).
    """
    _tok, _mod = _obtenir_modele_juge()
    _entrees = _tok(_prompt_formate, return_tensors="pt")
    _nb_tokens_entree = _entrees["input_ids"].shape[1]

    with torch.no_grad():
        _sorties = _mod.generate(
            **_entrees,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=_tok.eos_token_id,
        )

    _nouveaux_tokens = _sorties[0][_nb_tokens_entree:]
    _texte_genere = _tok.decode(_nouveaux_tokens, skip_special_tokens=True)

    _match = re.search(r'\d+\.\d+|\d+', _texte_genere)
    if _match is None:
        return 0.0

    _score = float(_match.group())
    return min(1.0, max(0.0, _score))


# Exercice 6 — Tests sur les erreurs levées
def test_erreur_message_vide():
    with pytest.raises(ValueError, match="non vide"):
        construire_prompt_service_client("")


def test_erreur_message_trop_long():
    with pytest.raises(ValueError, match="trop long"):
        construire_prompt_service_client("x" * 1001)


# Exercice 10 — Améliorer la couverture : gardes de construire_prompt_filtre_politique
def test_erreur_filtre_message_vide():
    from src.prompts import construire_prompt_filtre_politique
    with pytest.raises(ValueError, match="non vide"):
        construire_prompt_filtre_politique("")


def test_erreur_filtre_message_trop_long():
    from src.prompts import construire_prompt_filtre_politique
    with pytest.raises(ValueError, match="trop long"):
        construire_prompt_filtre_politique("x" * 1001)


# Exercice 7 — Tests paramétrés
@pytest.mark.xfail(
    reason="SmolLM2-135M trop petit pour suivre les instructions : répète le message au lieu de répondre"
)
@pytest.mark.parametrize("_cas", _cas_service_client)
def test_reponse_contient_mot_cle(_cas):
    _reponse = repondre_service_client(_cas["_entree"])
    assert _cas["_attendu"].lower() in _reponse.lower()


# Exercice 7 — Skip conditionnel (GPU)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Ce test necessite un GPU")
def test_inference_sur_gpu():
    _tok = AutoTokenizer.from_pretrained(_NOM_MODELE)
    _mod = AutoModelForCausalLM.from_pretrained(_NOM_MODELE).to("cuda")
    _entrees = _tok("Bonjour", return_tensors="pt")
    _entrees = {_k: _v.to("cuda") for _k, _v in _entrees.items()}
    with torch.no_grad():
        _sorties = _mod.generate(**_entrees, max_new_tokens=5, pad_token_id=_tok.eos_token_id)
    assert _sorties.shape[1] > 0


# Exercice 9 — Tests sémantiques
@pytest.mark.parametrize("_cas", _cas_service_client)
def test_service_client_coherent(_cas):
    _reponse = repondre_service_client(_cas["_entree"])
    _prompt_juge = _prompt_juge_coherence.format(
        _message=_cas["_entree"],
        _reponse=_reponse,
    )
    _score = _appeler_juge("juge_coherence", _prompt_juge)
    assert _score > 0.5


@pytest.mark.xfail(
    reason="SmolLM2-135M réécrit la consigne de scoring ('0 à 1...') au lieu de produire un score"
)
@pytest.mark.parametrize("_cas", _cas_service_client)
def test_ton_service_client_adapte(_cas):
    _reponse = repondre_service_client(_cas["_entree"])
    _prompt_juge = _prompt_juge_ton.format(
        _contexte=_cas["_contexte"],
        _reponse=_reponse,
    )
    _score = _appeler_juge("juge_ton", _prompt_juge)
    assert _score > 0.5


@pytest.mark.xfail(
    reason="SmolLM2-135M ne comprend pas la consigne OUI/NON du filtre politique"
)
def test_filtre_politique_coherent():
    _cas = _cas_contenu_sensible[0]
    _reponse = filtrer_contenu_politique(_cas["_entree"])
    _prompt_juge = _prompt_juge_coherence.format(
        _message=_cas["_entree"],
        _reponse=_reponse,
    )
    _score = _appeler_juge("juge_coherence", _prompt_juge)
    assert _score > 0.5


def test_hors_sujet_score_bas():
    _cas = _cas_hors_sujet[0]
    _reponse = repondre_service_client(_cas["_entree"])
    _prompt_juge = _prompt_juge_coherence.format(
        _message=_cas["_entree"],
        _reponse=_reponse,
    )
    _score = _appeler_juge("juge_coherence", _prompt_juge)
    assert _score < 0.5


def test_hallucination_score_eleve():
    _contexte_ref = "La commande 4521 a ete expediee le 3 juin et arrivera le 7 juin."
    _message = "Je voudrais annuler ma commande numero 4521."
    _reponse = repondre_service_client(_message)
    _prompt_juge = _prompt_juge_hallucination.format(
        _contexte_ref=_contexte_ref,
        _reponse=_reponse,
    )
    _score = _appeler_juge("juge_hallucination", _prompt_juge)
    assert _score > 0.5
