import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.langfuse_prompts import observe

_NOM_MODELE = "HuggingFaceTB/SmolLM2-135M"

_tokeniseur = None
_modele = None


def _charger():
    global _tokeniseur, _modele
    if _tokeniseur is None:
        _tokeniseur = AutoTokenizer.from_pretrained(_NOM_MODELE)
        if _tokeniseur.pad_token is None:
            _tokeniseur.pad_token = _tokeniseur.eos_token
        _modele = AutoModelForCausalLM.from_pretrained(_NOM_MODELE)
    return _tokeniseur, _modele


@observe()
def generer(_prompt: str, _nb_tokens_max: int = 50) -> str:
    _tok, _mod = _charger()
    _entrees = _tok(_prompt, return_tensors="pt")
    _nb_tokens_entree = _entrees["input_ids"].shape[1]
    with torch.no_grad():
        _sorties = _mod.generate(
            **_entrees,
            max_new_tokens=_nb_tokens_max,
            do_sample=False,
            pad_token_id=_tok.eos_token_id,
        )
    _nouveaux_tokens = _sorties[0][_nb_tokens_entree:]
    return _tok.decode(_nouveaux_tokens, skip_special_tokens=True)


def obtenir_logits(_prompt: str):
    _tok, _mod = _charger()
    _entrees = _tok(_prompt, return_tensors="pt")
    with torch.no_grad():
        _sorties = _mod(**_entrees)
    return _sorties.logits
