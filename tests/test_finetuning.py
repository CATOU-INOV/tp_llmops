import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_NOM_MODELE = "HuggingFaceTB/SmolLM2-135M"


def test_loss_diminue_apres_un_pas():
    """
    Verifie que la loss est strictement inferieure
    apres un pas de gradient sur un exemple connu.
    Propriete minimale d'un entrainement fonctionnel.
    """
    _tok = AutoTokenizer.from_pretrained(_NOM_MODELE)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _mod = AutoModelForCausalLM.from_pretrained(_NOM_MODELE)
    _optimiseur = torch.optim.AdamW(_mod.parameters(), lr=1e-4)

    _texte = "Le fine-tuning ameliore les modeles."
    _entrees = _tok(_texte, return_tensors="pt")
    _labels = _entrees["input_ids"].clone()

    with torch.no_grad():
        _loss_avant = _mod(**_entrees, labels=_labels).loss.item()

    _optimiseur.zero_grad()
    _mod(**_entrees, labels=_labels).loss.backward()
    _optimiseur.step()

    with torch.no_grad():
        _loss_apres = _mod(**_entrees, labels=_labels).loss.item()

    assert _loss_apres < _loss_avant


def test_poids_modifies_apres_gradient():
    """
    Verifie qu'au moins un parametre a change apres
    un pas d'optimisation.
    """
    _tok = AutoTokenizer.from_pretrained(_NOM_MODELE)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _mod = AutoModelForCausalLM.from_pretrained(_NOM_MODELE)
    _optimiseur = torch.optim.AdamW(_mod.parameters(), lr=1e-4)

    _poids_avant = {_nom: _p.clone() for _nom, _p in _mod.named_parameters()}

    _texte = "Le fine-tuning ameliore les modeles."
    _entrees = _tok(_texte, return_tensors="pt")
    _labels = _entrees["input_ids"].clone()

    _optimiseur.zero_grad()
    _mod(**_entrees, labels=_labels).loss.backward()
    _optimiseur.step()

    _modifie = any(
        not torch.equal(_poids_avant[_nom], _p)
        for _nom, _p in _mod.named_parameters()
    )
    assert _modifie
