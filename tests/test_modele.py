import pytest
import torch
from src.modele import generer, obtenir_logits, _charger

# Chargement unique du modèle pour tout le module
_tok, _mod = _charger()


def test_forme_logits():
    _logits = obtenir_logits("Bonjour")
    # Forme attendue : (1, nb_tokens, vocab_size)
    assert _logits.ndim == 3
    assert _logits.shape[0] == 1
    assert _logits.shape[2] == _mod.config.vocab_size


def test_generation_produit_du_texte():
    _prompt = "Il etait une fois"
    _texte = generer(_prompt, _nb_tokens_max=10)
    # Le texte généré (nouveaux tokens uniquement) doit être non vide
    assert len(_texte) > 0


@pytest.mark.parametrize("_nb_tokens_max", [10, 30, 50])
def test_longueur_generation_respectee(_nb_tokens_max):
    _entrees = _tok("Les reseaux de neurones sont", return_tensors="pt")
    _nb_tokens_entree = _entrees["input_ids"].shape[1]
    with torch.no_grad():
        _sorties = _mod.generate(
            **_entrees,
            max_new_tokens=_nb_tokens_max,
            do_sample=False,
            pad_token_id=_tok.eos_token_id,
        )
    _nb_nouveaux = _sorties.shape[1] - _nb_tokens_entree
    assert _nb_nouveaux <= _nb_tokens_max
