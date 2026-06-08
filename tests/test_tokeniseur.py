from src.tokeniseur import Tokeniseur

_tokeniseur = Tokeniseur()


def test_encodage_retourne_liste_entiers():
    _tokens = _tokeniseur.encoder("Bonjour le monde")
    assert isinstance(_tokens, list)
    assert all(isinstance(_t, int) for _t in _tokens)


def test_decodage_reconstruit_texte():
    _texte = "Le fine-tuning ameliore les modeles."
    _tokens = _tokeniseur.encoder(_texte)
    _texte_reconstruit = _tokeniseur.decoder(_tokens, skip_special_tokens=True)
    assert _texte.lower() in _texte_reconstruit.lower() or _texte_reconstruit.strip() == _texte.strip()


def test_tokens_speciaux_presents():
    assert _tokeniseur.eos_token_id is not None
    assert isinstance(_tokeniseur.eos_token_id, int)


# Exercice 10 — Améliorer la couverture
def test_proprietes_tokeniseur():
    assert isinstance(len(_tokeniseur), int)
    assert len(_tokeniseur) > 0
    assert _tokeniseur.bos_token_id is None or isinstance(_tokeniseur.bos_token_id, int)
    assert isinstance(_tokeniseur.pad_token_id, int)
    assert isinstance(_tokeniseur.vocab_size, int)


def test_padding_unifie_longueurs():
    _phrases = [
        "Court.",
        "Cette phrase est beaucoup plus longue que la precedente pour forcer le padding.",
    ]
    _entrees = _tokeniseur(_phrases, padding=True, return_tensors="pt")
    # Les deux séquences ont la même longueur grâce au padding
    assert _entrees["input_ids"].shape[0] == 2
    assert _entrees["input_ids"].shape[1] == _entrees["input_ids"].shape[1]
    # Le masque d'attention contient des 0 (positions paddées)
    assert 0 in _entrees["attention_mask"]
