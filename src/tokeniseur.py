from transformers import AutoTokenizer

_NOM_MODELE = "HuggingFaceTB/SmolLM2-135M"


class Tokeniseur:
    def __init__(self, _nom_modele: str = _NOM_MODELE):
        self._tok = AutoTokenizer.from_pretrained(_nom_modele)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

    def encoder(self, _texte: str, **kwargs) -> list:
        return self._tok.encode(_texte, **kwargs)

    def decoder(self, _tokens, **kwargs) -> str:
        return self._tok.decode(_tokens, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._tok)

    @property
    def eos_token_id(self):
        return self._tok.eos_token_id

    @property
    def bos_token_id(self):
        return self._tok.bos_token_id

    @property
    def pad_token_id(self):
        return self._tok.pad_token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size
