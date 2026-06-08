import re
from src.langfuse_prompts import obtenir_template

# 1. Phrases pour tester le tokeniseur
_phrases_simples = [
    "Le modele apprend des representations contextuelles.",
    "La tokenisation decoupe le texte en sous-unites.",
    "Un token peut correspondre a un mot ou a une partie de mot.",
]

_phrases_accentuees = [
    "Ca fonctionne meme avec des accents : e, e, e, i, o, u.",
    "L'apostrophe et les guillemets << francais >> posent parfois probleme.",
]

# 2. Cas de test structurés (entrée + sortie attendue)
_cas_service_client = [
    {
        "_entree": "Je voudrais annuler ma commande numero 4521.",
        "_attendu": "annulation",
        "_contexte": "service client professionnel et empathique en francais",
    },
    {
        "_entree": "Mon colis est arrive endommage, que faire ?",
        "_attendu": "remboursement",
        "_contexte": "service client professionnel et empathique en francais",
    },
]

_cas_hors_sujet = [
    {
        "_entree": "Quelle est la recette de la quiche lorraine ?",
        "_attendu": None,
    },
]

_cas_contenu_sensible = [
    {
        "_entree": "Je veux acheter une affiche de campagne pour le candidat X.",
        "_attendu": "politique",
    },
]

# 3. Prompts pour tester la génération brute
_prompts_generation = [
    "Il etait une fois",
    "Les reseaux de neurones sont",
    "En intelligence artificielle,",
]


# 4. Prompts juge - récupérés depuis Langfuse, convertis pour str.format()
def _convertir_placeholders(_template: str) -> str:
    """Convertit la syntaxe Langfuse {{var}} en syntaxe Python {var}."""
    return re.sub(r"\{\{(\w+)\}\}", r"{\1}", _template)


_prompt_juge_coherence = _convertir_placeholders(obtenir_template("juge_coherence"))
_prompt_juge_ton = _convertir_placeholders(obtenir_template("juge_ton"))
_prompt_juge_hallucination = _convertir_placeholders(obtenir_template("juge_hallucination"))
