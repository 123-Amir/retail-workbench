"""Curated retail alias tables (code-owned, auditable). Keys are *normalised stems*
(lowercase, alphanumerics only, simple plural stripped) - see app/pipeline/variants.py.
Values are the stem of the canonical form. Matches from these tables get confidence 0.8
and are flagged for review, because aliasing is a semantic judgement."""
import re

CATEGORY_ALIASES = {"tee": "tshirt", "denim": "jean", "trainer": "sneaker"}
BRAND_ALIASES = {"nikeinc": "nike", "handm": "hm"}
REGION_ALIASES = {"w": "west", "e": "east", "n": "north", "s": "south"}


def alias_table_for(column: str) -> dict:
    c = column.lower()
    if re.search(r"region", c):
        return REGION_ALIASES
    if re.search(r"brand", c):
        return BRAND_ALIASES
    if re.search(r"category|type", c):
        return CATEGORY_ALIASES
    return {}
