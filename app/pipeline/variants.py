"""Cluster spelling variants of categorical values ('T-Shirt', 'tshirt', 't-shirts', 'Tee').
Deterministic and explainable: every mapping carries a method and a confidence."""
import re


def stem(k: str) -> str:
    if k.endswith("ies") and len(k) > 4:
        return k[:-3] + "y"
    if re.search(r"(ch|sh|x|ss)es$", k):
        return k[:-2]
    if k.endswith("s") and not k.endswith("ss") and len(k) > 3:
        return k[:-1]
    return k


def norm_key(v: str) -> str:
    return stem(re.sub(r"[^a-z0-9]", "", v.lower()))


def _pref(v: str, count: int):
    """Canonical choice: prefer 'West' over 'WEST' and 'west'; IDs like 'ST01' are exempt (not purely alphabetic)."""
    if v.isalpha() and v.isupper() and len(v) > 3:
        style = 2
    elif v.isalpha() and v.islower():
        style = 1
    else:
        style = 0
    return (style, -count, v)


def cluster_variants(items, aliases=None):
    """items: [(stripped_value, count)] with distinct values. Returns clusters with >=2 distinct values."""
    aliases = aliases or {}
    groups = {}
    for v, c in items:
        k = norm_key(v)
        groups.setdefault(aliases.get(k, k), []).append((v, c, k))
    clusters = []
    for ck in sorted(groups):
        members = groups[ck]
        if len(members) < 2:
            continue
        base = [(v, c) for v, c, k in members if k == ck] or [(v, c) for v, c, _ in members]
        canonical = sorted(base, key=lambda vc: _pref(*vc))[0][0]
        canon_plain = " ".join(canonical.lower().split())
        variants, mapping = [], {}
        for v, c, k in sorted(members, key=lambda m: (-m[1], m[0])):
            if " ".join(v.lower().split()) == canon_plain:
                method, conf = "case_whitespace", 1.0
            elif k == ck:
                method, conf = "normalized_key", 0.95
            else:
                method, conf = "alias_table", 0.8
            variants.append({"value": v, "count": c, "method": method, "confidence": conf})
            if v != canonical:
                mapping[v] = canonical
        clusters.append({"key": ck, "canonical": canonical, "total": sum(m[1] for m in members),
                         "variants": variants, "mapping": mapping,
                         "needs_review": any(x["method"] == "alias_table" for x in variants)})
    return clusters
