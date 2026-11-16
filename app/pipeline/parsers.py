"""Pure, deterministic value parsers shared by the profiler (and later the cleaner).

parse_number: handles '$1,299.00', 'Rs 499', '10%'. Percent values are returned as fractions (10% -> 0.1).
parse_date:   handles ISO, dd/mm/yyyy, mm-dd-yyyy, '14 Mar 2025'. Day/month-ambiguous values
              (e.g. 03/04/2025) are FLAGGED, never silently guessed.
"""
import re
from datetime import date, datetime

NULL_TOKENS = {"n/a", "na", "null", "none", "nan", "-", "--", "?", "nil", "#n/a"}

_CURRENCY_PREFIX = re.compile(r"^(rs\.?|inr|usd|\$|₹)\s*", re.I)
_NUM_CORE = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$|^[-+]?\.\d+$")


def parse_number(value: str):
    """Return (float_value, symbolic) or None. symbolic=True if currency/percent/thousand-comma was present."""
    t = value.strip().lower()
    symbolic = False
    stripped = _CURRENCY_PREFIX.sub("", t)
    if stripped != t:
        symbolic, t = True, stripped
    pct = t.endswith("%")
    if pct:
        symbolic, t = True, t[:-1].strip()
    if not _NUM_CORE.match(t):
        return None
    if "," in t:
        symbolic = True
    v = float(t.replace(",", ""))
    return (v / 100.0 if pct else v), symbolic


_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMDATE = re.compile(r"^(\d{1,2})([/-])(\d{1,2})\2(\d{4})$")
_TEXTDATE = re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}$")


def _valid(y, m, d):
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(value: str):
    """Return (date, format_name, ambiguous) or None if unparseable.
    For ambiguous values the returned date is the day-first reading and must not be trusted."""
    t = value.strip()
    if _ISO.match(t):
        d = _valid(int(t[:4]), int(t[5:7]), int(t[8:10]))
        return (d, "iso", False) if d else None
    m = _NUMDATE.match(t)
    if m:
        a, sep, b, y = int(m[1]), m[2], int(m[3]), int(m[4])
        kind = "slash" if sep == "/" else "dash"
        if a > 12:
            d, fmt, amb = _valid(y, b, a), f"{kind}_dmy", False
        elif b > 12:
            d, fmt, amb = _valid(y, a, b), f"{kind}_mdy", False
        elif a == b:
            d, fmt, amb = _valid(y, b, a), f"{kind}_dmy", False
        else:
            d, fmt, amb = _valid(y, b, a), f"{kind}_ambiguous", True
        return (d, fmt, amb) if d else None
    if _TEXTDATE.match(t):
        t2 = re.sub(r"[.,]", "", t)
        for f in ("%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(t2, f).date(), "text_dmy", False
            except ValueError:
                continue
    return None
NUMDATE = _NUMDATE  # public alias used by issue detection
