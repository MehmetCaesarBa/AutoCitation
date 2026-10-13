"""
wikidata.py — structured values for the comparison program.

One question, asked of Wikidata's SPARQL endpoint: what is the value of this
dimension for the entity behind this English Wikipedia article, provided the
entity is the kind of thing the claim ranks?

    value("K2", "mountain", "height")  ->  WikidataValue(qid='Q43512', value=8611.0, ...)

WHY VALUES AND NOT RANKINGS. The obvious use of Wikidata is to rank a class
directly ("sort all mountains by elevation, take the second"). Measured: that
query took 19.8 s for mountains alone and timed out at 60 s once 'summit' was
added as a second class. A lookup bound to one entity takes 0.4-2 s. Rankings
also inherit every gap in class membership — a mountain typed only as
'massif' is silently absent — whereas a pairwise comparison of two named
entities needs nothing but their two values.

WHY THE SITELINK AND NOT THE LABEL. Matching the English label "K2" returned
two mountains (Q43512 at 8,611 m and another 'K2' at 3,253 m); "Russia"
returned Russia and the Russian Empire. The English Wikipedia article title is
unique by construction, and it is the same identifier the retriever already
works in.

THE CLASS IS A GUARD, NOT A DETAIL. A value is returned only when the entity is
an instance (or subclass-instance) of a class mapped for the claim's category.
"Is Mount Everest higher than the Burj Khalifa?" is not a question this module
should answer just because both have a height.

FAILS QUIET. Any network error, timeout, unmapped category or dimension, or
ambiguous result returns None, and the caller proceeds as if Wikidata did not
exist. This module can add a confirmation; it must never be the reason a claim
fails.
"""

import functools
from dataclasses import dataclass

import requests

ENDPOINT = "https://query.wikidata.org/sparql"
TIMEOUT_S = 20

# Wikimedia asks every client to identify itself; anonymous agents are throttled.
_HEADERS = {
    "User-Agent": "AutoCitation/0.1 (fact-checking research prototype)",
    "Accept": "application/sparql-results+json",
}

# Category noun (ComparisonFrame.type_word) -> Wikidata classes. Deliberately
# small, as program._DIMENSIONS is: each entry is a claim about what a word
# means, and a wrong one compares things that are not comparable.
CATEGORY_CLASSES = {
    "mountain":   ("Q8502",),
    "peak":       ("Q8502", "Q207326"),       # mountain, summit
    "river":      ("Q4022",),
    "lake":       ("Q23397",),
    "country":    ("Q6256", "Q3624078"),      # country, sovereign state
    "city":       ("Q515",),
    "island":     ("Q23442",),
    "building":   ("Q41176",),
    "skyscraper": ("Q11303",),
    "volcano":    ("Q8072",),
    "waterfall":  ("Q34038",),
}

# (category, dimension) -> property. Keyed by both because the same word means
# different properties for different things: a mountain's 'height' is its
# elevation above sea level (P2044), a building's is its structural height
# (P2048). Combinations not listed are not answered.
PROPERTIES = {
    **{(c, d): "P2044" for c in ("mountain", "peak", "volcano")
       for d in ("height", "elevation", "altitude")},
    ("river", "length"): "P2043",
    ("lake", "depth"): "P4511",
    ("lake", "area"): "P2046",
    ("country", "area"): "P2046",
    ("country", "population"): "P1082",
    ("city", "area"): "P2046",
    ("city", "population"): "P1082",
    ("island", "area"): "P2046",
    ("island", "population"): "P1082",
    ("building", "height"): "P2048",
    ("skyscraper", "height"): "P2048",
    ("waterfall", "height"): "P2048",
}

# Unit-less properties have no normalised value node, so they are read from the
# plain statement value instead. Measured: India and China returned nothing for
# P1082 through psn:.
_UNITLESS = {"P1082"}

_QUERY = """
SELECT ?item (MAX(?v) AS ?value) WHERE {{
  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name "{title}"@en .
  VALUES ?cls {{ {classes} }}
  ?item wdt:P31/wdt:P279* ?cls .
  {value_path}
}} GROUP BY ?item
"""


@dataclass(frozen=True)
class WikidataValue:
    """One entity's value, in SI base units (metres, square metres, a count)."""
    title: str
    qid: str
    value: float

    @property
    def url(self) -> str:
        return f"https://www.wikidata.org/wiki/{self.qid}"


def supports(category: str | None, dimension: str | None) -> bool:
    """Can this (category, dimension) be looked up at all? No network."""
    return (category or "").lower() in CATEGORY_CLASSES and \
        ((category or "").lower(), dimension) in PROPERTIES


@functools.lru_cache(maxsize=512)
def value(title: str, category: str, dimension: str) -> WikidataValue | None:
    """
    The value of `dimension` for the entity whose English Wikipedia article is
    `title`, if that entity is a `category`. None on any doubt.
    """
    category = category.lower()
    if not supports(category, dimension) or not title or '"' in title or "\\" in title:
        return None

    pid = PROPERTIES[(category, dimension)]
    value_path = (f"?item wdt:{pid} ?v ." if pid in _UNITLESS
                  else f"?item p:{pid}/psn:{pid}/wikibase:quantityAmount ?v .")
    query = _QUERY.format(
        title=title,
        classes=" ".join(f"wd:{q}" for q in CATEGORY_CLASSES[category]),
        value_path=value_path,
    )

    try:
        response = requests.get(ENDPOINT, params={"query": query},
                                headers=_HEADERS, timeout=TIMEOUT_S)
        response.raise_for_status()
        rows = response.json()["results"]["bindings"]
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[Wikidata] '{title}' {dimension}: unavailable ({type(e).__name__}).")
        return None

    if len(rows) != 1:
        # Zero: not a `category`, no such article, or no value. More than one
        # cannot happen for a sitelink, and is not guessed at if it does.
        print(f"[Wikidata] '{title}' {dimension}: {len(rows)} match(es) as a "
              f"{category} — no value.")
        return None

    qid = rows[0]["item"]["value"].rsplit("/", 1)[-1]
    try:
        number = float(rows[0]["value"]["value"])
    except (KeyError, ValueError):
        return None
    print(f"[Wikidata] '{title}' ({qid}) {dimension} = {number:g}")
    return WikidataValue(title=title, qid=qid, value=number)
