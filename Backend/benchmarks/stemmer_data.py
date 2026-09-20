"""
Test data for stemmer_bench.py — three groups, 50 items each.

WHY THREE GROUPS. "Which normaliser is best?" has no answer, because the two
things a normaliser can do wrong pull in opposite directions:

    too weak      inflectional variants stay apart, so a FAITHFUL rewrite
                  ('separates' -> 'separated') looks like foreign vocabulary
                  and the faithfulness gate rejects it. Cost: a wasted retry.

    too strong    unrelated words collapse together, so material the model
                  INVENTED collides with a source word and passes the gate.
                  Cost: an unfaithful claim reaches verification.

Measuring only one of those picks the wrong winner every time — the Lancaster
stemmer scores near-perfectly on group A and is unusable. Group C exists
because fact-checking prose is mostly names, and names are the case every pure
string stemmer gets wrong by construction.

The groups are deliberately independent of the project's inputs. None of these
words is drawn from a test paragraph, so a normaliser cannot score well here by
happening to suit one example.
"""

# ─────────────────────────────────────────────────────────────────────────────
# GROUP A — MUST COLLAPSE
# Inflectional variants of one lemma. A normaliser that keeps these apart makes
# the faithfulness gate reject rewrites that are perfectly faithful.
#
# Irregulars are included on purpose (held/hold, built/build, went/go, wrote/
# write). No suffix-stripping algorithm can reach them at all — they are the
# clearest demonstration of what a lexicon-backed lemmatiser buys.
# ─────────────────────────────────────────────────────────────────────────────
COLLAPSE_PAIRS = [
    # regular verbs — third person / past / participle
    ("separates", "separated"), ("forms", "formed"), ("discharges", "discharged"),
    ("founded", "founds"), ("opened", "opens"), ("carries", "carried"),
    ("flows", "flowed"), ("connects", "connected"), ("establishes", "established"),
    ("considers", "considered"), ("measures", "measured"), ("publishes", "published"),
    ("concludes", "concluded"), ("settles", "settled"), ("celebrates", "celebrated"),
    ("acknowledges", "acknowledged"), ("ranks", "ranked"), ("lists", "listed"),
    ("names", "named"), ("crosses", "crossed"), ("passes", "passed"),
    ("reaches", "reached"), ("stretches", "stretched"), ("borders", "bordered"),
    ("divides", "divided"), ("empties", "emptied"), ("supplies", "supplied"),
    ("occupies", "occupied"), ("identifies", "identified"), ("classifies", "classified"),

    # irregular verbs — unreachable by suffix stripping
    ("held", "holds"), ("built", "builds"), ("wrote", "writes"), ("went", "goes"),
    ("took", "takes"), ("made", "makes"), ("began", "begins"), ("grew", "grows"),
    ("rose", "rises"), ("found", "finds"),

    # noun plurals, including -y -> -ies
    ("rivers", "river"), ("settlements", "settlement"), ("countries", "country"),
    ("cities", "city"), ("bodies", "body"), ("museums", "museum"),
    ("paintings", "painting"), ("basins", "basin"), ("tributaries", "tributary"),
    ("boundaries", "boundary"),
]

# ─────────────────────────────────────────────────────────────────────────────
# GROUP B — MUST STAY DISTINCT
# Unrelated words that an aggressive normaliser fuses. Every collapse here is a
# hole in the faithfulness gate: invented vocabulary that matches the source by
# accident.
#
# Two failure families are represented deliberately.
#   -ing/-ed/-es as false suffixes:  'morning' is not an inflected 'morn',
#                                    'hundred' is not an inflected 'hundr'.
#   -ion/-ine/-oon endings:          Porter-family rules strip these into stems
#                                    that collide with common short words.
# ─────────────────────────────────────────────────────────────────────────────
DISTINCT_PAIRS = [
    # false -ing
    ("morning", "morn"), ("evening", "even"), ("ceiling", "ceil"),
    ("stocking", "stock"), ("herring", "her"), ("pudding", "pud"),
    ("shilling", "shill"), ("nothing", "noth"), ("during", "dur"),
    ("string", "str"),

    # false -ed / -es
    ("hundred", "hundr"), ("sacred", "sacr"), ("series", "seri"),
    ("species", "speci"), ("rabies", "rabi"), ("scabies", "scabi"),

    # -ion / -ine / -oon collisions
    ("university", "universe"), ("organization", "organ"), ("police", "policy"),
    ("communism", "community"), ("doctrine", "doctor"), ("particle", "particular"),
    ("marine", "mar"), ("experiment", "experience"), ("relativity", "relative"),
    ("civilian", "civil"), ("mission", "miss"), ("passion", "pass"),
    ("station", "state"), ("pension", "pen"), ("mansion", "man"),
    ("legion", "leg"), ("million", "mill"), ("billion", "bill"),
    ("gallon", "gall"), ("balloon", "ball"), ("cartoon", "cart"),
    ("lagoon", "lag"), ("harpoon", "harp"), ("typhoon", "type"),
    ("festoon", "fest"), ("bassoon", "bass"), ("platoon", "plate"),
    ("corner", "corn"), ("banner", "ban"), ("manner", "man"),

    # short-word collisions
    ("bases", "basis"), ("axes", "axis"), ("lenses", "lens"),
    ("business", "busy"),
]

# ─────────────────────────────────────────────────────────────────────────────
# GROUP C — MUST SURVIVE UNCHANGED
# Proper nouns, plus singular nouns that merely END in -s. Both are wrecked by
# terminal-s stripping, and both are everywhere in encyclopedia prose.
#
# This is the group that decides the benchmark. A mangled name stops matching
# its own mention in the evidence, and every claim this pipeline handles is
# about a named entity.
# ─────────────────────────────────────────────────────────────────────────────
PRESERVE_WORDS = [
    # place names ending in -s
    "Paris", "Athens", "Wales", "Naples", "Texas", "Kansas", "Arkansas",
    "Illinois", "Massachusetts", "Cyprus", "Belarus", "Honduras", "Bahamas",
    "Philippines", "Netherlands", "Barbados", "Maldives", "Seychelles",
    "Comoros", "Andes", "Alps", "Himalayas", "Pyrenees", "Thames", "Ganges",
    "Euphrates", "Tigris", "Indus", "Brussels",

    # personal names ending in -s
    "Charles", "James", "Thomas", "Nicholas", "Lucas", "Marcus", "Julius",
    "Augustus", "Erasmus", "Copernicus", "Archimedes", "Socrates",
    "Hippocrates", "Euripides", "Sophocles", "Pericles",

    # singular nouns ending in -s (not plurals)
    "news", "physics", "campus", "census", "chassis",
]

GROUPS = {
    "A_collapse": COLLAPSE_PAIRS,
    "B_distinct": DISTINCT_PAIRS,
    "C_preserve": PRESERVE_WORDS,
}
