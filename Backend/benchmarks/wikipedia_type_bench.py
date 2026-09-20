"""
wikipedia_type_bench.py — what does a Wikipedia type lookup actually cost?

The proposal is to stop guessing entity types with spaCy and look them up
instead: ask Wikipedia what a name refers to, so 'Juan Ponce de León' comes back
as a person and 'Saint Augustine, Florida' as a city. The objection to any such
proposal is latency, and the honest answer is that nobody knows until it is
measured ON THE MACHINE THAT WILL RUN IT.

That last part matters here more than usual. This project's own logs show
repeated ConnectTimeoutError against en.wikipedia.org, so the relevant question
is not "how fast is Wikimedia" but "how fast is Wikimedia FROM HERE, and how
often does it fail".

Run from Backend/:

    python -m benchmarks.wikipedia_type_bench
    python -m benchmarks.wikipedia_type_bench --repeat 3

WHAT IT MEASURES
    1. Cold latency per lookup, as a distribution rather than an average — a
       mean hides the timeouts, and the timeouts are the interesting part.
    2. Warm latency, i.e. the cache. Entities repeat across the claims of one
       document, so the second lookup of the same name should cost nothing.
    3. Failure rate, because a type check that blocks the pipeline is worse
       than no type check at all.
    4. The same cost expressed as a fraction of one verification, which is the
       only comparison that decides anything.
"""

import argparse
import json
import statistics
import time
import urllib.parse

import requests

REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# Short, and it must stay short. This lookup is a SANITY CHECK on an answer the
# pipeline already has — it is not on the critical path to a verdict, so it may
# never block one. The retriever's article fetches were observed failing with
# 'connect timeout=None', which lets a hung socket stall a request indefinitely;
# repeating that mistake for a type check would be indefensible.
TIMEOUT_S = 3.0

USER_AGENT = "AutoCitation/0.1 (PoC fact-checker; contact: mehmet17b.b@gmail.com)"

# A mix chosen to include the confusion this exists to resolve: an explorer and
# a city named after a saint, which is exactly the pair spaCy cannot separate.
TITLES = [
    "Juan Ponce de León",          # person  — the wrong answer to catch
    "Saint Augustine, Florida",    # city    — the right one, and PERSON-looking
    "Jamestown, Virginia",         # place   — tagged PERSON by en_core_web_sm
    "Amazon River",
    "Nile",
    "Mount Everest",
    "K2",
    "Bosporus",
    "Marie Curie",
    "Virginia Company",
]


def fetch_description(title: str) -> tuple[str | None, float, str]:
    """
    (description, seconds, status) for one title.

    FAILS OPEN. Every error path returns None rather than raising, because the
    caller's correct response to "I could not determine the type" is to skip the
    check, not to abandon the claim. A type oracle that can veto a verdict by
    being unreachable has made the system less reliable, not more.
    """
    url = REST_SUMMARY + urllib.parse.quote(title.replace(" ", "_"), safe="(),")
    t0 = time.perf_counter()
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return None, elapsed, f"HTTP {r.status_code}"
        data = r.json()
        # 'description' is the short type line ("Spanish explorer", "City in
        # Florida"). 'extract' is the first paragraph — the fallback when a page
        # has no short description.
        desc = data.get("description") or (data.get("extract") or "")[:80]
        return desc or None, elapsed, "ok"
    except requests.Timeout:
        return None, time.perf_counter() - t0, f"timeout >{TIMEOUT_S}s"
    except Exception as e:
        return None, time.perf_counter() - t0, type(e).__name__


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1,
                    help="passes over the title list (cold timings use pass 1)")
    args = ap.parse_args()

    print(f"\n  {len(TITLES)} titles, timeout {TIMEOUT_S}s, {args.repeat} pass(es)\n")
    print(f"  {'title':30} {'seconds':>9}  {'status':12} description")
    print("  " + "-" * 88)

    cold: list[float] = []
    failures = 0
    cache: dict[str, str | None] = {}

    for title in TITLES:
        desc, secs, status = fetch_description(title)
        cold.append(secs)
        cache[title] = desc
        if status != "ok":
            failures += 1
        shown = (desc or "—")[:38]
        print(f"  {title[:30]:30} {secs:8.3f}s  {status:12} {shown}")

    # Warm pass: the cache is a plain dict because entity mentions repeat across
    # the claims of one document, and a second lookup of the same name should
    # cost nothing at all.
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        for title in TITLES:
            _ = cache.get(title)
    warm_total = time.perf_counter() - t0

    print("\n  COLD")
    print(f"    median      {statistics.median(cold):.3f}s")
    print(f"    p90         {percentile(cold, 90):.3f}s")
    print(f"    max         {max(cold):.3f}s")
    print(f"    total       {sum(cold):.3f}s for {len(TITLES)} lookups")
    print(f"    failures    {failures}/{len(TITLES)}")

    print(f"\n  WARM (cached)")
    print(f"    total       {warm_total * 1000:.3f}ms for "
          f"{len(TITLES) * args.repeat} lookups")

    # The comparison that actually decides it. Observed verification times in
    # this project's logs run 70-636s; 150s is a representative middle.
    ref = 150.0
    median = statistics.median(cold)
    print(f"\n  IN CONTEXT")
    print(f"    one lookup          {median:.3f}s")
    print(f"    one verification    ~{ref:.0f}s  (observed range 70-636s)")
    print(f"    lookup as % of one verification   {100 * median / ref:.3f}%")
    print(f"\n    A superlative claim needs ONE lookup, to check the answer the")
    print(f"    program already produced. At the median above that is "
          f"{median:.3f}s added to a\n    request that takes ~500s.")

    if failures:
        print(f"\n  NOTE: {failures} lookup(s) failed. That is the number that")
        print(f"  matters — the check must fail open, so a failed lookup means")
        print(f"  'type unknown, let the answer through', never a lost claim.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
