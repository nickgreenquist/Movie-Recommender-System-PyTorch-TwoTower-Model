"""
Lightweight persona-grounding helper for the popularity-correction blog post.

Loads ONLY serving/feature_store.pt (+ data/corpus_raw_rating_counts.npy) — no model,
no 12GB feature parquets — so persona-hunting subagents can validate seed titles, look
up popularity, browse the genome-tag vocabulary, and preview the anchor movies a genome
tag injects, all in well under a second per call.

Title format matters: the user tower silently drops any seed whose exact title string
isn't in the corpus, so every persona seed must be confirmed here first.

CLI
    python analysis/persona_tools.py check  "Django (1966)" "Fake Movie (1999)" ...
    python analysis/persona_tools.py search "western"          # title substring (case-insensitive)
    python analysis/persona_tools.py tags   "western"          # genome-tag substring
    python analysis/persona_tools.py anchors "spaghetti western" ["another tag" ...]
    python analysis/persona_tools.py persona "Spaghetti Western fan" "spaghetti western" \
        --liked "Django (1966)" "Fistful of Dollars, A (1964)"   # full preview of one persona

`check`/`persona` print each title's raw MovieLens-32M rating count (the popularity badge
in the poster wall) so you can see at a glance whether a seed is a popular "gateway" title
or a long-tail pick.
"""
import argparse
import sys

import numpy as np
import torch

ANCHORS_PER_TAG = 5   # mirror src/evaluate.py:ANCHORS_PER_TAG


def _load():
    """Return (title_to_movieId, movieId_to_title, mid_to_count, genome helpers)."""
    fs     = torch.load('serving/feature_store.pt', map_location='cpu', weights_only=False)
    counts = np.load('data/corpus_raw_rating_counts.npy')
    top    = [int(m) for m in fs['top_movies']]
    mid_to_count = {top[i]: int(counts[i]) for i in range(len(top))}
    return {
        't2m':   fs['title_to_movieId'],
        'm2t':   fs['movieId_to_title'],
        'count': mid_to_count,
        'gnames': fs['genome_tag_names'],                 # genome_tag_id -> name
        'gtag_to_i': fs['genome_tag_to_i'],               # genome_tag_id -> column index
        'm2genome':  fs['movieId_to_genome_tag_context'], # movieId -> 1128-vec
        'top': top,
    }


def _fmt(n):
    if n is None:
        return '   —  '
    if n >= 1_000_000:
        return f'{n/1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n/1_000:.1f}k'
    return str(n)


def cmd_check(d, titles):
    print(f"{'count':>8}  {'ok':>3}  title")
    print('─' * 64)
    miss = 0
    for t in titles:
        mid = d['t2m'].get(t)
        if mid is None:
            miss += 1
            print(f"{'—':>8}  {'NO':>3}  {t}")
        else:
            print(f"{_fmt(d['count'].get(int(mid))):>8}  {'yes':>3}  {t}")
    if miss:
        print(f"\n{miss}/{len(titles)} titles NOT in corpus — fix the exact string "
              f"(try `search`), they are silently dropped otherwise.")


def cmd_search(d, sub):
    sub = sub.lower()
    hits = [(t, d['count'].get(int(mid)))
            for t, mid in d['t2m'].items() if sub in t.lower()]
    hits.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))
    print(f"{len(hits)} titles matching '{sub}' (by popularity):\n")
    for t, c in hits[:60]:
        print(f"  {_fmt(c):>8}  {t}")
    if len(hits) > 60:
        print(f"  … {len(hits) - 60} more")


def cmd_tags(d, sub):
    sub = sub.lower()
    hits = sorted({nm for nm in d['gnames'].values() if sub in nm.lower()})
    print(f"{len(hits)} genome tags matching '{sub}':\n")
    for nm in hits:
        print(f"  {nm}")


def _anchors_for(d, genome_tags, exclude=()):
    """Top ANCHORS_PER_TAG movies per genome tag (mirrors evaluate._get_anchor_titles)."""
    name_to_idx = {d['gnames'][tid]: d['gtag_to_i'][tid] for tid in d['gtag_to_i']}
    out, seen = [], set(exclude)
    for tag in genome_tags:
        if tag not in name_to_idx:
            print(f"  !! genome tag '{tag}' NOT in vocab — try `tags`", file=sys.stderr)
            continue
        ti = name_to_idx[tag]
        ordered = sorted(d['top'], key=lambda m: float(d['m2genome'][m][ti]), reverse=True)
        n = 0
        for m in ordered:
            if n >= ANCHORS_PER_TAG:
                break
            title = d['m2t'][m]
            if title not in seen:
                out.append((title, d['count'].get(int(m))))
                seen.add(title)
                n += 1
    return out


def cmd_anchors(d, tags):
    print(f"Genome anchors injected for {tags} (weight 1.0 each, like the canary):\n")
    for t, c in _anchors_for(d, tags):
        print(f"  {_fmt(c):>8}  {t}")


def cmd_persona(d, name, tag, liked):
    print(f"PERSONA: {name}")
    print(f"genome tag: {tag}\n")
    print("Liked seeds (weight 2.0):")
    cmd_check(d, liked)
    print("\nGenome anchors (weight 1.0, auto-derived):")
    for t, c in _anchors_for(d, [tag], exclude=set(liked)):
        print(f"  {_fmt(c):>8}  {t}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sp = sub.add_parser('check');   sp.add_argument('titles', nargs='+')
    sp = sub.add_parser('search');  sp.add_argument('substr')
    sp = sub.add_parser('tags');    sp.add_argument('substr')
    sp = sub.add_parser('anchors'); sp.add_argument('tags', nargs='+')
    sp = sub.add_parser('persona')
    sp.add_argument('name'); sp.add_argument('tag')
    sp.add_argument('--liked', nargs='+', required=True)
    a = p.parse_args()

    d = _load()
    if   a.cmd == 'check':   cmd_check(d, a.titles)
    elif a.cmd == 'search':  cmd_search(d, a.substr)
    elif a.cmd == 'tags':    cmd_tags(d, a.substr)
    elif a.cmd == 'anchors': cmd_anchors(d, a.tags)
    elif a.cmd == 'persona': cmd_persona(d, a.name, a.tag, a.liked)


if __name__ == '__main__':
    main()
