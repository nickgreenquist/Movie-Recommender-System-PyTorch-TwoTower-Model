# Phase-0 Extraction Task (Claude Code Sonnet)

You are doing structured feature extraction for a movie recommendation experiment.
Your job: read each movie's feed + prompts, score every feature, save the scores.
This does NOT require an API key — you are the extractor.

## Commands (run from repo root)

```bash
# See what's left
python -m llm_features.batch_extract --remaining

# Show the next movie's full task (feed + all 6 group prompts)
python -m llm_features.batch_extract --show

# Save your scores (only non-zero features needed)
python -m llm_features.batch_extract --save '<movieId>' '<json_dict>'

# After all 23 done: check similarity
python -m llm_features.batch_extract --similarity
```

## Scoring rules (same as the prompts, repeated here for reference)

- **0.0** = definitely absent
- **1.0** = extremely prominent / central to the film
- In between = partial, minor, or background presence
- **Use the full range** — do NOT default to 0.5 when unsure; make a calibrated estimate
- **Most features are 0.0** for any given film — only the few that genuinely apply get high scores
- Score from the feed text only; if the text is silent on something factual (award, setting), score 0.0

## What good output looks like

A film should have ~15–25 non-zero features out of 132. Mostly zeros. A few decisive highs (0.8–1.0).
No lazy 0.5 defaults. Example for City of God: `{"crime": 0.95, "gangster": 0.80, "based_on_true_story": 0.90, "coming_of_age": 0.65, "murder": 0.70, "violent": 0.85, "foreign_language": 1.0, ...}`

## When you're done with all 23 movies

Run `python -m llm_features.batch_extract --similarity` and paste the output back
to the main session for review.
