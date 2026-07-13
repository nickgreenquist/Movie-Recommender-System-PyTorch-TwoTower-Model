# Building an LLM-Powered Search and Discovery Experience for a Movie Recommender

> **TL;DR:** LLMs are becoming a core part of how industry recommendation systems work: Netflix's natural-language search, Spotify's AI playlists. This project brings that same idea to movies: describe what you want in your own words, and let an LLM translate it into a query a standard recommender retrieval system can serve.

## Background
**Netflix** is beta-testing a natural-language search that turns "something funny and upbeat" into a refinable list of titles. Type more into the same box and it narrows the results further; you don't start over. **Spotify**'s AI playlist tools take a sentence like "science explainers for my morning run" and hand back a generated queue, weighed against your listening history unless you tell it not to. Both use the LLM for one narrow job, turning what you typed into a query. The recommender that was already there still decides what comes back.

## Why Not Use an LLM for the Whole Recommendation Problem?

Why not just ask ChatGPT or Gemini for "movies like Rashomon" directly? Because it already knows the answer. Frontier LLMs have absorbed IMDb and Letterboxd along with everything else, which is what makes movies a convenient, checkable showcase, and also what makes them unrepresentative. An industrial recommender runs over a proprietary catalog the LLM has never seen, sized in the millions and far too large for any context window. The model has no memory to recommend from. That is the catalog this architecture is built for. The LLM extracts intent; a trained retrieval model that has seen the corpus does the matching.

### Why Not Fine-Tune Instead?

Fine-tuning an LLM on the catalog doesn't close the gap either. Generative recommenders that decode item names directly hallucinate titles the catalog doesn't contain, and they cost far more to serve. Indeed's fine-tuned GPT-3.5 recommender ran [6.7 seconds per inference](https://eugeneyan.com/writing/recsys-llm/), too slow to put online, and was swapped for a lightweight classifier in production. [Spotify's own generative-retrieval research](https://research.atspotify.com/2024/10/bridging-search-and-recommendation-with-generative-retrieval) found the approach consistently lagged specialized baselines like SASRec and BERT4Rec. My read is that text similarity alone doesn't capture collaborative-filtering signal.

## Why This Helps

The first problem is the cold start. Most recommenders need weeks of clicks before they're any good, and this project's model was built to skip that. A handful of liked movies is enough, with no retraining and no user-ID lookup. Describing a few movies you liked is exactly what a chat box is for.

The other user is the one who already has a history and wants something outside it. "Something to watch with my nephew this weekend" or "movies like Rashomon but not so bleak" won't surface in a For You row, and there's no title to type into a search bar. The Ask tab is for that gap. You describe what you're after and get a board back.

The business case:

- **Cuts new-user churn.** A good result on day one, no weeks of clicks required.
- **Catches high-intent users search can't serve.** They know what they want and can't name a title, so the search bar dead-ends on them. A sentence is enough for Ask.
- **Explainability comes for free.** The extraction already exists; showing it back to the user costs nothing.
- **Handles fuzzy titles and vague intent.** Misspellings, half-remembered titles, "I don't know exactly what I want."
- **Lets users steer.** They can push the results where they want them without waiting for the algorithm to catch on.

## Design
This project sits a small hosted LLM in front of a trained two-tower movie recommender. Type "classic samurai films full of sword duels and honor," or "movies directed by Akira Kurosawa, like Rashomon, Ran, and Ikiru," and Claude Haiku reads it and fills in a single structured object behind the scenes: liked titles, genres, people, mood, and hard constraints like a year range or a rating floor. A deterministic pipeline resolves that object into weighted movie anchors, hard filters, and re-rank terms. The trained model does the retrieval and the scoring, and nothing generative ever touches the ranking.

The prompt and the resolution logic were tuned against 500 synthetic queries generated to look like what a user would type, alongside my own manual testing. That pass moved the "good board" rate (boards where an LLM judge and my own review agreed the recommendations matched the query's intent) from 32.0% to 50.8%.

<p align="center">
  <img src="figures/fig1_landing.png" alt="The Ask tab: a chat-style search bar, a row of theme pills, a second row of related pills once a theme is open, and a results grid below" width="820">
</p>

*Click the suggested prompt "Samurai duels & honor" and a board of samurai films renders below (Seven Samurai, Yojimbo, Musashi Miyamoto, Rashomon, Throne of Blood). That click also opens a second row, "Riffing on: Samurai duels & honor," six related pills including "Directed by Kurosawa."*

## Where the LLM Stops
An LLM can't rank a catalog of thousands of movies by itself. There's no reliable scoring against items outside the few it can name, and no guarantee the titles it recites exist in the corpus at all. So the project never asks it to. Its words never reach the user either. There's no AI-written answer and no chatbot response rendered anywhere in the app; the model's job ends when the structured object is filled in, and everything past that point is the recommender. The "Show raw extraction" toggle lets you check that yourself:

<p align="center">
  <img src="figures/fig3_raw_extraction.png" alt="The Ask tab's debug expander open, showing the LLM's actual structured JSON output for a query about films directed by Akira Kurosawa" width="820">
</p>

*"Under the hood," normally collapsed. `liked_items` resolved to exact titles with years; a nested `hard_constraints.require_people` field carries the director's name. This is the entire LLM output for the query above: tool-call arguments, no prose.*

## Cost

Every live query is one Haiku call: about 12k tokens of system prompt and schema, a short user query, and up to 300 tokens back. At Haiku 4.5's list price ($1/M input tokens, $5/M output tokens), a cold call with no cache hit runs about **$0.017**. A warm call hits the 5-minute prompt cache on that 12k-token prefix and drops to about **$0.003**, roughly **6x cheaper**. A cache read costs a tenth of normal input price; writing the cache costs a 1.25x premium once. At the rate caps (20 queries per session, 60 per day across all visitors), the worst case, where every call misses the cache, is about **$1/day** for the entire public demo. In practice caching keeps it well under that.

The math changes at product scale. 10k daily users firing a couple of free-text queries each is 20k live calls a day, or **$2k-$10k a month** depending on cache hit rate. At that point you stop optimizing the call and start avoiding it. Semantic caching (GPTCache, Redis LangCache) catches near-duplicate queries and reuses the prior extraction instead of re-calling the model. The other lever is mining query logs to pre-resolve the most common patterns into that same cache, turning the demo's hand-curated pills into a lookup table nobody has to curate.

## What I Learned

What came out of the tuning, including a couple of things I'd rather have measured:

- **Few-shot examples beat more instructions.** A keyword-routing failure survived every rule I wrote at it, then went away once I added worked examples to the prompt.
- **Extended thinking added nothing.** Filling in a schema is a narrow extraction task, not open-ended reasoning. Forced tool use with thinking off is cheaper, and I never saw it cost accuracy.
- **Haiku is enough, though I never proved it.** No frontier model was ever run head-to-head against it here. The case is task fit (schema-filling, not chain-of-thought) plus a roughly 3x cost gap to Sonnet and 5x to Opus at list price.
- **The metadata carries more of the load than the model.** Person, studio, and keyword filters only work because a scraped metadata table exists for the LLM's output to resolve against. Before those tables existed, a request like "directed by Kurosawa" had nowhere to land, and the extraction silently dropped it while happily answering the rest of the query.
- **The remaining failures are specific.** Perfect routing was never the target, and what's left is narrow. Named titles, directors, genres, and concrete themes ("chess," "heist," "time loop") route reliably. Where it struggles is mood-only requests, place names like "movies set in Rome," and thin niche keywords. A term that matches only a few films corpus-wide will surface all of them regardless of fit, because there's nothing left to rank against.

## More Real Examples

Six root pills, each paired with one of its leaf pills. Expand any of them for the prompt and the top 5 posters:

<details>
<summary><strong>Anime, Ghibli to Akira</strong></summary>

**Prompt:** "Breathtaking anime films, from Studio Ghibli's warmth to the neon chaos of Akira and Ghost in the Shell."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/bLUUr474Go1DfeN1HLjE3rnZXBq.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/cMYCDADoLKLbB83g4WnJegaZimC.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/41XxSsJc5OrulP0m7TrrUeO2hoz.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/1ZJRbLDVr90KLtKdmTT4WZhT26E.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/41WtqvaObaYE3RuqgBfQFPV3aV1.jpg" width="90">
</p>

**Top 5:** Paprika (Papurika) (2006), Princess Mononoke (Mononoke-hime) (1997), Laputa: Castle in the Sky (Tenkû no shiro Rapyuta) (1986), Ghost in the Shell 2: Innocence (a.k.a. Innocence) (Inosensu) (2004), Metropolis (2001)

</details>

<details>
<summary><strong>Studio Ghibli classics</strong></summary>

**Prompt:** "Studio Ghibli movies like Spirited Away and My Neighbor Totoro."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/41XxSsJc5OrulP0m7TrrUeO2hoz.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/tcrkfB8SRPQCgwI88hQScua6nxh.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/cMYCDADoLKLbB83g4WnJegaZimC.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/13kOl2v0nD2OLbVSHnHk8GUFEhO.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/Aufa4YdZIv4AXpR9rznwVA5SEfd.jpg" width="90">
</p>

**Top 5:** Laputa: Castle in the Sky (Tenkû no shiro Rapyuta) (1986), Nausicaä of the Valley of the Wind (Kaze no tani no Naushika) (1984), Princess Mononoke (Mononoke-hime) (1997), Howl's Moving Castle (Hauru no ugoku shiro) (2004), Kiki's Delivery Service (Majo no takkyûbin) (1989)

</details>

<details>
<summary><strong>Christmas by the tree</strong></summary>

**Prompt:** "Cozy Christmas movies to watch by the tree."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/oOleziEempUPu96jkGs0Pj6tKxj.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/34nSHYqmb7222tiqiuKqKJmZiQa.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/2sMpffmvoJlW7WBcvyHTPR9DEsj.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/yF7SNheAGWavNnjbAronedwjC1N.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/7ir0iRuPK9OEuH569cp0nF5CJce.jpg" width="90">
</p>

**Top 5:** Elf (2003), A Christmas Story (1983), The Bishop's Wife (1947), Arthur Christmas (2011), How the Grinch Stole Christmas! (1966)

</details>

<details>
<summary><strong>Starring Santa himself</strong></summary>

**Prompt:** "Christmas movies where Santa Claus is a main character."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/g3JfCGX9fgHwmmKFhRzu7BYc29d.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/tZvGQF2623FD0MZgvOy18J5d7x5.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/9hTYFvEDnajUIbpdfc3fO6bZq2Y.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/hvV2rI60qOYELT7tHHLpxtafnBZ.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/pvaWMSRzRwtcxyHKNLX6phiQp8d.jpg" width="90">
</p>

**Top 5:** Prancer (1989), Santa Claus: The Movie (1985), I'll Be Home For Christmas (1998), The Santa Clause (1994), The Santa Clause 3: The Escape Clause (2006)

</details>

<details>
<summary><strong>Make me sob</strong></summary>

**Prompt:** "A devastating drama that will absolutely make me sob."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/qKnsyaJZLXfiL2JhIJEkpA8C3LU.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/jJiGGbuLW8IkEnh5073maEmIr7V.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/o9VXYOuaJxCEKOxbA86xqtwmqYn.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/yAhoeIwwtPxJ7uf9HLIMHIqWFnL.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/1ONOsIo5gby9hABPdknL7OAYBLm.jpg" width="90">
</p>

**Top 5:** Short Term 12 (2013), Still Alice (2014), Manchester by the Sea (2016), The Good Lie (2014), Wit (2001)

</details>

<details>
<summary><strong>Dogs that break your heart</strong></summary>

**Prompt:** "Emotional dramas about the bond between a person and their dog, like Hachi: A Dog's Story and My Dog Skip. No comedies."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/Xe7AB0ffhGPSdBZ4qJ9AKJfllL.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/5I7SYsNQmZRZpQ2MAarIQYU9vaX.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/wRB0cTPBIWHEq4FmWuX6voagePf.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/A4lH22nlFF7MdalGfcvSrlC1ttt.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/cVtF8SW4zMbculTXD52RDvmQAAR.jpg" width="90">
</p>

**Top 5:** Old Yeller (1957), Umberto D. (1952), Shiloh (1997), Amores Perros (Love's a Bitch) (2000), Wendy and Lucy (2008)

</details>

<details>
<summary><strong>Samurai duels & honor</strong></summary>

**Prompt:** "Classic samurai films full of sword duels and honor."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/lOMGc8bnSwQhS4XyE1S99uH8NXf.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/tN7kYPjRhDolpui9sc9Eq9n5b2O.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/2nCSzKqOslKrsvUHb2T1jsuh2eQ.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/vL7Xw04nFMHwnvXRFCmYYAzMUvY.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/zaZFMNxJST0TtPd68yF7fNt1he8.jpg" width="90">
</p>

**Top 5:** Seven Samurai (Shichinin no samurai) (1954), Yojimbo (1961), Samurai I: Musashi Miyamoto (Miyamoto Musashi) (1954), Rashomon (Rashômon) (1950), Throne of Blood (Kumonosu jô) (1957)

</details>

<details>
<summary><strong>Directed by Kurosawa</strong></summary>

**Prompt:** "Movies directed by Akira Kurosawa, like Rashomon, Ran, and Ikiru."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/zaZFMNxJST0TtPd68yF7fNt1he8.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/tN7kYPjRhDolpui9sc9Eq9n5b2O.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/zW47oIH3bc3ggmmmzTvKqM4Fqjk.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/iYNH3Re0JSypoYaWvZtERUnNaJL.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/fJgqj9s8HNZz9zwX6femVJn8HEB.jpg" width="90">
</p>

**Top 5:** Throne of Blood (Kumonosu jô) (1957), Yojimbo (1961), Sanjuro (Tsubaki Sanjûrô) (1962), Red Beard (Akahige) (1965), Kagemusha (1980)

</details>

<details>
<summary><strong>Nothing but time travel</strong></summary>

**Prompt:** "Movies where time travel is the whole point, like Back to the Future and Primer."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/hQq8xZe5uLjFzSBt4LanNP7SQjl.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/crzoVQnMzIrRfHtQw0tLBirNfVg.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/r67KzlHkdiwe3fBWzImEkNOg1Y.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/zdxQRP7mzczMJBwKnv8MCPRq7rQ.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/38Xr1JnV1ZcLQ55zmdSp6n475cZ.jpg" width="90">
</p>

**Top 5:** Back to the Future Part II (1989), Back to the Future Part III (1990), Frequently Asked Questions About Time Travel (2009), Timecrimes (Cronocrímenes, Los) (2007), Predestination (2014)

</details>

<details>
<summary><strong>Stuck in a time loop</strong></summary>

**Prompt:** "Groundhog Day-style time loops, like Edge of Tomorrow."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/nTr0lvAzeQmUjgSgDEHTJpnrxTz.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/mU4VeXVK18JtCZsy7i0zczlA9p7.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/cRlMrbc4Iof7vN0ZqHwJnMBYBLi.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/eowzonDJMCuNXoJGVkP9Z7oCmiM.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/wGHoYDswvx96PVF43qFErVAnQHS.jpg" width="90">
</p>

**Top 5:** Source Code (2011), ARQ (2016), Triangle (2009), Before I Fall (2017), Synchronicity (2015)

</details>

<details>
<summary><strong>Gritty pre-2000 New York</strong></summary>

**Prompt:** "Gritty New York City movies from before 2000, no musicals."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/rex4oaiSCD8UQg9204R6hQcFrX9.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/vNhywp9w1DVG6BytxKp4kjtaaGC.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/d6YZpBq4BhQr1K985J3CuL1cA8J.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/ckklq45UxUkwgHve9xItXqXr06r.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/1MyH4MJAJZJbb6wDVeOc2bTECtK.jpg" width="90">
</p>

**Top 5:** The Naked City (1948), The Taking of Pelham One Two Three (1974), The Warriors (1979), Midnight Cowboy (1969), King of New York (1990)

</details>

<details>
<summary><strong>Scorsese's New York</strong></summary>

**Prompt:** "Martin Scorsese's New York movies."

<p align="center">
<img src="https://image.tmdb.org/t/p/w342/1nD40aUcPAxYdE1WxERrTjZuFGe.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/eamOBurHBu0MIxohTIVcfxmZ6Z7.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/ekstpH614fwDX8DUln1a2Opz0N8.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/9msfwOeGc9uL1iRRTBdEf15XonC.jpg" width="90"> <img src="https://image.tmdb.org/t/p/w342/mViGEH5dfsAnUgJmce1RJkFycAi.jpg" width="90">
</p>

**Top 5:** New York, New York (1977), After Hours (1985), Taxi Driver (1976), Mean Streets (1973), New York Stories (1989)

</details>

## Try It

**[Try it in the Ask tab of the live demo.](https://movie-recommender-system-two-tower-model.streamlit.app/)** For how it works under the hood (the schema, the resolution pipeline, the tradeoffs behind keeping the LLM's output invisible), see the About tab in the app, or the [GitHub README](../../README.md).
