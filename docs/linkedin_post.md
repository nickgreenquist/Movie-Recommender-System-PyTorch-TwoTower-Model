$200 vs $200k: Generating Item Features With an LLM Instead of Hand Tagging

Two-tower retrieval models can run solely on item-ID embeddings — but almost every industry model adds item features. The cheap alternative to expensive, human-curated item features — scrape the web for text (available for most items), then have an LLM score it.

I ran a controlled experiment on MovieLens 32M: same architecture, same training, same eval, only content varied.

The result, using Mean Reciprocal Rank:

1. Base model (no movie features): 0.1121
2. Curated tags: 0.1148
3. LLM features: 0.1155

The $200 option matches 15 years of human curation — at a fraction of the cost.

Why? An LLM recovers most of the rich item features from web text: genre, era, plot.

Most teams have no human tagging budget. The tagging path isn't only expensive — it's unavailable.

Solution: leverage LLMs
