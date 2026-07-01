# Architectural Blueprint of the Bookend LLM Framework for Multi-Stage Recommendation Systems

Large Language Models (LLMs) have introduced transformative paradigms for personalized discovery by enabling systems to interpret unstructured user intents, subtle contextual nuances, and expressive natural language. However, deploying end-to-end generative models directly to retrieve and rank items from massive enterprise databases is computationally prohibitive, lacks deterministic safety guarantees, and is prone to hallucinating non-existent inventory. To resolve these limitations, modern enterprise architectures are adopting a hybrid design pattern known as the "Bookend LLM" framework.

This architectural blueprint wraps traditional, high-scale multi-stage recommendation pipelines (candidate generation, filtering, and deep ranking) in two specialized language model layers: an Intake LLM (the Front Bookend) and a Delivery LLM (the Back Bookend). The Front Bookend acts as an intelligent conversational interface that translates natural language dialog into structured search queries and database filters. The core recommendation engine processes these queries with sub-millisecond latency. Finally, the Back Bookend synthesizes the retrieved candidates and their accompanying metadata to generate natural, personalized, and contextually grounded explanations.

## The Intake LLM (Front Bookend Design)

The Front Bookend is designed to resolve the ambiguity of natural language expressions and translate them into machine-readable parameters. This process requires robust intent classification, strict slot extraction, and a strategy for managing state across conversational turns without causing token bloat.

### Prompt Structuring and Intent Extraction

The Intake LLM must classify user intent and extract precise constraints from free-form text. System prompts are structured using role definition, markdown formatting, and explicit XML tags to prevent data-instruction confusion. The Front Bookend generally operates a multi-label intent classifier to determine whether a user is providing a preference, inquiring about a specific item, or accepting or rejecting a previously suggested candidate.

To handle indirect context expansion and negative constraints, the model leverages few-shot templates containing historical exemplars. When a user states, *"I'm watching my weight, but I want something quick and spicy,"* the Intake LLM must programmatically infer a hard negative constraint (avoiding high-calorie categories) and expand the positive criteria to include attributes like "low-calorie", "fast food", and "spicy". Prompt files like `constraints_updater_fewshots.csv` are utilized to demonstrate how positive patterns are extracted into structured formats, prioritizing positive instruction modeling over negative prohibitions.

### Dialogue State Tracking and JSON State Schemas

Rather than passing raw text transcripts to downstream services, the Intake LLM serializes the conversation state into a semi-structured JSON representation. This data structure divides constraints into hard parameters (mandatory filters) and soft parameters (flexible preferences). The JSON state is dynamically updated on each turn using the latest user utterance and the previous constraints.

| Schema Key | Extraction Type | Target Match Field | Constraint Resolution Logic |
| :-- | :-- | :-- | :-- |
| `hard_constraints` | Exact Attribute Extraction | location, cuisine_type, price_range | Must strictly match database fields; triggers database filtering. |
| `soft_constraints` | Semantic Attribute Extraction | dietary_restrictions, atmosphere, service_speed | Maps to vector embeddings or text indices for soft scoring. |
| `avoidance_filters` | Negative Phrase Extraction | excluded_genres, banned_ingredients | Appends a hard logical NOT condition to the query engine. |
| `item_of_interest` | Entity Recognition | product_id, item_title | Tracks the focal point of current multi-turn evaluation. |
| `dialogue_act_history` | Sequence Categorization | interaction_states | Prevents conversational loops (e.g., repeating clarifying questions). |
| `others` | Catch-All Extraction | unclassified_aspects | Captures unspecified preference types for semantic query expansion. |

The system uses this JSON schema to execute database queries. If a user says, *"Show me a cozy Italian spot, but definitely not in the downtown core,"* the Intake LLM updates the dialogue state as follows:

```json
{
  "intent": "PROVIDE_PREFERENCE",
  "hard_constraints": {
    "cuisine_type": ["italian"]
  },
  "soft_constraints": {
    "atmosphere": ["cozy"]
  },
  "avoidance_filters": {
    "location": ["downtown"]
  },
  "item_of_interest": null,
  "dialogue_act_history": ["ASKING", "PROVIDE_PREFERENCE"],
  "others": null
}
```

### State Management and Context Compression

To maintain context across long conversations, passing the entire raw chat log on every API call is economically and computationally unfeasible. Instead, conversational recommenders use a multi-tiered memory architecture to optimize token efficiency and minimize latency.

```
User Query ---> [Intake LLM (Front Bookend)]
                     |
                     +---> Extract Intent & Constraints
                     |
                     +---> State-Aware Memory Layer
                              |
                              +---> Active State (JSON Update)
                              +---> Vectorized Memory (Episodic Store)
                              +---> Recursive Summarization (Historical Log)
```

First, the system maintains the current turn's JSON dialogue state as the absolute working memory of user preferences. Second, when the dialogue history exceeds a specific token threshold, a background process summarizes older exchanges into a dense factual representation while keeping the most recent exchanges verbatim in the active window. Third, past interactions are converted into vector embeddings and written to a vector database. During live turns, the incoming user query retrieves only the semantically similar historical snippets, appending them to the prompt context.

To manage this workflow, modern implementations utilize a shared working memory object, such as `AgentsModule`, which acts as a typed dataclass passed directly into each agent's instruction function. By wrapping this typed state using a generic context wrapper, systems avoid global variables and enforce thread safety. This design pattern clearly separates factual memory—such as the extracted `UserProfile`—from behavioral memory, which contains strategy suggestions generated during error-driven reflection. Consequently, the system avoids polluting the user's actual preference profile with operational instructions on how the agent should behave, preventing interaction loops and minimizing token overhead.

## The Handoff Mechanics to Core Rec-Systems

Once the Intake LLM generates a structured JSON payload, this state must be converted into inputs that traditional retrieval and ranking components can ingest.

```
[Intake JSON State]
         |
         v
[Query Generation & Translation Layer]
         |
         +---> Hard SQL Filters (e.g., location != 'downtown')
         +---> Semantic Vector Generation (Text & Graph Embeddings)
         |
         v
[Core Multi-Stage Engine] ---> [1. Candidate Generation] ---> [2. Deep Ranking (Collaborative Signals)]
```

### Translating Structured LLM Outputs into Retrieval Inputs

Traditional recommendation architectures use a multi-stage funnel consisting of candidate generation (retrieval), filtering, and ranking. The handoff layer maps the JSON state keys directly onto these stages. This mapping is executed through a dedicated translation layer that reads from a filtering configuration file, specifying the matching logic to apply between the conversational state keys and the database schema.

| Filter Type | Matching Logic | Target State Key | Applied Database Field |
| :-- | :-- | :-- | :-- |
| Exact Word Matching | Case-insensitive matching; retains items if state value exactly matches database string or list. | `hard_constraints.location` | `item_catalog.location_id` |
| Word In Matching | Case-insensitive lookup; accounts for plural forms and partial matches. | `hard_constraints.cuisine_type` | `item_catalog.category_tags` |
| Value Range Filter | Numeric comparison; retains items if database value falls within specified range boundaries. | `hard_constraints.price_range` | `item_catalog.price_numeric` |
| Item Filter | Exact entity resolution; filters or matches specific item identifiers or recommended item lists. | `item_of_interest` | `item_catalog.product_id` |

Soft constraints are converted into text embeddings. These query vectors are matched against pre-computed item embeddings in a vector database using Approximate Nearest Neighbor (ANN) search algorithms, yielding a semantic candidate set. The similarity score between a query vector and an item vector is calculated via cosine similarity, the overall user profile is updated dynamically based on real-time feedback, and to ensure diversity the candidate generator can select items across distinct thematic clusters.

### Balancing Long-Term History against Immediate Session Context

A key challenge in modern recommendation architecture is balancing a user's long-term historical profile (global collaborative signals) with their immediate conversational inputs. Relying solely on historical behavior causes system bias, making it difficult for the user to steer the recommendations in a new direction. Conversely, ignoring historical context results in generic recommendations that overlook established user preferences (e.g., preferred brands or price sensitivity).

To achieve this balance, systems use a hybrid scoring engine that blends historical user vectors with active session vectors using a dynamic attention parameter. The dynamic parameter shifts based on explicit indicators in the conversation. For instance, if the Front Bookend detects high "turn pressure" or explicit instructional feedback (such as *"I need something for a friend's baby shower, not for myself"*), the parameter is set to temporarily de-bias the ranking pipeline from the user's personal historical preferences.

Furthermore, to maintain high catalog coverage and improve long-tail performance, the candidate generation layer can completely disregard collaborative filtering signals during the initial semantic retrieval stage. This strategy prioritizes pure semantic matching between the active user preferences and item descriptions, ensuring that newer or less popular items are not prematurely filtered out due to a lack of historical interaction data. Collaborative signals and historical interaction history are then re-introduced in the downstream deep ranking and scoring layers to refine the final selection.

### RAG Integration and Retrieval Late Fusion

To enhance the performance of the candidate generation and scoring steps, architectures utilize Reviewed-Item Retrieval (RIR) based on late fusion techniques. Rather than summarizing user reviews at an item level before scoring (early fusion), a late fusion pipeline scores individual text reviews directly against the generated query and then aggregates these scores to the item level.

Mathematically, this late fusion approach preserves critical, localized semantic nuances that are typically lost during early text summarization. These aggregated scores act as high-fidelity matching indicators that are fed directly into the downstream ranking model, establishing a direct link between conversational intent and specific qualitative catalog reviews.

## The Delivery LLM (Back Bookend Design and Grounding)

The final phase of the framework uses a generative language model (the Delivery LLM) to present the selected candidates to the user. The Delivery LLM must format recommendations naturally, ground explanations in verified catalog data, and eliminate hallucinated product claims.

```
[Traditional Rec Engine Matches] ---> [Catalog Database Lookup]
                                             |
                                             v
[Enriched Input Packet] ---------------------+
  - Core Match Scores                        |
  - Raw Catalog Metadata (Attributes/Tags)   |
  - Top Verified Customer Reviews            |
                                             v
                                  [Delivery LLM (Back Bookend)]
                                             |
                                             v
                                  [Output Verification Filter] ---> Grounded Response
```

### Prompt Engineering Strategies for Factual and Friendly Explanations

The Delivery LLM translates structured database outputs into natural explanations. The prompt structure employs role-prompting and constraint-based directives to balance an engaging conversational tone with factual accuracy. System instructions explicitly restrict the model from making unsupported claims. For example, the prompt structures require that every descriptive assertion about an item be directly traceable to the provided metadata attributes or customer review snippets.

To verify the logic of these recommendations and understand why a black-box model produced specific candidates, architectures utilize surrogate models. By training a generative model (such as RecExplainer) to mimic and comprehend the complex latent spaces of deep recommendation models, the system can output natural language explanations that align with the actual math of the core engine.

### Input Metadata Enrichment and Comparison Generation

To ground its responses, the Delivery LLM is supplied with an enriched data packet containing both raw catalog metadata and internal engine signals. This package includes categorical and numeric metadata (e.g., pricing, attributes, stock status) combined with internal engine signals (e.g., matching indicators, top scoring reviews retrieved via RIR).

When comparing multiple items, a standard single-decoder natural language generation model can struggle to maintain factual consistency across multi-dimensional product attributes. To resolve this, architectures utilize a multi-decoder, multi-task generative framework (such as Human Centered Product Comparison, or HCPC). This design splits the generation process: one decoder is dedicated to producing the conversational comparison text, while a second parallel decoder generates structured supportive tokens representing exact product attribute names and values. This dual-stream decoding mathematically forces the narrative text to remain faithful to the structured input data, preventing factual drift during complex comparative tasks.

### Architectural Safeguards Against Product Hallucination

To completely eliminate product hallucination, the system implements multi-layered architectural safeguards rather than relying solely on prompt instructions:

- **Dynamic Bounding-Box Retrieval:** Rather than allowing the LLM to access the entire catalog, the delivery layer uses dynamic bounding-box retrievals. It streams only the localized, valid candidate dataset into the model's immediate context window, preventing the model from discussing unavailable or out-of-stock items.
- **Automated Exam-Generation (Item Response Theory):** To systematically measure and prevent hallucinations in RAG pipelines, systems deploy an automated exam-generation workflow. This framework uses an LLM to generate difficult multiple-choice questions from the source catalog documentation. It applies semantic and Jaccard similarity metrics to filter out low-quality questions, then tests the RAG pipeline's factual accuracy using Item Response Theory (IRT). This creates a robust, model-agnostic benchmark to assess hallucination rates before production deployment.
- **Deterministic Post-Filtering:** A rule-based parser runs downstream of the Delivery LLM to verify that every product name or ID mentioned in the generated text exists in the verified inventory dataset. Any output that fails this token check is discarded, and the system falls back to a template-driven response.

## Industry Case Studies and Architectural Blueprints

Major engineering teams have successfully implemented variations of the Bookend LLM architecture to handle scale, improve accuracy, and lower computational overhead.

| Company | Intake Architecture (Front Bookend) | Handoff & Core Engine Mechanics | Delivery Architecture (Back Bookend) | Evaluation & Validation Patterns |
| :-- | :-- | :-- | :-- | :-- |
| Netflix | Interaction Tokenization; parses sequences of multi-modal catalog events. | Orthogonal Low-Rank Transformations to stabilize embedding spaces. | Multi-modal chronological mapping (Cassandra nested docs). | Speech & Music Activity Detection (SMAD) on noisy labels. |
| Spotify | Multi-Agent router using Agent Development Kit (Gemini Pro). | Function calling via Google ADK to map intent to APIs. | Creative storytelling grounded in listening logs & country context. | Offline-online signal calibration; evaluation funnels. |
| Amazon | Rufus custom shopping-trained language model. | RAG over Catalog, Reviews, and live Store APIs. | Multi-decoder (HCPC) for structured comparisons. | IRT-based automated exam generation for RAG pipeline testing. |
| DoorDash | Few-shot KNN extraction via OpenAI Embeddings on unstructured data. | Brand and attribute verification against internal Knowledge Graph. | Structured attribute injection into downstream ranking models. | Centralized model platform for unified prompting and deployment. |

### Netflix: Large-Scale Foundation Models and Multi-Modal Search

Netflix operates a large-scale foundation model architecture designed to consolidate personalized recommendations across its entire user base. To bridge the gap between massive user interaction history and standard LLM context windows, Netflix uses a process called interaction tokenization. This compression balances sequence length against the detail retained in individual tokens. Rather than training many small models, a central model predicts auxiliary targets (such as genre or original language) to narrow down the candidate list before predicting the specific next item ID.

Because recommendation embedding dimensions can become incompatible across model training runs, Netflix applies an orthogonal low-rank transformation to stabilize the user and item embedding space. This ensures that downstream consumer models do not break during retraining cycles. To support multi-modal video search, Netflix manages disjointed chronological annotations across billions of frames. It uses Apache Cassandra for transactional persistence, structuring each temporal bucket as a nested document where the root level captures asset context and child documents house multi-modal labels and vector coordinates, enabling real-time cross-annotation queries. Additionally, the audio pre-processing pipeline utilizes Speech and Music Activity Detection (SMAD) trained on noisy labels of the catalog to segment audio frames, prepping clean corpora for downstream translation and localization engines.

### Spotify: Multi-Agent Intent Resolution and Evaluation Funnels

Spotify's advertising platform (Ads AI) utilizes a multi-agent system built on Google's Agent Development Kit (ADK) and Vertex AI (Gemini Pro) to translate natural language media plans into programmatic API calls. The system delegates orchestration through a specialized routing layout:

```
User Intent ---> [RouterAgent] ---> [GoalResolverAgent] -----> Programmatic API Call
                      | ----------> [AudienceResolverAgent] -> (Standardized JSON)
                      | ----------> [BudgetAgent] -----------^
```

The incoming user request is first parsed by a RouterAgent (the traffic controller) which identifies which parameters are present, preventing unnecessary parallel LLM calls. The work is then distributed to specialized resolution agents in parallel: the GoalResolverAgent maps the intent to standardized campaign objectives, the AudienceResolverAgent extracts targeting parameters (demographics, geographics, and interests) from a predefined taxonomy, and the BudgetAgent resolves financial constraints. These agents execute tools to validate data against real, bounded database endpoints.

To optimize this system, Spotify employs a rigorous evaluation funnel. Automated LLM evaluations assess dimensions like relevance, coherence, and tone before live A/B testing begins. These offline evaluations are continuously calibrated against online user interaction metrics to ensure that offline judgments align with actual business outcomes.

For creative features like Wrapped Archive, Spotify generates personalized narrative reports. The prompt structure ingests raw listening logs, pre-computed stats blocks, previously generated reports (to avoid repetition), and user location data (to ensure correct spelling and vocabulary). This pipeline is evaluated via LLM-as-a-judge models combined with human-in-the-loop review to balance creative expression with safety guidelines. To manage data across its platform, Spotify uses an AI data assistant that organizes petabytes of schema metadata into clusters managed by human domain experts, teaching the LLM business-specific terminology and definitions.

### Amazon: Conversational Assistant and Structured Product Comparisons

Amazon's conversational shopping assistant, Rufus, uses a custom-trained large language model optimized for shopping behavior. The assistant utilizes a RAG pipeline that pulls evidence from customer reviews, community Q&As, the product catalog, and stores APIs. To support simultaneous requests, Rufus uses a streaming architecture that outputs responses token-by-token along with formatting markup instructions.

To evaluate RAG pipeline performance and mitigate hallucination risk, Amazon deploys an automated exam-generation framework. This framework uses an LLM to generate difficult multiple-choice questions from the source catalog documentation. It applies semantic and Jaccard similarity filters to remove low-quality questions, and then evaluates the RAG pipeline's accuracy using Item Response Theory (IRT).

For product comparisons, Amazon uses a multi-task generative framework called Human Centered Product Comparison (HCPC). This architecture detects key product attributes from customer reviews using unsupervised sentiment mining. It then deploys a multi-decoder transformer model: one decoder produces the natural language comparison text, while a second decoder generates structured attribute-value pairs to support and validate the narrative. To resolve categorical attributes that do not have a natural numeric order, the system determines similarity based on semantic knowledge and customer search behavior, enabling scalable, weakly supervised comparisons.

### DoorDash: Automated Attribute Extraction and Knowledge Graph Grounding

DoorDash uses an LLM-based pipeline to extract product attributes and build a high-quality retail catalog from unstructured merchant SKU data. This structural mapping is critical for downstream personalization, stock substitution, and search indexing.

```
Merchant Catalog SKU ---> [LLM Extraction Pipeline] ---> [Entity Validation Graph]
                                                                |
                                                                v
                                                   Structured Categorical Tags
                                                                |
                                                                v
                                                   [Traditional Deep Ranker] ---> User Feed
```

The intake pipeline uses an LLM to extract brand names and descriptive attributes from raw string listings and optical character recognition (OCR) data from packaging photos. To resolve entity duplication, the extracted attributes are passed to a second LLM that queries a product knowledge graph to check for duplicate listings.

To scale annotations with high precision, the system employs a KNN-RAG approach. It maps unannotated SKUs to text embeddings, retrieves the most similar SKUs from a golden annotation set using approximate nearest neighbors, and passes these matched examples to GPT-4 as few-shot context. This approach ensures that the in-context examples are highly relevant to the target item, reducing hallucination rates. The validated, structured attributes are then fed directly into traditional deep ranking models, ensuring real-time personalization without serving-time LLM overhead.

## Architectural Synthesis and Design Guidelines

Implementing a Bookend LLM framework requires balancing trade-offs across latency, cost, explanation quality, and safety. Designers can use the following principles to guide system implementation:

- **Decouple Text Generation from Catalog Retrieval:** Never allow the LLM to search for items directly from its internal parametric memory. The candidate pool must always be determined by a traditional search and database engine before being passed to the delivery layer.
- **Optimize the State Payload:** Do not pass the entire conversation history to the LLM on every turn. Instead, use state tracking to maintain an active JSON representation of user constraints. This approach reduces prompt token size, decreases costs, and lowers response latency.
- **Align Evaluation Funnels:** Use offline simulation and LLM-as-a-judge frameworks to evaluate conversational quality. This strategy helps calibrate prompt revisions and verify response factualness before running live A/B tests.
- **Incorporate Hard Stop Safeguards:** Implement deterministic filtering downstream of the generative models. Real-time validation parsers must verify that any mentioned product is available in the inventory before the final text is displayed to the user.

## Works Cited

1. The Application of Large Language Models in Recommendation Systems — arXiv, <https://arxiv.org/html/2501.02178v2>
2. arXiv:2408.10946v2 [cs.AI] 28 May 2025, <https://arxiv.org/pdf/2408.10946>
3. microsoft/RecAI: Bridging LLM and Recommender System — GitHub, <https://github.com/microsoft/recai>
4. [2406.00033] Retrieval-Augmented Conversational Recommendation with Prompt-based Semi-Structured Natural Language State Tracking — arXiv, <https://arxiv.org/abs/2406.00033>
5. Building Trust in the Skies: A Knowledge-Grounded LLM-based Framework for Aviation Safety — arXiv, <https://arxiv.org/html/2604.13101v1>
6. D3Mlab/llm-convrec: LLM-based Conversational Recommendation Architecture — GitHub, <https://github.com/D3Mlab/llm-convrec>
7. Leverage LLM for Next-Gen Recommender Systems: Technical Deep Dive into LLM-Enhanced Recommender Architectures, <https://lfaidata.foundation/communityblog/2025/08/25/leverage-llm-for-next-gen-recommender-systems-technical-deep-dive-into-llm-enhanced-recommender-architectures/>
8. Retrieval-Augmented Conversational Recommendation with Prompt-based Semi-Structured Natural Language State Tracking — arXiv, <https://arxiv.org/html/2406.00033v1>
9. Retrieval-Augmented Conversational Recommendation with Prompt-based Semi-Structured Natural Language State Tracking, <https://ssanner.github.io/papers/sigir24_llmrec.pdf>
10. Building Recommendation Systems with Streaming Data | Conduktor, <https://www.conduktor.io/glossary/building-recommendation-systems-with-streaming-data>
11. Recommendation System Design: (Step-by-Step Guide), <https://www.systemdesignhandbook.com/guides/recommendation-system-design/>
12. LLM Summarization Techniques For Managing Chat History 2026 — Mem0, <https://mem0.ai/blog/llm-chat-history-summarization-guide-2025>
13. How Should I Manage Memory for my LLM Chatbot? — Vellum, <https://www.vellum.ai/blog/how-should-i-manage-memory-for-my-llm-chatbot>
14. prompt-engineering-frontier-llms.md — GitHub, <https://github.com/stevekinney/stevekinney.net/blob/main/writing/prompt-engineering-frontier-llms.md>
15. 13 Practical Tips to Get the Most Out of GPT-4.1 (Based on a Lot of Trial & Error) — Reddit, <https://www.reddit.com/r/PromptEngineering/comments/1k0ft09/13_practical_tips_to_get_the_most_out_of_gpt41/>
16. Retrieval-Augmented Conversational Recommendation with Prompt-based Semi-Structured Natural Language State Tracking — arXiv, <https://arxiv.org/pdf/2406.00033>
17. Multi-Agent Conversational Recommender System: Agentic Patterns with OpenAI Agents SDK | DecodedPapers, <https://decodedpapers.com/posts/lessons-learned-multi-agent-conversational-recommender-system/>
18. How to Make AI Remember User Preferences Across Conversations (May 2026), <https://supermemory.ai/blog/how-to-make-ai-remember-user-preferences-across-conversations>
19. LLM sessions and manual history management — Koog, <https://docs.koog.ai/sessions/>
20. Aman's AI Journal • Recommendation Systems • LLM, <https://aman.ai/recsys/LLM/>
21. LLMs as Retrieval and Recommendation Engines — Part 1 | by Moein Hasani | Medium, <https://medium.com/@moeinh77/llms-as-retrieval-and-recommendation-engines-part-1-43ceecb8e79b>
22. Conversational Recommender System — Kaushik Rangadurai, <https://www.weak-learner.com/blog/2020/04/15/conversational-recommender/>
23. How Collaborative Filtering Powers Netflix, Amazon, and Spotify | by Urvesh Pralhad Somwanshi | Medium, <https://medium.com/@somwanshiurvesh/how-collaborative-filtering-powers-netflix-amazon-and-spotify-e9bc36088499>
24. What is a Recommendation System? The Invisible Force Behind Netflix, Amazon, and Spotify | Hightouch, <https://hightouch.com/blog/recommendation-system>
25. A Three-Layer Playbook for Reducing LLM Bias: Prompts, Data and Filters — Medium, <https://medium.com/@Ayo.ore/a-three-layer-playbook-for-reducing-llm-bias-prompts-data-and-filters-f345c4458164>
26. Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation — arXiv, <https://arxiv.org/html/2508.05657>
27. A Multi-Agent Conversational Recommender System — arXiv, <https://arxiv.org/html/2402.01135v1>
28. Inside the Archive: The Tech Behind Your 2025 Wrapped Highlights | Spotify Engineering, <https://engineering.atspotify.com/2026/3/inside-the-archive-2025-wrapped>
29. What Is AI Grounding and How Does It Work? — You.com, <https://you.com/resources/ai-grounding>
30. Generating Explainable Product Comparisons for Online Shopping — Amazon Science, <https://assets.amazon.science/5d/03/2f7e2ab8407cb37e679211c2c677/generating-explainable-product-comparisons-for-online-shopping.pdf>
31. The technology behind Amazon's GenAI-powered shopping assistant, Rufus, <https://www.amazon.science/blog/the-technology-behind-amazons-genai-powered-shopping-assistant-rufus>
32. How 10 AI Startups are Grounding AI in the Real World with Overture, <https://overturemaps.org/blog/2026/how-10-ai-startups-are-grounding-ai-in-the-real-world-with-overture/>
33. Building DoorDash's product knowledge graph with large language models, <https://careersatdoordash.com/blog/building-doordashs-product-knowledge-graph-with-large-language-models/>
34. Our Multi-Agent Architecture for Smarter Advertising | Spotify Engineering, <https://engineering.atspotify.com/2026/2/our-multi-agent-architecture-for-smarter-advertising>
35. Foundation Model for Personalized Recommendation | by Netflix Technology Blog, <https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39>
36. Synchronizing the Senses: Powering Multimodal Intelligence for Video Search | by Netflix Technology Blog, <https://netflixtechblog.com/powering-multimodal-intelligence-for-video-search-3e0020cf1202>
37. Detecting Speech and Music in Audio Content | by Netflix Technology Blog, <https://netflixtechblog.com/detecting-speech-and-music-in-audio-content-afd64e6a5bf8>
38. Better Experiments with LLM Evals — A funnel, not a fork | Spotify Engineering, <https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork>
39. Encoding Your Domain Expert: The Context Layer Behind Spotify's Data Assistant, <https://engineering.atspotify.com/2026/6/encoding-your-domain-expert-the-context-layer-behind-spotifys-data-assistant>
