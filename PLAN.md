# Plan: Web Deployment

## Status

- [x] Stage 1 — Serving artifacts (`serving/model.pth`, `movie_embeddings.pt`, `feature_store.pt`)
- [x] Stage 2 — Streamlit app (`app.py`) with recommend, similar, explore genre/genome, about tabs
- [x] `requirements.txt` created
- [ ] Stage 3 — Deploy to Streamlit Community Cloud

---

## Next Step: Deploy

1. Commit `serving/` and push
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the GitHub repo, set `app.py` as the entry point
3. Watch the build logs — if torch install fails, pin to a lower version that Streamlit Cloud supports

**Note on `requirements.txt`:** `torch==2.9.1` is very recent and may not be available on Streamlit Cloud. If the build fails on torch, drop to the latest supported version (check build logs).
