## Homework

Answers can be found in `readme.ipynb`

### Installation

To install the full architecture:

`docker-compose up --build`

This will deploy the backend on port 8000 and steamlit UI on port 8501. In the UI you can query lexical or semantic search using MiniLm or TFIDF.

Run lexical retrieval by running the script:

`python src/lexical_retrieval.py `

Run semantic search by running the script:

`python src/semantic_search.py`


 Following options can be added:

`--from_cache <cache_path>` to use a precomputed embeddings (cached_embedding.pkl)

`--cache` cache the embeddings for future run

`---plot` to plot tsne and Umap to inspect embedding vector space.

**Disclamer: I used Cursor to assist me.**
