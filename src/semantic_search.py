## SEMANTIC SEARCH
# Install required packages if not already installed
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from utils import search_queries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import argparse
import pickle


# Function to split text into chunks rather than just sentences
def split_into_chunks(text, chunk_size=3):
    """Split text into chunks of multiple sentences.

    Args:
        text (str): The input text
        chunk_size (int): Number of sentences per chunk

    Returns:
        list: List of text chunks
    """
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


# Create semantic search function
def semantic_search(query, model, chunk_embeddings, chunks, top_k=5):
    """Search for chunks semantically similar to the query.

    Args:
        query (str): The search query
        top_k (int): Number of results to return

    Returns:
        list: Top k most similar chunks with scores
    """
    # Get query embedding
    query_embedding = model.encode([query])[0]

    # Calculate cosine similarity between query and all chunks
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

    # Get indices of top k similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Return results
    results = []
    for idx in top_indices:
        results.append({"passage": chunks[idx], "score": similarities[idx]})

    return results


# Add visualization functions for the vector space
def visualize_embeddings():
    """Generate visualizations of the embedding vector space"""
    print("Generating vector space visualizations...")

    # Sample some chunks for visualization (use all if fewer than 1000)
    max_samples = min(1000, len(chunks))
    sample_indices = np.random.choice(len(chunks), max_samples, replace=False)
    sample_embeddings = chunk_embeddings[sample_indices]

    # Create some query embeddings to show in the same space
    example_queries = [
        "magic wand",
        "Harry Potter",
        "Voldemort",
        "school",
        "friendship",
    ]
    query_embeddings = model.encode(example_queries)

    # 1. t-SNE visualization (2D)
    print("Generating t-SNE plot...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(np.vstack([sample_embeddings, query_embeddings]))

    plt.figure(figsize=(12, 10))
    plt.scatter(
        tsne_results[:max_samples, 0],
        tsne_results[:max_samples, 1],
        c="lightgray",
        alpha=0.5,
        s=30,
        label="Text chunks",
    )
    plt.scatter(
        tsne_results[max_samples:, 0],
        tsne_results[max_samples:, 1],
        c=sns.color_palette("husl", len(example_queries)),
        s=100,
        marker="*",
    )

    # Add labels for queries
    for i, query in enumerate(example_queries):
        plt.annotate(
            query,
            (tsne_results[max_samples + i, 0], tsne_results[max_samples + i, 1]),
            fontsize=12,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("t-SNE Visualization of Document and Query Embeddings", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tsne_plot.png", dpi=300)
    plt.show()

    # 2. UMAP visualization (more preserves global structure)
    print("Generating UMAP plot...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_results = reducer.fit_transform(
        np.vstack([sample_embeddings, query_embeddings])
    )

    plt.figure(figsize=(12, 10))
    plt.scatter(
        umap_results[:max_samples, 0],
        umap_results[:max_samples, 1],
        c="lightgray",
        alpha=0.5,
        s=30,
        label="Text chunks",
    )
    plt.scatter(
        umap_results[max_samples:, 0],
        umap_results[max_samples:, 1],
        c=sns.color_palette("husl", len(example_queries)),
        s=100,
        marker="*",
    )

    # Add labels for queries
    for i, query in enumerate(example_queries):
        plt.annotate(
            query,
            (umap_results[max_samples + i, 0], umap_results[max_samples + i, 1]),
            fontsize=12,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("UMAP Visualization of Document and Query Embeddings", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("umap_plot.png", dpi=300)
    plt.show()

    # 3. 3D visualization with t-SNE
    print("Generating 3D t-SNE plot...")
    tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    tsne_results_3d = tsne_3d.fit_transform(
        np.vstack([sample_embeddings, query_embeddings])
    )

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Plot chunks
    ax.scatter(
        tsne_results_3d[:max_samples, 0],
        tsne_results_3d[:max_samples, 1],
        tsne_results_3d[:max_samples, 2],
        c="lightgray",
        alpha=0.5,
        s=30,
        label="Text chunks",
    )

    # Plot queries with different colors
    colors = sns.color_palette("husl", len(example_queries))
    for i, query in enumerate(example_queries):
        ax.scatter(
            tsne_results_3d[max_samples + i, 0],
            tsne_results_3d[max_samples + i, 1],
            tsne_results_3d[max_samples + i, 2],
            c=[colors[i]],
            s=100,
            marker="*",
            label=query,
        )

    ax.set_title("3D t-SNE Visualization of Embeddings", fontsize=16)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_zlabel("Dimension 3", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tsne_3d_plot.png", dpi=300)
    plt.show()

    # 4. Similarity heatmap for queries
    print("Generating similarity heatmap...")
    query_similarity = cosine_similarity(query_embeddings)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        query_similarity,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=example_queries,
        yticklabels=example_queries,
    )
    plt.title("Semantic Similarity Between Example Queries", fontsize=16)
    plt.tight_layout()
    plt.savefig("query_similarity_heatmap.png", dpi=300)
    plt.show()

    # 5. Analyze specific query with similarity coloring
    print("Generating similarity-colored plot for a specific query...")
    focus_query = "magic at Hogwarts"
    focus_embedding = model.encode([focus_query])[0]

    # Calculate similarities to the focus query
    similarities = cosine_similarity([focus_embedding], sample_embeddings)[0]

    # Plot with color based on similarity
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        umap_results[:max_samples, 0],
        umap_results[:max_samples, 1],
        c=similarities,
        cmap="viridis",
        s=30,
        alpha=0.7,
    )

    plt.colorbar(scatter, label='Similarity to "magic at Hogwarts"')

    # Highlight top 5 most similar chunks
    top5_indices = similarities.argsort()[-5:][::-1]
    plt.scatter(
        umap_results[top5_indices, 0],
        umap_results[top5_indices, 1],
        c="red",
        s=100,
        alpha=0.8,
        marker="x",
        label="Top 5 similar chunks",
    )

    plt.title(f'Chunk Similarity to "{focus_query}" Query', fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=14)
    plt.ylabel("UMAP Dimension 2", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("similarity_focus_query.png", dpi=300)
    plt.show()

    return "Visualizations complete!"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cache_chunks", action="store_true")
    parser.add_argument("--from_cache", type=str, default=None)
    args = parser.parse_args()
    # Load the sentence transformer model
    model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    if not args.from_cache:
        # Load the text file
        with open("harry_potter.txt", "r", encoding="utf-8") as file:
            text = file.read()

        # Split the text into passages (using sentences as passages)
        passages = sent_tokenize(text)
        # Visualize the embedding space
        # Create text chunks
        chunks = split_into_chunks(text, chunk_size=3)
        print(f"Created {len(chunks)} chunks from {len(passages)} sentences")
        # Generate embeddings for all chunks
        chunk_embeddings = model.encode(chunks)
        if args.cache_chunks:
            pickle.dump(chunk_embeddings, "cached_embedding.pkl")
    else:
        with open(args.from_cache, "rb") as f:
            chunk_embeddings = pickle.load(f)

    if args.plot:
        visualize_embeddings()

    # Run the same queries as with TF-IDF for comparison
    print("SEMANTIC SEARCH RESULTS")
    print(80 * "=")
    QUERY1 = "magic wand"
    QUERY2 = "Harry be careful!"
    QUERY3 = "Voldemort is here"
    search_queries(queries=[QUERY1, QUERY2, QUERY3], search_fn=semantic_search, top_k=5)
