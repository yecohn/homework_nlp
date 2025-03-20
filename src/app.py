import streamlit as st
from typing import Literal
import requests


# Configure page
st.set_page_config(
    page_title="Harry Potter Semantic Search", page_icon="🔍", layout="wide"
)


def search_text(
    query: str, top_k: int = 5, method: Literal["semantic", "tfidf"] = "semantic"
):
    """
    Send search request to FastAPI server
    """
    url = f"http://localhost:8000/search/{method}"

    # Prepare the request payload
    payload = {"q": query, "top_k": top_k}

    try:
        response = requests.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to server: {str(e)}")
        return None


def main():
    # Title and description
    st.title("🔍 Harry Potter Semantic Search")
    st.markdown(
        """
    Search through Harry Potter text using semantic search.
    The search understands context and meaning, not just exact matches.
    """
    )

    # Create two columns for input
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        # Search query input
        query = st.text_input(
            "Enter your search query", placeholder="Example: magic wand at Hogwarts"
        )

    with col2:
        # Number of results selector
        top_k = st.number_input("Number of results", min_value=1, max_value=20, value=5)

    with col3:
        # Search method selector
        method = st.selectbox(
            "Search method",
            ["semantic", "tfidf"],
            index=0,
        )

    # Search button
    if st.button("🔍 Search"):
        if query:
            with st.spinner("Searching..."):
                results = search_text(query, top_k, method)

                if results:
                    st.subheader("Search Results")

                    # Display results in cards
                    for i, result in enumerate(results["results"], 1):
                        with st.container():
                            st.markdown(
                                f"""
                            ---
                            ### Result {i}
                            **Passage:** {result['passage']}
                            
                            **Relevance Score:** {result['score']:.4f}
                            """
                            )

                    # Display metadata
                    st.sidebar.markdown("### Search Metadata")
                    st.sidebar.json(
                        {
                            "query": results["query"],
                            "search_type": results["search_type"],
                            "total_results": len(results["results"]),
                        }
                    )
        else:
            st.warning("Please enter a search query")


if __name__ == "__main__":
    main()
