"""FastMCP server for searching the CultPass knowledge base.

Input: ticket_text (str), category (str), account_id (str)
Output: list of matching articles with scores
"""

from pathlib import Path
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.models import udahub as udahub_models

mcp = FastMCP("kb_search")

_BASE_DIR = Path(__file__).resolve().parent.parent.parent

_STOP_WORDS = {
    "the", "a", "an", "is", "was", "to", "for", "of", "in", "it", "on", "at",
    "my", "i", "me", "can", "not", "do", "does", "have", "has", "this", "that",
    "with", "from", "be", "been", "and", "how", "what", "why", "but", "or",
    "are", "were", "being", "would", "could", "should", "your", "you", "they",
    "them", "their", "all", "any", "each", "some", "than", "then", "just",
    "about", "also", "very", "will", "which", "who", "when", "where",
}


@mcp.tool()
def kb_search(ticket_text: str, category: str = "unknown", account_id: str = "cultpass") -> list[dict]:
    """Search the Knowledge table using keyword matching on tags and content.

    Only returns articles with a minimum score of 4 to ensure relevance.

    Args:
        ticket_text: The customer ticket text to search for.
        category: The classified ticket category.
        account_id: The account ID to scope the search to.

    Returns:
        A list of matching articles with scores, sorted by relevance.
    """
    keywords = [w.lower().strip(".,!?;:'\"") for w in ticket_text.split()
                if w.lower().strip(".,!?;:'\"") not in _STOP_WORDS and len(w) > 3]

    if not keywords:
        return []

    udahub_db = str(_BASE_DIR / "data" / "core" / "udahub.db")
    engine = create_engine(f"sqlite:///{udahub_db}", echo=False)

    with sessionmaker(bind=engine)() as session:
        articles = session.query(udahub_models.Knowledge).filter(
            udahub_models.Knowledge.account_id == account_id
        ).all()

        scored = []
        for article in articles:
            score = 0
            tag_str = (article.tags or "").lower()
            content_str = (article.content or "").lower()
            title_str = (article.title or "").lower()

            for kw in keywords:
                if kw in tag_str:
                    score += 3
                if kw in title_str:
                    score += 2
                if kw in content_str:
                    score += 1

            if category != "unknown" and category in tag_str:
                score += 2

            scored.append((score, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]
        return [
            {
                "article_id": a.article_id,
                "title": a.title,
                "content": a.content,
                "tags": a.tags,
                "score": s,
            }
            for s, a in top if s >= 4
        ]


if __name__ == "__main__":
    mcp.run()
