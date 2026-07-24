"""FastMCP server for searching the CultPass knowledge base.

Input: ticket_text (str), category (str), account_id (str)
Output: list of matching articles with scores

Ranking algorithm:
    1. Tokenize ticket text into keywords (stop words removed).
    2. Extract 2-word phrases for phrase-level matching.
    3. Stem keywords by stripping common English suffixes.
    4. Score articles by: tag match (4pts), title match (5pts), content match (2pt),
       phrase match in tags (8pts), phrase match in title (10pts),
       category alignment (4pts), exact word boundary bonus (3pts),
       negative penalty for mismatched category (-3pts).
    5. Return top 3 articles with score >= 6.
"""

import re
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

_SUFFIXES = ["tion", "sion", "ment", "ness", "ing", "ful", "less", "able", "ible", "ous", "ive", "ly", "ed", "er", "est", "al", "ity"]


def _stem(word: str) -> str:
    """Simple suffix-stripping stemmer for ranking purposes."""
    w = word.lower()
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def _tokenize(text: str) -> list[str]:
    """Extract keywords from text, removing stop words."""
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in raw if w not in _STOP_WORDS and len(w) > 2]


def _phrases(keywords: list[str], n: int = 2) -> list[str]:
    """Extract contiguous n-word phrases from keyword list."""
    return [" ".join(keywords[i:i+n]) for i in range(len(keywords) - n + 1)]


def _word_boundary_match(pattern: str, text: str) -> bool:
    """Check if pattern appears as a whole word in text."""
    return bool(re.search(r'\b' + re.escape(pattern) + r'\b', text))


CATEGORY_KEYWORDS = {
    "billing": ["charge", "payment", "invoice", "refund", "subscription", "price", "cost", "bill", "credit", "debit"],
    "account": ["login", "password", "email", "profile", "settings", "account", "signup", "register", "verify"],
    "technical": ["crash", "error", "bug", "slow", "freeze", "load", "app", "update", "install", "download"],
    "subscription": ["cancel", "renew", "upgrade", "downgrade", "plan", "tier", "subscription", "member"],
    "reservation": ["book", "reserve", "cancel", "change", "date", "time", "slot", "reservation", "visit"],
    "content": ["article", "guide", "help", "tutorial", "video", "blog", "content", "news", "discover"],
    "onboarding": ["start", "begin", "first", "setup", "tutorial", "welcome", "new", "intro"],
}


@mcp.tool()
def kb_search(ticket_text: str, category: str = "unknown", account_id: str = "cultpass") -> list[dict]:
    """Search the Knowledge table using keyword matching on tags and content.

    Only returns articles with a minimum score of 6 to ensure relevance.

    Args:
        ticket_text: The customer ticket text to search for.
        category: The classified ticket category.
        account_id: The account ID to scope the search to.

    Returns:
        A list of matching articles with scores, sorted by relevance.
    """
    keywords = _tokenize(ticket_text)
    phrases = _phrases(keywords)

    if not keywords:
        return []

    stemmed_keywords = [_stem(kw) for kw in keywords]

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
            combined_text = f"{tag_str} {title_str} {content_str}"

            # Single keyword scoring
            for kw, stemmed_kw in zip(keywords, stemmed_keywords):
                # Tag matches (highest signal)
                if _word_boundary_match(kw, tag_str) or stemmed_kw in tag_str:
                    score += 4
                # Title matches (strong signal)
                if _word_boundary_match(kw, title_str) or stemmed_kw in title_str:
                    score += 5
                # Content matches (supporting signal)
                if _word_boundary_match(kw, content_str) or stemmed_kw in content_str:
                    score += 2

            # Phrase matching (very strong signal)
            for phrase in phrases:
                if phrase in tag_str:
                    score += 8
                if phrase in title_str:
                    score += 10
                if phrase in content_str:
                    score += 4

            # Category alignment bonus
            if category and category != "unknown":
                if category in tag_str:
                    score += 4
                elif category in title_str:
                    score += 3
                else:
                    # Check if article category matches ticket category via keyword mapping
                    ticket_cat_keywords = CATEGORY_KEYWORDS.get(category, [])
                    article_text = f"{tag_str} {title_str}"
                    if any(kw in article_text for kw in ticket_cat_keywords):
                        score += 2
                    else:
                        # Mild penalty for category mismatch
                        score -= 3

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
            for s, a in top if s >= 6
        ]


if __name__ == "__main__":
    mcp.run()
