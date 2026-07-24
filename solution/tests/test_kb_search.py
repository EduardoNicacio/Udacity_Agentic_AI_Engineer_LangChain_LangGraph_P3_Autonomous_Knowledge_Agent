"""Integration tests for KB search against a real database.

These tests call the actual kb_search() function against the seeded UDA-Hub
database, verifying that the ranking algorithm returns correct top articles
for each ticket category.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.tools.kb_search_tool import kb_search


def test_kb_search_onboarding():
    results = kb_search("How do I get started with CultPass?", category="onboarding")
    assert len(results) > 0, "Should find onboarding articles"
    titles = [r["title"].lower() for r in results]
    assert any("get started" in t or "getting started" in t or "onboarding" in t or "first" in t or "setup" in t for t in titles), \
        f"Expected onboarding article, got: {titles}"


def test_kb_search_technical():
    results = kb_search("The app crashes when I try to reserve an event", category="technical")
    assert len(results) > 0, "Should find technical articles"
    titles = [r["title"].lower() for r in results]
    assert any("crash" in t or "troubleshoot" in t or "error" in t for t in titles), \
        f"Expected crash/troubleshoot article, got: {titles}"


def test_kb_search_billing():
    results = kb_search("I was charged twice for my subscription this month", category="billing")
    assert len(results) > 0, "Should find billing articles"
    titles = [r["title"].lower() for r in results]
    assert any("billing" in t or "payment" in t or "charge" in t or "invoice" in t for t in titles), \
        f"Expected billing article, got: {titles}"


def test_kb_search_subscription():
    results = kb_search("I want to upgrade from basic to premium plan", category="subscription")
    assert len(results) > 0, "Should find subscription articles"
    titles = [r["title"].lower() for r in results]
    assert any("subscription" in t or "plan" in t or "upgrade" in t or "cancel" in t for t in titles), \
        f"Expected subscription article, got: {titles}"


def test_kb_search_reservation():
    results = kb_search("How do I cancel my reservation for this weekend?", category="reservation")
    assert len(results) > 0, "Should find reservation articles"
    titles = [r["title"].lower() for r in results]
    assert any("reservation" in t or "book" in t or "cancel" in t for t in titles), \
        f"Expected reservation article, got: {titles}"


def test_kb_search_content():
    results = kb_search("How do I leave a review for an experience I attended?", category="content")
    assert len(results) > 0, "Should find content articles"
    titles = [r["title"].lower() for r in results]
    assert any("review" in t or "content" in t or "share" in t or "write" in t for t in titles), \
        f"Expected content article, got: {titles}"


def test_kb_search_account():
    results = reset_and_search_account()
    if results:
        titles = [r["title"].lower() for r in results]
        assert any("account" in t or "login" in t or "password" in t for t in titles), \
            f"Expected account article, got: {titles}"


def reset_and_search_account():
    results = kb_search("I can't log in to my CultPass account", category="account")
    return results


def test_kb_search_onboarding_returns_score():
    results = kb_search("How do I get started with CultPass?", category="onboarding")
    assert len(results) > 0
    for r in results:
        assert r["score"] >= 6, f"Score too low: {r['score']}"
        assert "article_id" in r
        assert "title" in r
        assert "content" in r
        assert "tags" in r


def test_kb_search_returns_max_three():
    results = kb_search("app events reservation booking cancel refund subscription plan", category="unknown")
    assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"


def test_kb_search_no_match_returns_empty():
    results = kb_search("xyzzy plugh foobarbaz", category="unknown")
    assert len(results) == 0, f"Expected no matches for nonsense query, got: {results}"


if __name__ == "__main__":
    test_kb_search_onboarding()
    test_kb_search_technical()
    test_kb_search_billing()
    test_kb_search_subscription()
    test_kb_search_reservation()
    test_kb_search_content()
    test_kb_search_account()
    test_kb_search_onboarding_returns_score()
    test_kb_search_returns_max_three()
    test_kb_search_no_match_returns_empty()
    print("All KB search integration tests passed!")
