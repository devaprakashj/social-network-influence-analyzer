"""
data_fetcher.py
===============
Fetches REAL social network data from public APIs.
- GitHub follower graph  (no API key required, 60 req/hr free)
- Reddit commenter graph (no API key required, public JSON API)

All fetchers are GENERATORS → they yield status dicts so the
Streamlit UI can update in real-time as data streams in.

Yield format:
  { "type": "status"|"node"|"edge"|"done"|"error",
    "msg": str, "G": nx.Graph | None }
"""

import time
import requests
import networkx as nx
from typing import Generator, Dict, Any

HEADERS = {"User-Agent": "SocialNetworkAnalyzer/1.0 (mini-project)"}


# ─── GitHub Follower Network ──────────────────────────────────────────────────

def fetch_github_network(
    seed_user: str,
    depth: int = 2,
    max_nodes: int = 80,
    github_token: str = "",
) -> Generator[Dict[str, Any], None, None]:
    """
    BFS from `seed_user` through GitHub follower graph.
    Each API call is yielded immediately so the UI streams live.
    """
    G = nx.Graph()
    headers = dict(HEADERS)
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    visited   = set()
    queue     = [(seed_user, 0)]
    api_calls = 0

    def _get_user(username: str):
        nonlocal api_calls
        try:
            r = requests.get(
                f"https://api.github.com/users/{username}",
                headers=headers, timeout=8
            )
            api_calls += 1
            if r.status_code == 403:
                return None, "rate_limit"
            if r.status_code == 404:
                return None, "not_found"
            r.raise_for_status()
            return r.json(), None
        except Exception as e:
            return None, str(e)

    def _get_followers(username: str, per_page: int = 30):
        nonlocal api_calls
        try:
            r = requests.get(
                f"https://api.github.com/users/{username}/followers",
                headers=headers, timeout=8,
                params={"per_page": per_page}
            )
            api_calls += 1
            if r.status_code == 403:
                return None, "rate_limit"
            r.raise_for_status()
            return [u["login"] for u in r.json()], None
        except Exception as e:
            return None, str(e)

    yield {"type": "status", "msg": f"🚀 Starting BFS from **{seed_user}** (depth={depth}, max={max_nodes} nodes)", "G": G}

    while queue and len(G.nodes()) < max_nodes:
        username, current_depth = queue.pop(0)
        if username in visited:
            continue
        visited.add(username)

        # Fetch user profile
        user_data, err = _get_user(username)
        if err == "rate_limit":
            yield {"type": "error", "msg": "⚠️ GitHub rate limit hit (60 req/hr). Add a token to increase to 5000/hr.", "G": G}
            break
        if err or user_data is None:
            continue

        # Add node
        G.add_node(username,
                   username  = username,
                   platform  = "GitHub",
                   followers = user_data.get("followers", 0),
                   posts     = user_data.get("public_repos", 0),
                   name      = user_data.get("name") or username,
                   avatar    = user_data.get("avatar_url", ""),
                   bio       = (user_data.get("bio") or "")[:80])

        yield {"type": "node",
               "msg": f"✅ Added **{username}** — {user_data.get('followers',0):,} followers, "
                      f"{user_data.get('public_repos',0)} repos",
               "G": G}

        if current_depth >= depth:
            continue

        # Fetch followers
        foll_limit = min(20, max_nodes - len(G.nodes()))
        if foll_limit <= 0:
            break

        followers, err = _get_followers(username, per_page=foll_limit)
        if err == "rate_limit":
            yield {"type": "error", "msg": "⚠️ GitHub rate limit. Consider adding a Personal Access Token.", "G": G}
            break
        if err or followers is None:
            continue

        yield {"type": "status",
               "msg": f"🔗 Found {len(followers)} followers of **{username}**",
               "G": G}

        for follower in followers:
            if len(G.nodes()) >= max_nodes:
                break
            G.add_edge(username, follower)
            if follower not in visited:
                queue.append((follower, current_depth + 1))

        time.sleep(0.05)  # polite delay

    yield {"type": "done",
           "msg": f"🎉 Done! Graph has **{G.number_of_nodes()} nodes** and "
                  f"**{G.number_of_edges()} edges**. API calls used: {api_calls}.",
           "G": G}


# ─── Reddit Commenter Network ─────────────────────────────────────────────────

def fetch_reddit_network(
    subreddit: str = "Python",
    post_limit: int = 20,
    min_comments: int = 2,
) -> Generator[Dict[str, Any], None, None]:
    """
    Build a network where:
      - Nodes = Reddit users who commented in top posts
      - Edges = two users commented in the same post
    """
    G = nx.Graph()
    base = "https://www.reddit.com"
    headers = dict(HEADERS)

    yield {"type": "status",
           "msg": f"🔍 Fetching top {post_limit} posts from **r/{subreddit}**…",
           "G": G}

    try:
        r = requests.get(
            f"{base}/r/{subreddit}/hot.json",
            headers=headers, timeout=10,
            params={"limit": post_limit}
        )
        r.raise_for_status()
        posts = r.json()["data"]["children"]
    except Exception as e:
        yield {"type": "error", "msg": f"❌ Could not fetch posts: {e}", "G": G}
        return

    yield {"type": "status", "msg": f"📬 Found {len(posts)} posts. Fetching comments…", "G": G}

    post_commenters = {}  # post_id → list of commenters

    for i, post in enumerate(posts):
        p     = post["data"]
        pid   = p["id"]
        title = p["title"][:60]

        try:
            cr = requests.get(
                f"{base}/r/{subreddit}/comments/{pid}.json",
                headers=headers, timeout=10,
                params={"limit": 80, "depth": 2}
            )
            cr.raise_for_status()
            data = cr.json()
        except Exception as e:
            yield {"type": "status", "msg": f"⚠️ Skipped post {pid}: {e}", "G": G}
            time.sleep(0.5)
            continue

        commenters = _extract_commenters(data)
        if len(commenters) < min_comments:
            time.sleep(0.5)
            continue

        post_commenters[pid] = commenters
        yield {"type": "status",
               "msg": f"📝 Post {i+1}/{len(posts)}: *{title}* → {len(commenters)} unique commenters",
               "G": G}

        # Add nodes
        for user in commenters:
            if user not in G:
                G.add_node(user,
                           username = user,
                           platform = f"Reddit/r/{subreddit}",
                           followers= 0,
                           posts    = 0)
                yield {"type": "node",
                       "msg": f"👤 Added user **u/{user}**",
                       "G": G}

        # Add edges (co-commenter links)
        for j in range(len(commenters)):
            for k in range(j + 1, len(commenters)):
                u, v = commenters[j], commenters[k]
                if G.has_edge(u, v):
                    G[u][v]["weight"] = G[u][v].get("weight", 1) + 1
                else:
                    G.add_edge(u, v, weight=1)

        time.sleep(0.6)  # reddit rate limit: ~1 req/sec

    # Remove isolated nodes (users who appeared in only 1 post, no co-commenter)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    yield {"type": "done",
           "msg": f"🎉 Reddit network complete! **{G.number_of_nodes()} users**, "
                  f"**{G.number_of_edges()} co-comment links** across {len(post_commenters)} posts.",
           "G": G}


def _extract_commenters(data: list, max_users: int = 80) -> list:
    """Recursively pull unique (non-deleted) commenter names from a Reddit thread."""
    users = set()

    def _walk(items):
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            d = item.get("data", {})
            # Top-level listing
            if "children" in d:
                _walk(d["children"])
                continue
            author = d.get("author", "")
            if author and author not in ("[deleted]", "[removed]", "AutoModerator"):
                users.add(author)
            # Recurse into replies
            replies = d.get("replies", "")
            if isinstance(replies, dict):
                _walk(replies.get("data", {}).get("children", []))

    _walk(data)
    return list(users)[:max_users]
