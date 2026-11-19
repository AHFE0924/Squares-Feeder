"""Live sync for the Kaggle optimizer notebook, without bloating the repo.

Pushes your current best CSV to the branch "live" as a single, force-pushed commit, so the
repository does not grow by a full copy of the file on every push, and it never touches
latest_public.csv on main (that is the scraper's file, which merges "live" in on every run).
It only pushes when the score improved by at least `min_gain` and at most once per
`min_interval` seconds. Call it with force=True once at the end of the notebook so the last
improvement is not held back.

Usage in the notebook (internet on, a GitHub token with write access in a Kaggle secret):

    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    from kaggle_live_sync import live_sync
    live_sync("submission.csv", score, token)               # after an improvement
    live_sync("submission.csv", score, token, force=True)   # once, at the very end
"""
import os
import time
import shutil
import tempfile
import subprocess

_state = {"last_push": 0.0, "last_score": None}


def live_sync(csv_path, score, token, repo="AHFE0924/Squares-Feeder", branch="live",
              min_gain=1e-6, min_interval=1800, force=False, remote_url=None):
    """Returns True if it pushed. Never raises on a failed push (the optimizer keeps running)."""
    now = time.time()
    if not force:
        last = _state["last_score"]
        if last is not None and last - score < min_gain:
            return False
        if now - _state["last_push"] < min_interval:
            return False
    url = remote_url or f"https://x-access-token:{token}@github.com/{repo}.git"
    work = tempfile.mkdtemp()
    try:
        shutil.copy(csv_path, os.path.join(work, "live_best.csv"))

        def git(*args):
            subprocess.run(["git", *args], cwd=work, check=True, capture_output=True, text=True)

        git("init", "-q")
        git("checkout", "-q", "-b", branch)
        git("add", "live_best.csv")
        git("-c", "user.name=Kaggle-Live-Worker", "-c", "user.email=worker@kaggle.com",
            "commit", "-q", "-m", f"Live best: score {score:.6f}")
        git("push", "-q", "--force", url, f"HEAD:refs/heads/{branch}")
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").replace(token or "\0", "***") if token else (e.stderr or "")
        print(f"[live_sync] push failed: {msg.strip()[:300]}")
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _state.update(last_push=now, last_score=score)
    print(f"[live_sync] pushed score {score:.6f} to branch {branch}")
    return True
