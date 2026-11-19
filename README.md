# Squares-Feeder
This is used to pull latest code from the Squares in Squares competition in Kaggle.

- `latest_public.csv` – best valid packing for every n (1..1000) found in public notebooks, your
  own notebooks and the `live`, `live-1`, `live-2` and `live-3` branches. Safe to submit as is: ids run 0..500499, angles are in
  radians, and every pair of squares is at least 1e-10 apart.
- `scrapescrapescrape.py` – the scraper; GitHub Actions runs it every 4 hours
  (`.github/workflows/feed.yml`). `python scrapescrapescrape.py --merge-only --extra file.csv`
  merges local files without touching Kaggle.
- `kaggle_live_sync.py` – call this from the Kaggle optimizer notebook instead of pushing
  `latest_public.csv` to main. Each of the three shard notebooks force-pushes one commit to its own
  branch (`live-1`, `live-2`, `live-3`), at most every 30 minutes, so the repo stops growing by 28 MB
  per push.
