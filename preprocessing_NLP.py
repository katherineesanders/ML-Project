from preprocessing import load_all_data


def resolve_tweets_csv(base_dir, dataset_entry, TWEETS_FILE):
    candidates = [
        os.path.join(base_dir, dataset_entry),
        os.path.join(base_dir, dataset_entry, TWEETS_FILE),
        os.path.join(base_dir, dataset_entry, dataset_entry, TWEETS_FILE),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_tweets_nlp(base_dir, dataset_entry, TWEET_FEATURES, TWEETS_FILE,
                    chunksize=50000, max_text_chars=500):

    path = resolve_tweets_csv(base_dir, dataset_entry, TWEETS_FILE)
    if path is None:
        return None

    agg_data = {}
    text_data = {}

    try:
        reader = pd.read_csv(
            path,
            chunksize=chunksize,
            encoding="latin-1",
            on_bad_lines="skip",
            low_memory=False
        )

        for chunk in reader:
            if "user_id" not in chunk.columns:
                continue

            chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce")

            # ---- numeric aggregation ----
            for col in TWEET_FEATURES:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            for uid, grp in chunk.groupby("user_id"):

                # numeric features
                if uid not in agg_data:
                    agg_data[uid] = {c: [] for c in TWEET_FEATURES}

                for c in TWEET_FEATURES:
                    if c in grp.columns:
                        agg_data[uid][c].extend(grp[c].dropna().tolist())

                # text
                if "text" in grp.columns:
                    existing = text_data.get(uid, "")
                    if len(existing) < max_text_chars:
                        new_text = " ".join(grp["text"].dropna().astype(str))
                        text_data[uid] = (existing + " " + new_text)[:max_text_chars]

    except Exception:
        return None

    if not agg_data and not text_data:
        return None

    rows = []
    all_ids = set(agg_data) | set(text_data)

    for uid in all_ids:
        row = {"user_id": uid}

        if uid in agg_data:
            for c, vals in agg_data[uid].items():
                row[c] = sum(vals) / len(vals) if vals else float("nan")

        if uid in text_data:
            row["text"] = text_data[uid]

        rows.append(row)

    return pd.DataFrame(rows)


def build_raw_dataset_nlp(DATASETS, BASE_DIR, TWEET_FEATURES, TWEETS_FILE):
    raw = load_all_data(DATASETS, BASE_DIR)

    tweet_aggs = []
    for dataset_entry in DATASETS:
        agg = load_tweets_nlp(BASE_DIR, dataset_entry, TWEET_FEATURES, TWEETS_FILE)
        if agg is not None:
            tweet_aggs.append(agg)

    if tweet_aggs:
        all_tweets = pd.concat(tweet_aggs, ignore_index=True)

        raw["id"] = pd.to_numeric(raw["id"], errors="coerce")
        all_tweets["user_id"] = pd.to_numeric(all_tweets["user_id"], errors="coerce")

        raw = raw.merge(all_tweets, left_on="id", right_on="user_id", how="left")

    return raw