import os
import pandas as pd
import warnings
import csv


USERS_FILE = "users.csv"


def resolve_users_csv(base_dir, dataset_entry):
    candidates = [
        os.path.join(base_dir, dataset_entry),
        os.path.join(base_dir, dataset_entry, USERS_FILE),
        os.path.join(base_dir, dataset_entry, dataset_entry, USERS_FILE),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def load_all_data(dataset_map, base_dir):
    frames = []
    missing_files = []

    for dataset_entry, label in dataset_map.items():
        path = resolve_users_csv(base_dir, dataset_entry)

        if path is None:
            missing_files.append(dataset_entry)
            continue

        df = pd.read_csv(
            path,
            encoding="utf-8",
            on_bad_lines="skip",
            low_memory=False,
        )

        df["label"] = int(label)
        df["source_file"] = dataset_entry
        frames.append(df)

        print(f"Loaded {dataset_entry}: {len(df):,} rows")

    if missing_files:
        print("\nMissing dataset entries:")
        for p in missing_files:
            print("-", p)

    if not frames:
        raise ValueError("No dataset files were loaded.")

    all_data = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows loaded: {len(all_data):,}")

    return all_data


def load_tweets(base_dir, dataset_entry, TWEET_FEATURES):
    candidates = [
        os.path.join(base_dir, dataset_entry, "tweets.csv"),
        os.path.join(base_dir, dataset_entry, dataset_entry, "tweets.csv"),
    ]

    for path in candidates:
        if os.path.isfile(path):

            available = ["user_id"] + TWEET_FEATURES

            chunks = []
            for chunk in pd.read_csv(
                path,
                usecols=lambda c: c in available,
                chunksize=100_000,
                encoding="utf-8",
                on_bad_lines="skip"
            ):
                chunks.append(chunk)

            dataframe = pd.concat(chunks, ignore_index=True)

            agg_cols = [c for c in TWEET_FEATURES if c in dataframe.columns]
            dataframe[agg_cols] = dataframe[agg_cols].apply(pd.to_numeric, errors="coerce")

            agg_cols = [c for c in agg_cols if pd.api.types.is_numeric_dtype(dataframe[c])]

            if not agg_cols:
                return None

            return dataframe.groupby("user_id")[agg_cols].mean().reset_index()

    return None


def build_raw_dataset(DATASETS, BASE_DIR, TWEET_FEATURES):
    raw = load_all_data(DATASETS, BASE_DIR)

    tweet_aggs = []
    for dataset_entry in DATASETS:
        agg = load_tweets(BASE_DIR, dataset_entry, TWEET_FEATURES)
        if agg is not None:
            tweet_aggs.append(agg)

    if tweet_aggs:
        all_tweets = pd.concat(tweet_aggs, ignore_index=True)

        if "id" in raw.columns:
            raw = raw.rename(columns={"id": "user_id"})

        raw = raw.merge(all_tweets, on="user_id", how="left")

    return raw


warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
