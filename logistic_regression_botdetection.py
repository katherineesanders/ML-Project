import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = "datasets_full.csv/datasets_full.csv"
 
DATASETS = {
   "genuine_accounts.csv": 0, 
   "social_spambots_1.csv": 1, 
   "social_spambots_2.csv": 1,
   "social_spambots_3.csv": 1,
   "traditional_spambots_1.csv": 1,
   "traditional_spambots_2.csv": 1,
   "traditional_spambots_3.csv": 1,
   "traditional_spambots_4.csv": 1,
}

# statuses_count	followers_count	friends_count	favourites_count	listed_count	
# url	lang	time_zone	location	default_profile	default_profile_image	
# geo_enabled	profile_image_url	profile_banner_url	profile_use_background_image	
# profile_background_image_url_https	profile_text_color	profile_image_url_https	profile_sidebar_border_color	\
# profile_background_tile	profile_sidebar_fill_color	profile_background_image_url	
# profile_background_color	profile_link_color	utc_offset	is_translator	follow_request_sent	
# protected	verified	notifications	description	contributors_enabled	
# following	created_at	timestamp	crawled_at	updated

USER_FEATURES = [
   "statuses_count", "followers_count", "friends_count",
   "favourites_count", "listed_count", "default_profile",
   "default_profile_image", "verified", "created_at"
]

# id	text	source	user_id	truncated	in_reply_to_status_id	
# in_reply_to_user_id	in_reply_to_screen_name	retweeted_status_id	
# geo	place	contributors	retweet_count	reply_count	favorite_count	
# favorited	retweeted	possibly_sensitive	num_hashtags	num_urls	
# num_mentions	created_at	timestamp	crawled_at	updated

TWEET_FEATURES = [
   "text", "reply_count", "favorite_count", "num_urls", "num_mentions"
]

# load data through chunks due to size 
def load_data(path, usecols=None, chunksize=None):
   if chunksize:
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols,
                            chunksize=chunksize,
                            encoding="utf-8", on_bad_lines="skip"):
        chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.read_csv(path, usecols=usecols, encoding="utf-8", on_bad_lines="skip")


