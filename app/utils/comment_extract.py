import os
import json
import re
import pandas as pd
import polars as pl
import datetime
import string

# Load stopwords function
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())  # Convert to set for faster lookup
    return stopwords

# Get the current directory of the script and adjust the path
script_dir = os.path.dirname(os.path.realpath(__file__))
stopwords_path = os.path.join(script_dir, '..', 'utils', 'stopwords.txt')
stopwords = load_stopwords(stopwords_path)

replace_pattern = '[\s\W_]+http\S+[\s\W_]+'  # Tautan
replace_pattern += '|[\s\W_]+\w[\s\W_]+'  # Satu karakter
replace_pattern += '|[^\x00-\x7F]+'  # Non ASCII
replace_pattern += '|\s+'  # Useless space

# Extract comments function and apply the text cleaning to comments and replies
def extract_comments(file_path):
    # Load the data from the provided JSON file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract relevant information and structure it for DataFrame
    comments_data = []
    
    # Loop through each video and extract comments
    for video in data:
        # video_id = video['video_id']
        # video_title = video['video_title']
        # video_description = video['video_description']
        
        # Extract the main comment and clean it
        main_comment = {
            # 'video_id': video_id,
            # 'video_title': video_title,
            # 'video_description': video_description,
            'author': video['author'],
            'comment': video['comment'],
            # 'comment2': video['comment'],  # Fixing to get the correct comment
            # 'published_at': video['published_at'],
            # 'reply_count': video['reply_count']
        }
        
        # Append the main comment to the data list
        comments_data.append(main_comment)
        
        # # Extract replies to the main comment and clean them
        # for reply in video['replies']:
        #     reply_info = {
        #         # 'video_id': video_id,
        #         # 'video_title': video_title,
        #         # 'video_description': video_description,
        #         # 'author': reply['author'],
        #         'comment': reply['comment'],  # Fixing to get the correct comment
        #         # 'comment2': reply['comment'],  # Fixing to get the correct comment
        #         # 'published_at': reply['published_at'],
        #         # 'reply_count': 0  # Replies don't have further replies, so setting it to 0
        #     }
        #     comments_data.append(reply_info)

    # Create a Polars DataFrame from the list of dictionaries
    df_filtered = pl.DataFrame(comments_data)

    # Remove duplicates in 'comment' column
    df_filtered = df_filtered.unique(subset=['comment'], maintain_order=True, keep='first')

    stopword_list = [' dan ', ' atau ',
                ' serta ', ' yaitu ', ' yakni ', ' adalah ', ' ialah ', 'merupakan ',]
    
    df_filtered = df_filtered.with_columns(
        pl.col('comment').str.to_lowercase().str.replace_many(stopword_list, ' ')
        .str.replace_all(replace_pattern, ' ').str.strip_chars().str.replace(r'\s+', ' ').
        str.replace(r'[{}]+'.format(re.escape(string.punctuation)), '')
    ).clone()

    # Return the filtered DataFrame
    return df_filtered.to_pandas()