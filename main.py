from app.utils.comment_extract import extract_comments
from app.utils.BertTopic import # type: ignore
from app.utils. import # type: ignore

# Define the path to your JSON file
file_path = './dataset/yt_data.json'

# Input
video_id = input()

# Call the function to extract the comments
comments_df = extract_comments(file_path)

# Excecution Sentiment
sentiment_result = topic(video_id)# type: ignore

# Excecution IndoBert Topic
topic_result = topic(video_id)# type: ignore