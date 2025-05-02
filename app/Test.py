import pandas as pd

# Given emotion counts
emotion_counts = pd.Series({
    'netral': 4295,
    'senang': 301,
    'apresiasi': 234,
    'sedih': 96,
    'tenang': 76,
    'benci': 73,
    'antusias': 66,
    'takut': 56,
    'marah': 24
})

# Desired emotion order
emotion_order = [
    'anger', 'enthusiasm', 'happiness', 'calm', 'neutral', 'disgust', 'fear', 'appreciation', 'sadness'
]

# Reindex emotion_counts to match the emotion_order and fill missing values with the existing values from emotion_counts
emotion_counts = emotion_counts.reindex(emotion_order, inplace=True)

# Print the result
print(emotion_counts)