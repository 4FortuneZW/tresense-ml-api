# Import necessary libraries
import os
import re
import torch
import pandas as pd
import polars as pl
from tqdm.auto import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap.umap_ import UMAP
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
import datamapplot
from utils.comment_extract import extract_comments
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3 

import io
from google.cloud import storage
from datetime import datetime, timezone


# Set display options for pandas
pd.set_option('display.max_rows', None)

# Set the device for model loading (use CUDA if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to load the comment data
def load_comment_data(file_path='./app/dataset/yt_data2.json', video_id=None):
    """Load the comment data from the specified file path."""
    comments_df = extract_comments(file_path)
    return comments_df

# Mendefinisikan fungsi klasifikasi
def text_emotion_classification(text: str) -> str:
    emotions_keywords = {
        # Kamus kata kunci emosi berdasarkan SenticNet 8 knowledge base

        "sadness": [
            "sedih", "pilu", "murung", "nangis", "melankolis", "duka",
            "tertekan", r'patah hati', "duka", "terharu", r'hati hancur',
            "kesedihan", "terpuruk", r'menyayat hati', "nestapa", r'menggugah hati',
            "sedih", "kecewa", "patah hati", "murung", "terpuruk", "kesepian",
            "merana", "galau", "menangis", "depresi", "frustrasi", "sakit hati",
            "penembakan", 'utang', 'iri', 'pedih', 'penipu', ' [nt]ipu ',
            'simpati','anomali', 'basi', 'bau', 'berdosa', 'dusta', 'bohong',
            'muram', 'buronan', 'celaka', 'cemburu', 'derita', 'dicuri',
            'gagal', 'kalah', 'putus asa', 'suram', 'sulit', r'tidak\s*mampu',
            'korban', 'krisis', 'kritik', 'masam', 'cengeng', 'siksa', 'ratapi',
            'negatif', 'noda', 'parasit', 'sesal', 'nyesal', 'nyesel', 'pesimis',
            'pucat', 'rapuh', 'rendah', 'sendirian', 'kesepian', 'tekanan', 'tertinggal',
            'tuduhan', 'tuli', 'buta', 'terbuang', 'tertinggal',
        ],
        "disgust": [
            "jijik", "muak", "benci", "mual", "jengah", "menjijikkan", "ngeri",
            "terganggu", r'tidak\s*enak', "menyebalkan", "memualkan", "muntah", "menyampah",
            "enyek", "ngeri",
            "hina", "menyebalkan", "kurang ajar", "sampah",
            "jelek", "rasis", 'menjelekkan', 'angkuh', 'babi', 'bajingan', 'bodoh',
            'buruk', 'busuk', 'cacat', 'cemooh', 'cerca', 'egois', 'gendut', 'jahat',
            'kafir', 'keji', 'kikir', 'konyol', 'pelit', 'korup', 'malas', 'malu',
            'manja', 'lusuh', 'nakal', 'palsu', 'mustahil', 'menyimpang', 'menyalahkan',
            'khianat', 'gila', 'naif', 'pengecut', 'remeh', 'rasis', 'sarkas', 'sembrono',
            'sempit', 'serakah', 'sekandal', 'sombong',

            "goblok", "kampret", "kontol", "anjing", " taik? ", 'janc[ou]k', 'tolol',
            "jembut", "diam", 'bosan',
        ],
        "anger": [
            "marah", "murka", "kesal", "geram", "emosi", "berang",
            r'naik\s*darah', "amarah", "kemarahan", r'naik\s*pitam', "gusar", "mengamuk", 'amukan',
            "beringas", "singgung", "dendam",
            "meradang", "mengguntur", r"panas\s*hati", "tersulut", r"tidak\s*puas",
            "marah", "kesal", "geram", "dongkol",
            "nyeb[ae]l", "jengkel", "seb[ae]l",
            "ngegas",

            'antagonis', 'bersaing', 'cekcok', 'cacian', 'cemberut', 'idiot', 'ganas',
            'gila', 'cibir', 'ngomel', 'sialan', 'berapi',

        ],
        "fear": [
            "takut", "cemas", "gelisah", "kh?awatir", "seram", "serem", "ngeri", "ketakutan",
            "panik", r'was\s*was', "gemet[ae]r", "kuatir", "mencekam", "terr?or",
            "paranoid", "parno", "merinding", "[pf]obia", "takut", "khawatir",
            "ancam", "menyeramkan", "hor?ror", "ngeri", "panik", "tegang", "tersangka", 'bahaya',
            'bencana', 'berdarah', 'diktator', 'gugup', 'histeri', 'mundur', 'ragu',
            'penjara', 'punah', 'rentan', 'resah', 'pusing', 'siksa', 'skeptis', 'tegang',
            'trauma', 'bersalah', 'waspada',
        ],
        "happiness": [
            "bahagia", "senang", "gembira", "riang", "ceria", "tawa", r"suka\s*cita",
            "terhibur", "bersorak", "euforia", "kesenangan", "keceriaan", r'suka\s*ria', "kegirangan",
            "menyenangkan", "seru", r'mood\s*booster', "tertawa", "lucu",
            "asy?ik", "semangat", "merinding", "excited", "hibur", 'hepi', 'happy', 'cantik',
            'cerdas', 'cermat', 'dewasa', 'efektif', 'ekonomis', 'enak', 'gagah', 'gratis',
            'gurih', 'hadiah', 'ideal', ' imut ', 'jenius', 'ajaib', 'kekasih', 'kompak',
            'meriah', 'menang', 'nikmat', 'ridh?o', 'unggul', 'gelitik', 'mewah', 'modern',
            'mudah', 'mujarab', 'optimis', 'optimal', 'pantas', 'pembaruan', 'pembebasan',
            'gemar', 'positif', 'prestasi', 'sanjung', 'romantis', 'segar', 'surga', 'senyum',
            'lawak', 'pesta', 'lelucon',
        ],
        "calm": [
            "lega", "tenang", "bebas", "terlepas", "ketenangan", "kelegaan", r'hilang\s*beban',
            "damai", "sejuk", "hening", "nyaman", "tentram", "stabil", "santai", 'relax',
            "teratur", "terkendali", "kalem", "rileks?",  "calm", "mimpi", "syukur",
            'terima\s*kasih', 'empati', 'kekal', 'bajik', 'luwes', 'mampu', 'megah', 'sehat',
            'sabar', 'jamin', 'janji', 'alha?mdulillah',
        ],
        "appreciation": [
            "percaya", "yakin", "aman", "terpercaya", "jujur", "setia", "pasti", "kepercayaan",
            "bersahabat", "kredibel", "solid", "andal", "integritas", "loyalitas",
            "ketulusan", "kebersamaan", "keteguhan", "terjamin",
            "menyokong", "dukung", "peduli", "loyal", "bersahabat", "komitmen",
            "andal",
            "baik", "bagus", "bangga", "hebat", "mantap", "gokil", "keren", "kece",
            "luar biasa", "hebat", 'adil', 'asli', r"\saman\s", 'bangga', 'berani',
            'berguna', 'berpengalaman', 'cemerlang', 'cerah', 'efisien', 'elite',
            'favorit', 'halal', 'halus', 'agung', 'layak', 'lembut', 'lezat',
            'manfaat', 'manis', 'melengkapi', 'pesona', 'memuj[ai]', 'takjub', 'menarik',
            'tegas', 'asyik', ' suka ', 'murni', 'pintar', 'populer', 'rajin', 'pujian',
            'apresiasi', 'saleh', 'sederhana', 'sempurna', 'sopan', 'tulus', 'adil',
            'berani', 'fasih', 'ahli', 'fleksibel', 'kh?arisma', 'istimewa', 'konsisten',
            'mahir', 'makmur', 'pahlawan', 'mampu', 'mulia',

        ],
        "enthusiasm": [
            "menunggu", "mengharapkan", "berharap", "menanti", "bersiap",
            "ekspektasi", r'tidak sabar', "penasaran", "menanti",
            "persiapan", "menyambut", "sambut", "antisipasi",
            "harapan", "antusias", "semoga",
            "gairah", "giat", "minat",
            "tertarik", "motivasi", "ingin", " mau ", "pengen",
            "dedikasi", ' ye?ay ', 'bersemangat',
        ],

    }


    text = re.sub(r'(@\S+)', r'@user', text)
    text = re.sub(r'\s(e?n?gg?a?k)\s', r'tidak', text)
    text = re.sub(r'\s(t?id?ak)\s', r'tidak', text)

    keyword_existence_list = []
    for label in emotions_keywords.keys():
        keyword = r"|".join(emotions_keywords[label])

        if re.search(keyword, " " + text + " "):
            keyword_existence_list.append(label)

    if len(keyword_existence_list) >= 1:
        return keyword_existence_list[0]
    return 'neutral'

def emotion_to_sentiment(label: str) -> str:
    if label in ['label ganda', 'neutral']:
        return 'neutral'
    elif label in ['fear', 'sadness', 'anger', 'disgust']:
        return 'negative'
    return 'positive'

# Create Function to Emotion Count from Dataset
# Create Function to Sentiment Count  from Dataset
    
def classify_emotions_and_sentiments(comments_df):
    """Apply emotion classification and sentiment conversion on comments."""
    comments_df['emotion'] = comments_df['comment'].apply(text_emotion_classification)
    comments_df['sentiment'] = comments_df['emotion'].apply(emotion_to_sentiment)

    return comments_df
    
# Function to plot emotion counts and save the plot
def plot_emotion_counts(comments_df, bucket_name='tresense_bucket'):
    """Generate a bar plot and upload it to Google Cloud Storage with unique filename."""

    colors = {
        'anger': '#DA655B', 'enthusiasm': '#DD649E', 'happiness': '#F7E991', 'calm': '#ECB974',
        'disgust': '#5F4594', 'fear': '#8A77AB', 'appreciation': '#59BCD7', 'sadness': '#5DA4D2'
    }

    emotion_order = ['anger', 'enthusiasm', 'happiness', 'calm', 'disgust', 'fear', 'appreciation', 'sadness']
    ordered_colors = [colors[emotion] for emotion in emotion_order]

    # Hitung jumlah masing-masing emosi
    emotion_counts = comments_df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    emotion_counts = emotion_counts.set_index('emotion')
    emotion_counts = emotion_counts.reindex(emotion_order).fillna(0)

    # Buat plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(emotion_counts.index, emotion_counts["count"], color=ordered_colors, height=0.87)

    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{int(width)}',
                 va='center', ha='left', fontsize=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ticklabel_format(style='plain', axis='x')
    plt.gca().get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f'{int(x):,}'.replace(",", ".")))

    # Simpan ke buffer
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png', bbox_inches='tight')
    plt.close()
    image_buffer.seek(0)

    # Buat nama unik berdasarkan timestamp
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination_blob = f"plots/emotion_plot_{timestamp}.png"

    # Upload ke GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_file(image_buffer, content_type='image/png')

    return blob.public_url

# Main function to process comments and generate the emotion plot
def main(file_path='./app/dataset/yt_data2.json', output_filepath='./emotion_plot.png'):
    comments_df = load_comment_data(file_path)
    comments_df = classify_emotions_and_sentiments(comments_df)
    output_path = plot_emotion_counts(comments_df, output_filepath)
    print(f"Emotion plot saved at: {output_path}")


if __name__ == "__main__":
    main()