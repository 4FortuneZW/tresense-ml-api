# Import necessary libraries
import os
import re
import torch
import pandas as pd
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

# Set display options for pandas
pd.set_option('display.max_rows', None)

# Set the device for model loading (use CUDA if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the comment data from the specified file path
file_path = './app/dataset/yt_data.json'
comments_df = extract_comments(file_path)["comment"]

# Load Sentence Transformer model
sentence_model = SentenceTransformer('LazarusNLP/all-indobert-base-v2', device=device)

# Generate embeddings for the comments
embeddings = sentence_model.encode(comments_df, show_progress_bar=True, device=device)

# Reduce embeddings for visualization using UMAP
reduced_embeddings = UMAP(
    n_neighbors=24, n_components=2, min_dist=0, metric='cosine', random_state=42, verbose=True
).fit_transform(embeddings)

# Define UMAP and HDBSCAN models for topic modeling
umap_model = UMAP(n_neighbors=16, n_components=2, min_dist=0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=40, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# KeyBERT and MMR models for topic representation
keybert = KeyBERTInspired()
mmr = MaximalMarginalRelevance(diversity=0.4)

# Store the representation models
representation_model = {
    "KeyBERT": keybert,
    "MMR": mmr,
}

# Initialize BERTopic model
topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    top_n_words=10,
    verbose=True
)

# Train the topic model
topics, probs = topic_model.fit_transform(comments_df, embeddings)

# Retrieve topic information
representative = topic_model.get_topic_info()["Representation"]
representative_docs = topic_model.get_topic_info()["Representative_Docs"]


# Function to process text using Llama3.1 model
def process_text_with_llama(input_text, model_name="llama-3.1-8b-instant", api_key="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"):
    """
    Processes the input text using the Llama 3.1 model and returns the result.
    """

    # Change this Later using topic specified
    def create_prompt():
        return """You are a helpful, respectful and honest assistant for labeling topics. Use Indonesian language to get keyword of topic. To the point"""
    
    def create_messages(prompt, input_text):
        return [
            ChatMessage(
                role="system", content=prompt,
                temperature=1,
                max_tokens=4096,
                top_p=1,
            ),
            ChatMessage(role="user", content=input_text),
        ]

    def call_model(messages):
        llm = Groq(model=model_name, api_key=api_key)
        response = llm.chat(messages)
        return response
    
    prompt = create_prompt()
    messages = create_messages(prompt, input_text)
    response = call_model(messages)
    return {"message": "Processed text", "result": response}



def generate_interactive_plot(reduced_embeddings, labels, output_path="interactive_plot.html"):
    """
    Generates an interactive plot using `datamapplot` and saves it as an HTML file.
    """
    plot = datamapplot.create_interactive_plot(
        reduced_embeddings,
        all_labels,
        # label_font_size=11,
        title="IndoBERT Topic Result",
        sub_title="Topics labeled with `Llama 3.1",
        # label_wrap_width=20,
        # use_medoids=True,
        # logo=bertopic_logo,
        # logo_width=0.16,
        # label_over_points=True,
        # dynamic_label_size=True,
        # darkmode=True,
        enable_search=True,
    )

    # Save the plot as an interactive HTML file
    # plot.save(output_path)
    plot.save(output_path)
    print(f"Interactive plot saved at {output_path}")

    return output_path

# Get all data before to topic
result = []
for i in tqdm(range(0, len(representative_docs))):
    prompt = f"""I have a topic that contains the following documents:
    {representative_docs[i]}
    The topic is described by the following keywords: {representative[i]}.
    Based on the information about the topic above, please create a short label of this topic. Make sure you only return the label and nothing more. Don't use break \n, and dont output"""
    # Text generation with Gemma2
    llama_result = process_text_with_llama(prompt, model_name="llama-3.1-8b-instant", api_key="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")
    result.append(llama_result['result'].message.content)

# Create a label for each document
llm_labels = [re.sub(r'\W+', ' ', label.split("\n")[0].replace('"', '')) for label in result]
llm_labels = [label if label else "Unlabelled" for label in llm_labels]
all_labels = [llm_labels[topic + topic_model._outliers] if topic != -1 else "Unlabelled" for topic in topics]

# print((all_labels))

interactive_plot_path = generate_interactive_plot(reduced_embeddings, all_labels, output_path="app/dataset/BertTopic.html")
print(f"Plot saved at {interactive_plot_path}")
