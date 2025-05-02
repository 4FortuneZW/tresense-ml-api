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

# Function to load the comment data
def load_comment_data(file_path='./app/dataset/yt_data.json'):
    """Load the comment data from the specified file path."""
    comments_df = extract_comments(file_path)["comment"]
    return comments_df

# Function to generate sentence embeddings using SentenceTransformer
def generate_embeddings(comments_df, device='cpu'):
    """Generate embeddings for the comments using SentenceTransformer."""
    sentence_model = SentenceTransformer('LazarusNLP/all-indobert-base-v2', device=device)
    embeddings = sentence_model.encode(comments_df, show_progress_bar=True, device=device)
    return embeddings

# Function to reduce embeddings for visualization using UMAP
def reduce_embeddings_for_visualization(embeddings):
    """Reduce embeddings for visualization using UMAP."""
    reduced_embeddings = UMAP(
        n_neighbors=40, n_components=2, min_dist=0.02, metric='cosine', random_state=33, verbose=True
    ).fit_transform(embeddings)
    return reduced_embeddings

# Function to initialize and train BERTopic model
def train_bertopic_model(comments_df, embeddings):
    """Initialize and train BERTopic model."""
    umap_model = UMAP(n_neighbors=40, n_components=2, min_dist=0.02, metric='cosine', random_state=33)
    hdbscan_model = HDBSCAN(min_cluster_size=40, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    keybert = KeyBERTInspired()
    mmr = MaximalMarginalRelevance(diversity=0.4)

    representation_model = {
        "KeyBERT": keybert,
        "MMR": mmr,
    }

    topic_model = BERTopic(
        embedding_model='LazarusNLP/all-indobert-base-v2',
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(comments_df, embeddings)
    return topic_model, topics, probs

# Function to process text using Llama3.1 model
def process_text_with_llama(input_text, model_name="llama-3.1-8b-instant", api_key="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"):
    """Processes the input text using the Llama 3.1 model and returns the result."""
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

# Function to generate interactive plot
def generate_interactive_plot(reduced_embeddings, labels, output_path="interactive_plot.html"):
    """Generates an interactive plot using `datamapplot` and saves it as an HTML file."""
    plot = datamapplot.create_interactive_plot(
        reduced_embeddings,
        labels,
        title="IndoBERT Topic Result",
        sub_title="Topics labeled with `Llama 3.1`",
        enable_search=True,
    )

    plot.save(output_path)
    print(f"Interactive plot saved at {output_path}")
    return output_path

# Function to generate labels for the topics
def generate_labels_for_topics(topic_model, representative_docs, representative, topics):
    """Generate labels for topics using Llama 3.1 model."""
    result = []
    for i in tqdm(range(0, len(representative_docs))):
        prompt = f"""I have a topic that contains the following documents:
        {representative_docs[i]}
        The topic is described by the following keywords: {representative[i]}.
        Based on the information about the topic above, please create a short label of this topic. Make sure you only return the label and nothing more. Don't use break \n, and dont output"""
        
        llama_result = process_text_with_llama(prompt)
        result.append(llama_result['result'].message.content)

    llm_labels = [re.sub(r'\W+', ' ', label.split("\n")[0].replace('"', '').replace(r'label|Label|LABEL', '').strip()) for label in result]
    llm_labels = [label if label else "Unlabelled" for label in llm_labels]
    all_labels = [llm_labels[topic + topic_model._outliers] if topic != -1 else "Unlabelled" for topic in topics]
    
    return all_labels

# Main function to execute all tasks
def main(file_path='./app/dataset/yt_data.json', output_path="interactive_plot.html"):
    # Load comment data
    comments_df = load_comment_data(file_path)
    
    # Generate embeddings
    embeddings = generate_embeddings(comments_df)
    
    # Reduce embeddings for visualization
    reduced_embeddings = reduce_embeddings_for_visualization(embeddings)
    
    # Train BERTopic model
    topic_model, topics, probs = train_bertopic_model(comments_df, embeddings)
    
    # Generate labels
    representative = topic_model.get_topic_info()["Representation"]
    representative_docs = topic_model.get_topic_info()["Representative_Docs"]
    all_labels = generate_labels_for_topics(topic_model, representative_docs, representative, topics)
    
    # Generate interactive plot
    interactive_plot_path = generate_interactive_plot(reduced_embeddings, all_labels, output_path)
    print(f"Plot saved at {interactive_plot_path}")

if __name__ == "__main__":
    main()
