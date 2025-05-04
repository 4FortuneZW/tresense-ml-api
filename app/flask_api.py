import os
import pandas as pd

from flask import Flask, request, jsonify

from app.Sentimen import classify_emotions_and_sentiments, plot_emotion_counts
from app.BertTopic import load_comment_data, generate_embeddings, reduce_embeddings_for_visualization, train_bertopic_model, generate_labels_for_topics, generate_interactive_plot

from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))

app = Flask(__name__)

@app.route("/api/v1/analyze-sentiment", methods=["POST"])
def analyze_sentiment():
    try:
        data = request.get_json()

        # Validasi bahwa data harus list of dicts
        if not isinstance(data, list):
            return jsonify({"error": "Expected a list of objects"}), 400

        # Konversi ke DataFrame
        df = pd.DataFrame(data)

        df = classify_emotions_and_sentiments(df)

        plot_url = plot_emotion_counts(df)

        return jsonify({
            "data": df.to_dict(orient="records"),
            "plot_url": plot_url
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/analyze-bertopic", methods=["POST"])
def analyze_bertopic():
    try:
        data = request.get_json()

        # Validasi bahwa data harus list
        if not isinstance(data, list):
            return jsonify({"error": "Expected a list of objects"}), 400

        # Mengonversi data menjadi DataFrame
        comments_df = pd.DataFrame(data)

        # Generate embeddings
        embeddings = generate_embeddings(comments_df["comment"])

        # Reduksi embeddings untuk visualisasi
        reduced_embeddings = reduce_embeddings_for_visualization(embeddings)

        # Latih model BERTopic
        topic_model, topics, probs = train_bertopic_model(comments_df["comment"], embeddings)

        # Menghasilkan label untuk topik-topik
        representative = topic_model.get_topic_info()["Representation"]
        representative_docs = topic_model.get_topic_info()["Representative_Docs"]
        all_labels = generate_labels_for_topics(topic_model, representative_docs, representative, topics)

        # Menghasilkan plot interaktif
        interactive_plot_url = generate_interactive_plot(reduced_embeddings, all_labels)

        return jsonify({
            "topics": topics,
            "labels": all_labels,
            "interactive_plot": interactive_plot_url
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)