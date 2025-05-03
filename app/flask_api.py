from flask import Flask, request, jsonify
# from Sentimen import classify_emotions_and_sentiments, plot_emotion_counts
import pandas as pd
import os
# from BertTopic import (
#     load_comment_data, generate_embeddings, reduce_embeddings_for_visualization, 
#     train_bertopic_model, generate_labels_for_topics, generate_interactive_plot
# )

app = Flask(__name__)

@app.route("/hello", methods=["GET"])
def hello_world():
    return "Hello World"

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    try:
        # Mengambil data JSON dari request
        data = request.get_json()

        # Memeriksa apakah data ada dan terdapat key 'comments'
        if not data or 'comments' not in data:
            return jsonify({"error": "JSON must include a 'comments' list"}), 400

        # Mengambil komentar dari 'comments'
        comments = data['comments']
        
        # Memeriksa apakah 'comments' adalah list
        if not isinstance(comments, list):
            return jsonify({"error": "'comments' must be a list"}), 400

        # Membuat DataFrame dari list komentar
        df = pd.DataFrame(comments)

        # Memeriksa apakah kolom 'comment' ada dalam DataFrame
        if 'comment' not in df.columns:
            return jsonify({"error": "Each item must contain a 'comment' key"}), 400

        # Mengambil hanya kolom 'comment' dan mengembalikannya sebagai response
        comment_list = df['comment'].tolist()

        return jsonify({"comments": comment_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route("/topic-analysis", methods=["POST"])
# def analyze_topics():
#     try:
#         data = request.get_json()

#         if not data or 'comments' not in data:
#             return jsonify({"error": "JSON must include a 'comments' list"}), 400

#         comments = data['comments']
#         if not isinstance(comments, list):
#             return jsonify({"error": "'comments' must be a list"}), 400

#         # Proses menggunakan BERTopic
#         comments_df = pd.Series(comments)
#         embeddings = generate_embeddings(comments_df)

#         # Reduksi embeddings untuk visualisasi
#         reduced_embeddings = reduce_embeddings_for_visualization(embeddings)

#         # Latih model BERTopic
#         topic_model, topics, probs = train_bertopic_model(comments_df, embeddings)

#         # Dapatkan label untuk setiap topik
#         representative_docs = topic_model.get_topic_info()["Representative_Docs"]
#         representative = topic_model.get_topic_info()["Representation"]
#         all_labels = generate_labels_for_topics(topic_model, representative_docs, representative, topics)

#         # Generate plot interaktif dan simpan
#         interactive_plot_path = generate_interactive_plot(reduced_embeddings, all_labels)

#         # Kembalikan hasil analisis topik
#         return jsonify({
#             "topics": all_labels,
#             "interactive_plot_saved": interactive_plot_path  # Menyimpan path file plot
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)