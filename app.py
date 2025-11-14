from flask import Flask, jsonify, request, render_template
from db import get_connection, init_db
from sentiment_service import analyze_text

app = Flask(__name__)
init_db()  # Initialize database tables

# ---------------------------
# 🌐 Web Routes (Templates)
# ---------------------------

@app.route("/")
def home():
    """Render the homepage (text analysis form)."""
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    """Render the dashboard page showing recent analyses."""
    return render_template("dashboard.html")


# ---------------------------
# ⚙️ API Routes
# ---------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze text and store result in MySQL."""
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    # Analyze using Hugging Face or internal model
    result = analyze_text(text)

    # Store in DB
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analyses (text, sentiment, confidence, toxic, toxicity_score)
        VALUES (%s, %s, %s, %s, %s)
    """, (text, result["sentiment"], result["confidence"], result["toxic"], result["toxicity_score"]))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify(result)


@app.route("/api/analyses", methods=["GET"])
def get_all():
    """Fetch recent 10 analyses from MySQL."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify(rows)


# ---------------------------
# 🚀 Main Entry Point
# ---------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
