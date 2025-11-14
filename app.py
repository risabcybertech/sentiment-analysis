from flask import Flask, jsonify, request, render_template
from db import get_connection, init_db
from sentiment_service import analyze_text

app = Flask(__name__)
init_db()

# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ---------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Text is required"}), 400

    result = analyze_text(text)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analyses (text, sentiment, confidence, toxic, toxicity_score)
        VALUES (?, ?, ?, ?, ?)
    """, (text, result["sentiment"], result["confidence"], int(result["toxic"]), result["toxicity_score"]))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify(result)

@app.route("/api/analyses", methods=["GET"])
def get_all():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 10")
    rows = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify(rows)

# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
