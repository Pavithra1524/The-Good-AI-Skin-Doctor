import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path("instance/good_ai.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    skin_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    answers TEXT,
    result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    image_path TEXT,
    prediction TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    doctor_name TEXT,
    date TEXT,
    time TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    summary TEXT,
    url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Seed a few articles (idempotent-ish)
        cur = conn.execute("SELECT COUNT(*) AS c FROM articles")
        if cur.fetchone()[0] == 0:
            samples = [
                ("Understanding Acne Triggers", "Diet, stress, and skincare habits—what matters?", "https://example.com/acne-triggers"),
                ("Sunscreen 101", "Why SPF is your daily non‑negotiable.", "https://example.com/sunscreen-101"),
                ("Melanoma: ABCDE Checks", "Spot changes early using the ABCDE rule.", "https://example.com/melanoma-abcde"),
            ]
            conn.executemany(
                "INSERT INTO articles(title, summary, url) VALUES(?,?,?)",
                samples,
            )


# --- User helpers ---

def create_user(name, email, password):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users(name, email, password_hash) VALUES(?,?,?)",
            (name, email, generate_password_hash(password)),
        )


def get_user_by_email(email):
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE email=?", (email,))
        return cur.fetchone()


def verify_user(email, password):
    user = get_user_by_email(email)
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def get_user_by_id(user_id):
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
        return cur.fetchone()


def save_quiz(user_id, answers, result):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO quiz_results(user_id, answers, result) VALUES(?,?,?)",
            (user_id, answers, result),
        )
        conn.execute("UPDATE users SET skin_type=? WHERE id=?", (result, user_id))


def save_analysis(user_id, image_path, prediction, confidence):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO analysis_results(user_id, image_path, prediction, confidence) VALUES(?,?,?,?)",
            (user_id, image_path, prediction, confidence),
        )


def list_analysis(user_id):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM analysis_results WHERE user_id=? ORDER BY created_at DESC",
            (user_id,),
        )
        return cur.fetchall()


def list_appointments(user_id):
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM appointments WHERE user_id=? ORDER BY date DESC, time DESC",
            (user_id,),
        )
        return cur.fetchall()


def create_appointment(user_id, doctor_name, date, time, notes):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO appointments(user_id, doctor_name, date, time, notes) VALUES(?,?,?,?,?)",
            (user_id, doctor_name, date, time, notes),
        )


def list_articles():
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM articles ORDER BY created_at DESC")
        return cur.fetchall()