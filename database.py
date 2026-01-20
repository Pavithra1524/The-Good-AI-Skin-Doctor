import sqlite3
from sqlite3 import Error
from datetime import datetime
import os
import pytz

DATABASE_FILE = "users.db"
local_tz = pytz.timezone('Asia/Kolkata')  # Indian Standard Time

def create_connection():
    """Create a database connection and return the connection object"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def init_db():
    """Initialize the database with required tables"""
    conn = create_connection()
    if conn is not None:
        try:
            # Create users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create or upgrade history table
            # First create with basic structure
            conn.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    image_path TEXT,
                    skin_condition TEXT NOT NULL DEFAULT 'Unknown condition',
                    confidence REAL DEFAULT 0.0,
                    clinical_recommendation TEXT DEFAULT '',
                    natural_remedies TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    filename TEXT DEFAULT 'image.jpg',
                    source TEXT DEFAULT 'Analysis',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Then add new columns if they don't exist
            try:
                conn.execute('ALTER TABLE history ADD COLUMN clinical_recommendations TEXT')
            except sqlite3.OperationalError:
                # Column might already exist
                pass

            # Create OTP table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS otp_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    otp TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')

            conn.commit()
            print("Database initialized successfully")
        except Error as e:
            print(f"Error initializing database: {e}")
        finally:
            conn.close()
    else:
        print("Error: Could not create database connection")

def add_user(email, name, password_hash):
    """Add a new user to the database"""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (email, name, password) VALUES (?, ?, ?)",
                (email, name, password_hash)
            )
            conn.commit()
            return True
        except Error as e:
            print(f"Error adding user: {e}")
            return False
        finally:
            conn.close()
    return False

def get_user_by_email(email):
    """Get user details by email"""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, email, name, password FROM users WHERE email = ?",
                (email,)
            )
            user = cursor.fetchone()
            if user:
                return {
                    'id': user[0],
                    'email': user[1],
                    'name': user[2],
                    'password': user[3]
                }
        except Error as e:
            print(f"Error getting user: {e}")
        finally:
            conn.close()
    return None

def store_otp(email, otp):
    """Store OTP in the database with expiration"""
    conn = create_connection()
    if conn is not None:
        try:
            # First, delete any existing OTP for this email
            conn.execute("DELETE FROM otp_store WHERE email = ?", (email,))
            
            # Store new OTP with 10-minute expiration
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO otp_store (email, otp, expires_at) VALUES (?, ?, datetime('now', '+10 minutes'))",
                (email, otp)
            )
            conn.commit()
            return True
        except Error as e:
            print(f"Error storing OTP: {e}")
            return False
        finally:
            conn.close()
    return False

def verify_otp(email, otp):
    """Verify OTP for the given email"""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT otp FROM otp_store WHERE email = ? AND expires_at > datetime('now')",
                (email,)
            )
            stored_otp = cursor.fetchone()
            if stored_otp and stored_otp[0] == otp:
                # Delete the used OTP
                conn.execute("DELETE FROM otp_store WHERE email = ?", (email,))
                conn.commit()
                return True
            return False
        except Error as e:
            print(f"Error verifying OTP: {e}")
            return False
        finally:
            conn.close()
    return False

def add_history_record(user_id, record):
    """Add a history record for a user"""
    print("\n=== Adding History Record ===")
    print("Record:", record)  # Debug print
    conn = create_connection()
    if conn is not None:
        try:
            # Check if a record with the same image_path exists for this user
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM history 
                WHERE user_id = ? AND image_path = ? AND created_at >= datetime('now', '-1 minute')
            """, (user_id, record.get('image_path')))
            
            existing_record = cursor.fetchone()
            if existing_record:
                print("Found existing record:", existing_record)  # Debug print
                # Skip if we already have a record for this image in the last minute
                return existing_record[0]

            # Get the disorder name from the record
            disease_name = record.get('disorder')
            if not disease_name:
                disease_name = record.get('skin_condition')
            print(f"Disease name: {disease_name}")  # Debug print

            # Get the disorder name from the record
            disease_name = record.get('disorder') or record.get('skin_condition')
            
            # If no recent record exists, insert the new one
            cursor.execute("""
                INSERT INTO history (
                    user_id, image_path, skin_condition, confidence,
                    clinical_recommendation, natural_remedies, filename, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                record.get('image_path'),
                disease_name,  # Store the disease name directly
                record.get('confidence'),
                record.get('clinical_recommendation') or record.get('clinical_recommendations', ''),
                record.get('natural_remedies'),
                record.get('filename'),
                record.get('source')
            ))
            conn.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Error adding history record: {e}")
            return None
        finally:
            conn.close()
    return None

def get_user_history(user_id):
    """Get analysis history for a user"""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, image_path, skin_condition, confidence,
                       clinical_recommendation, natural_remedies, 
                       datetime(created_at, 'localtime') as created_at,
                       filename, source
                FROM history
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
            
            history = []
            for row in cursor.fetchall():
                # Get the disease name from the database
                disease_name = row[2]
                
                # Only fallback to "Unknown condition" if truly None or empty
                if disease_name in (None, '', 'None', 'none'):
                    disease_name = "Unknown condition"
                
                # Create the record with the disease name
                record = {
                    'id': row[0],
                    'image_path': row[1],
                    'skin_condition': disease_name,
                    'disorder': disease_name,
                    'confidence': row[3] if row[3] is not None else 0.0,
                    'clinical_recommendation': row[4] if row[4] is not None else "",
                    'natural_remedies': row[5] if row[5] is not None else "",
                    'date': datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.UTC).astimezone(local_tz),
                    'filename': row[7] if row[7] is not None else "image.jpg",
                    'source': row[8] if row[8] is not None else "Analysis"
                }
                history.append(record)
            return history
        except Error as e:
            print(f"Error getting user history: {e}")
            return []
        finally:
            conn.close()
    return []

# Initialize the database when this module is imported
init_db()