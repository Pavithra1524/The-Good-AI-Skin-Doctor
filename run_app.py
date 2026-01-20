#!/usr/bin/env python3
"""
Simple startup script for the Skin Disease Classifier Flask app
"""

import os
import sys
import time

def main():
    print("🏥 Starting Good AI Skin Doctor...")
    print("=" * 50)
    
    # Check if model file exists
    model_path = 'skin_disease_classifier_v1_final.h5'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please make sure the model file is in the current directory.")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"📁 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Import and start the Flask app
    try:
        print("\n🚀 Starting Flask application...")
        print("⏳ Loading TensorFlow (this may take a moment on first run)...")
        
        from app import app
        
        print("✅ Application loaded successfully!")
        print("\n🌐 Starting web server...")
        print("📱 Open your browser and go to: http://127.0.0.1:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the Flask development server
        app.run(debug=True, host='127.0.0.1', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
