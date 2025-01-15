from flask import Flask
from flask_cors import CORS
from app.routes import api
from dotenv import load_dotenv
import os


def create_app():
    """
    Factory function to create and configure the Flask app.
    """
    app = Flask(__name__)
    app.register_blueprint(api, url_prefix="/api")
    if not load_dotenv():
        print("Could not load .env")
    print("allowed origin:", os.environ["ALLOWED_ORIGIN"])
    app.config["CORS_HEADERS"] = "Content-Type"

    CORS(app, origins=[os.environ["ALLOWED_ORIGIN"]])

    return app
