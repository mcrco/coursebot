from flask import Flask
from app.routes import api


def create_app():
    """
    Factory function to create and configure the Flask app.
    """
    app = Flask(__name__)
    app.register_blueprint(api, url_prefix="/api")

    return app
