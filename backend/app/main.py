from flask import Flask
from flask_cors import CORS
from app.services.data_loader import initialize_data
from app.api.routes import api_bp

def create_app():
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": [
                "http://localhost",
                "http://localhost:80",
                "http://127.0.0.1",
                "http://127.0.0.1:80",
                "http://localhost:3000",
                "http://localhost:5000"
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Register Blueprints
    app.register_blueprint(api_bp)
    
    @app.route('/')
    def health_check():
        return "Safety Analysis Service Running"
        
    return app

app = create_app()

if __name__ == '__main__':
    initialize_data()
    app.run(host='0.0.0.0', port=5000, debug=True)
