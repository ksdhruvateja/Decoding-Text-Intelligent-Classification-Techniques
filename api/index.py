"""
Vercel Serverless Function for Flask API
This wraps the Flask app to work as serverless functions on Vercel
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize classifier (lazy loading)
classifier_service = None
conversation_logs = []

def get_classifier():
    """Lazy load classifier to avoid cold start issues"""
    global classifier_service
    if classifier_service is None:
        try:
            from multistage_classifier import initialize_multistage_classifier
            classifier_service = initialize_multistage_classifier()
            if classifier_service:
                print("[SYSTEM] Multi-Stage Classifier loaded successfully")
                return classifier_service
        except Exception as e:
            print(f"[SYSTEM] Could not load Multi-Stage Classifier: {e}")
            try:
                from rule_classifier import rule_based_classifier
                classifier_service = rule_based_classifier
                print("[SYSTEM] Rule-based classifier initialized (fallback mode)")
                return classifier_service
            except Exception as e2:
                print(f"[SYSTEM] Could not load rule-based classifier: {e2}")
                return None
    return classifier_service

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    classifier = get_classifier()
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/classify', methods=['POST', 'OPTIONS'])
def classify_text_endpoint():
    """Classify a single text input"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        request_data = request.get_json()
        if not request_data or 'text' not in request_data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        input_text = request_data['text']
        confidence_threshold = request_data.get('threshold', 0.5)
        
        if not input_text or not input_text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        classifier = get_classifier()
        if classifier is None:
            return jsonify({'error': 'Classifier not available'}), 503
        
        try:
            classification_result = classifier.classify(
                input_text,
                threshold=confidence_threshold
            )
        except TypeError:
            classification_result = classifier.classify(input_text)
        
        classification_result['timestamp'] = datetime.now().isoformat()
        conversation_logs.append(classification_result)
        
        return jsonify(classification_result), 200
    except Exception as error:
        return jsonify({'error': str(error)}), 500

@app.route('/api/batch-classify', methods=['POST', 'OPTIONS'])
def batch_classify_endpoint():
    """Classify multiple texts at once"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        request_data = request.get_json()
        if not request_data or 'texts' not in request_data:
            return jsonify({'error': 'Missing required field: texts'}), 400
        
        text_list = request_data['texts']
        confidence_threshold = request_data.get('threshold', 0.5)
        
        if not isinstance(text_list, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        classifier = get_classifier()
        if classifier is None:
            return jsonify({'error': 'Classifier not available'}), 503
        
        batch_results = []
        for text in text_list:
            try:
                result = classifier.classify(text, threshold=confidence_threshold)
            except TypeError:
                result = classifier.classify(text)
            batch_results.append(result)
        
        return jsonify({
            'results': batch_results,
            'count': len(batch_results),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as error:
        return jsonify({'error': str(error)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all available classification categories"""
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return jsonify({
        'categories': categories,
        'count': len(categories)
    })

@app.route('/api/history', methods=['GET'])
def get_conversation_history():
    """Get classification history"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({
        'history': conversation_logs[-limit:],
        'total': len(conversation_logs)
    })

@app.route('/api/history/clear', methods=['DELETE', 'OPTIONS'])
def clear_history():
    """Clear conversation history"""
    global conversation_logs
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    conversation_logs = []
    return jsonify({'message': 'History cleared successfully'})

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get classification statistics"""
    if not conversation_logs:
        return jsonify({
            'total_classifications': 0,
            'category_counts': {},
            'average_confidence': 0
        })
    
    category_counts = {}
    total_confidence = 0
    prediction_count = 0
    
    for log_entry in conversation_logs:
        for prediction in log_entry.get('predictions', []):
            label = prediction['label']
            category_counts[label] = category_counts.get(label, 0) + 1
            total_confidence += prediction['confidence']
            prediction_count += 1
    
    return jsonify({
        'total_classifications': len(conversation_logs),
        'category_counts': category_counts,
        'average_confidence': total_confidence / prediction_count if prediction_count > 0 else 0,
        'unique_categories_found': len(category_counts)
    })

@app.errorhandler(404)
def not_found_handler(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error_handler(error):
    return jsonify({'error': 'Internal server error'}), 500

# Vercel serverless function handler
def handler(request):
    """Vercel serverless function entry point"""
    return app(request.environ, lambda status, headers: None)

