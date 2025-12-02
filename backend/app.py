from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime

application = Flask(__name__)
CORS(application)

# Initialize classifier
classifier_service = None

# Store conversation history
conversation_logs = []

def initialize_classifier():
    """Initialize the Multi-Stage classifier or fallback to rule-based"""
    global classifier_service
    # Use the new MultiStageClassifier as the main classifier
    try:
        from multistage_classifier import initialize_multistage_classifier
        classifier_service = initialize_multistage_classifier()
        if classifier_service:
            print("="*60)
            print("[SYSTEM] Multi-Stage Classifier loaded successfully")
            print("[SYSTEM] Model trained with advanced pipeline and calibration")
            print("[SYSTEM] Ready for robust mental health classification")
            print("="*60)
            return
    except Exception as e:
        print(f"[SYSTEM] Could not load Multi-Stage Classifier: {e}")
        # Fallback to rule-based classifier if needed
        from rule_classifier import rule_based_classifier
        classifier_service = rule_based_classifier
        print("="*60)
        print("[SYSTEM] Rule-based classifier initialized (fallback mode)")
        print("[SYSTEM] Train BERT model for higher accuracy")
        print("="*60)


@application.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier_service is not None,
        'timestamp': datetime.now().isoformat()
    })


@application.route('/api/classify', methods=['POST'])
def classify_text_endpoint():
    """
    Classify a single text input
    
    Expected JSON:
    {
        "text": "Your text here"
    }
    """
    try:
        request_data = request.get_json()
        if not request_data or 'text' not in request_data:
            return jsonify({'error': 'Missing required field: text'}), 400
        input_text = request_data['text']
        confidence_threshold = request_data.get('threshold', 0.5)
        if not input_text or not input_text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        # Perform classification using the main classifier (multi-stage)
        try:
            classification_result = classifier_service.classify(
                input_text,
                threshold=confidence_threshold
            )
        except TypeError:
            # If classify() does not accept threshold, call without it
            classification_result = classifier_service.classify(input_text)
        classification_result['timestamp'] = datetime.now().isoformat()
        conversation_logs.append(classification_result)
        return jsonify(classification_result), 200
    except Exception as error:
        return jsonify({'error': str(error)}), 500


@application.route('/api/batch-classify', methods=['POST'])
def batch_classify_endpoint():
    """
    Classify multiple texts at once
    
    Expected JSON:
    {
        "texts": ["text1", "text2", ...]
    }
    """
    try:
        request_data = request.get_json()
        if not request_data or 'texts' not in request_data:
            return jsonify({'error': 'Missing required field: texts'}), 400
        text_list = request_data['texts']
        confidence_threshold = request_data.get('threshold', 0.5)
        if not isinstance(text_list, list):
            return jsonify({'error': 'texts must be a list'}), 400
        # Perform batch classification using the main classifier (multi-stage)
        batch_results = []
        for text in text_list:
            try:
                result = classifier_service.classify(text, threshold=confidence_threshold)
            except TypeError:
                result = classifier_service.classify(text)
            batch_results.append(result)
        return jsonify({
            'results': batch_results,
            'count': len(batch_results),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as error:
        return jsonify({'error': str(error)}), 500


@application.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all available classification categories"""
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return jsonify({
        'categories': categories,
        'count': len(categories)
    })


@application.route('/api/history', methods=['GET'])
def get_conversation_history():
    """Get classification history"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({
        'history': conversation_logs[-limit:],
        'total': len(conversation_logs)
    })


@application.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    """Clear conversation history"""
    global conversation_logs
    conversation_logs = []
    return jsonify({'message': 'History cleared successfully'})


@application.route('/api/stats', methods=['GET'])
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
        for prediction in log_entry['predictions']:
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


@application.errorhandler(404)
def not_found_handler(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@application.errorhandler(500)
def internal_error_handler(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    initialize_classifier()
    port_number = int(os.environ.get('PORT', 5000))
    application.run(
        host='0.0.0.0',
        port=port_number,
        debug=True
    )