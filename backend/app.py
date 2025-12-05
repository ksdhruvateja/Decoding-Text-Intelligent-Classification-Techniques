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

def _format_detected_labels(all_scores: dict) -> dict:
    # Normalize raw scores to sum to 1.0 first (proper softmax-like normalization)
    raw_scores = {
        'emotional_distress': float(all_scores.get('emotional_distress', 0.0)),
        'negative': float(all_scores.get('negative', 0.0)),
        'neutral': float(all_scores.get('neutral', 0.0)),
        'positive': float(all_scores.get('positive', 0.0)),
        'self_harm_high': float(all_scores.get('self_harm_high', 0.0)),
        'self_harm_low': float(all_scores.get('self_harm_low', 0.0)),
        'stress': float(all_scores.get('stress', 0.0)),
        'unsafe_environment': float(all_scores.get('unsafe_environment', 0.0)),
    }
    
    # Calculate threat_of_violence ONLY from unsafe_environment (no mixing)
    # This prevents false threat detection
    threat_score = float(all_scores.get('unsafe_environment', 0.0))
    
    # Add threat to raw scores
    raw_scores_all = {**raw_scores, 'threat_of_violence': threat_score}
    
    # Normalize to sum to 1.0
    total = sum(raw_scores_all.values())
    if total <= 0.01:
        # Default: 100% neutral for empty/unclear input
        normalized = {k: 0.0 for k in raw_scores_all.keys()}
        normalized['neutral'] = 1.0
    else:
        normalized = {k: v / total for k, v in raw_scores_all.items()}
    
    # Convert to percentages (sum = 100.0)
    percent = {k: round(v * 100.0, 1) for k, v in normalized.items()}
    
    # Map to output label format
    output = {
        'emotional distress': percent['emotional_distress'],
        'negative': percent['negative'],
        'neutral': percent['neutral'],
        'positive': percent['positive'],
        'self harm_high': percent['self_harm_high'],
        'self harm_low': percent['self_harm_low'],
        'stress': percent['stress'],
        'unsafe environment': percent['unsafe_environment'],
        'threat_of_violence': percent['threat_of_violence'],
    }
    
    # Force exact 100.0 sum by adjusting dominant label
    actual_sum = sum(output.values())
    if abs(actual_sum - 100.0) > 0.1:
        drift = 100.0 - actual_sum
        top_label = max(output.keys(), key=lambda k: output[k])
        output[top_label] = round(output[top_label] + drift, 1)
    
    return output

def _format_block_output(primary_label: str, all_scores: dict) -> str:
    labels = _format_detected_labels(all_scores)
    order = [
        'emotional distress','negative','neutral','positive',
        'self harm_high','self harm_low','stress','unsafe environment','threat_of_violence'
    ]

    lines = [
        "Emotion vector",
        primary_label,
        "",
        "Detected labels",
        "",
    ]
    for key in order:
        lines.append(key)
        lines.append(f"{labels.get(key, 0.0)} %")
    return "\n".join(lines)

def initialize_classifier():
    """Initialize classifier - use production hybrid classifier"""
    global classifier_service
    if classifier_service is not None:
        return  # Already initialized
    
    # Try production classifier (hybrid BERT + rules)
    try:
        from production_classifier import ProductionClassifier
        classifier_service = ProductionClassifier()
        print("="*60)
        print("[SYSTEM] Production Classifier loaded successfully")
        print("[SYSTEM] Hybrid: Rule-based + BERT (when available)")
        print("[SYSTEM] Accurate classification with graceful fallback")
        print("="*60)
        return
    except Exception as e:
        print(f"[SYSTEM] Could not load Production Classifier: {e}")
    
    # Fallback to simple classifier
    try:
        from simple_classifier import SimpleClassifier
        classifier_service = SimpleClassifier()
        print("="*60)
        print("[SYSTEM] Simple Classifier loaded (fallback)")
        print("[SYSTEM] Rule-based classification active")
        print("="*60)
        return
    except Exception as e:
        print(f"[SYSTEM] Could not load any classifier: {e}")
        classifier_service = None

# Initialize classifier when module loads (for Gunicorn)
initialize_classifier()


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
        
        # Check if classifier is available
        if classifier_service is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        # Perform classification using the main classifier
        # Debug logging snapshot for deployment issues
        try:
            debug_snapshot = {
                'received_text_preview': str(input_text)[:120],
                'received_len': len(str(input_text)),
                'has_newlines': '\n' in str(input_text),
                'whitespace_only': str(input_text).strip() == '',
                'model_impl': type(classifier_service).__name__,
            }
            print(f"[DEBUG] classify snapshot: {debug_snapshot}")
        except Exception:
            pass

        classification_result = classifier_service.classify(str(input_text))
        classification_result['timestamp'] = datetime.now().isoformat()
        conversation_logs.append(classification_result)
        # Support optional formatted output via query or payload flag
        want_formatted = request.args.get('formatted', 'false').lower() == 'true' or bool(request_data.get('formatted', False))
        if want_formatted:
            block = _format_block_output(classification_result['primary_category'], classification_result.get('all_scores', {}))
            return block, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        return jsonify(classification_result), 200
    except Exception as error:
        return jsonify({'error': str(error)}), 500

@application.route('/api/classify-formatted', methods=['POST'])
def classify_text_formatted_endpoint():
    """Return the classification in the strict block format required by the frontend."""
    try:
        request_data = request.get_json()
        if not request_data or 'text' not in request_data:
            return jsonify({'error': 'Missing required field: text'}), 400
        input_text = request_data['text']
        if not input_text or not str(input_text).strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        if classifier_service is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        result = classifier_service.classify(str(input_text))
        block = _format_block_output(result['primary_category'], result.get('all_scores', {}))
        return block, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as error:
        return jsonify({'error': str(error)}), 500
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
        
        # Check if classifier is available
        if classifier_service is None:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        # Perform batch classification using the main classifier
        batch_results = []
        for text in text_list:
            if not text or not str(text).strip():
                continue  # Skip empty texts
            result = classifier_service.classify(str(text))
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
        for prediction in log_entry.get('predictions', []):
            label = prediction['label']
            category_counts[label] = category_counts.get(label, 0) + 1
            # Use 'score' field from simple classifier
            score = prediction.get('score', prediction.get('confidence', 0))
            total_confidence += score
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
    port_number = int(os.environ.get('PORT', 5000))
    print(f"\nStarting Flask server on http://localhost:{port_number}")
    print(f"Classifier: {'Multi-Stage' if 'multistage' in str(type(classifier_service)).lower() else 'Rule-based'}")
    print(f"Ready to classify text!\n")
    application.run(
        host='0.0.0.0',
        port=port_number,
        debug=False,
        use_reloader=False
    )