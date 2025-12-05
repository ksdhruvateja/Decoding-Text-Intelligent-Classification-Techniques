"""Start Flask server with proper error handling"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import application, classifier_service
    
    print("✅ Flask app imported successfully")
    print(f"✅ Classifier loaded: {type(classifier_service).__name__}")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting Flask server on http://localhost:{port}")
    print(f"📊 Classifier: {type(classifier_service).__name__}")
    print(f"✅ Ready to classify text!\n")
    
    application.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )
except KeyboardInterrupt:
    print("\n\n⏹️  Server stopped by user")
except Exception as e:
    print(f"\n\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
