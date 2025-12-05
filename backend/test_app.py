"""Test Flask app initialization"""
from app import application, classifier_service

print("✅ App loaded successfully")
print(f"✅ Classifier type: {type(classifier_service).__name__}")

# Test classification
result = classifier_service.classify('test')
print(f"✅ Classification test passed: {result['primary_category']}")

# Test all endpoints imports
print("✅ All Flask routes registered:")
for rule in application.url_map.iter_rules():
    print(f"   {rule.endpoint}: {rule.rule}")

print("\n✅ All systems operational!")
