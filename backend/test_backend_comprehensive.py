"""Comprehensive backend testing script"""
import sys
import traceback

print("="*70)
print("COMPREHENSIVE BACKEND TEST")
print("="*70)

# Test 1: Import app
print("\n[TEST 1] Importing Flask app...")
try:
    from app import application, classifier_service
    print("✅ App imported successfully")
    print(f"✅ Classifier type: {type(classifier_service).__name__}")
    print(f"✅ Classifier loaded: {classifier_service is not None}")
except Exception as e:
    print(f"❌ ERROR importing app: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check classifier initialization
print("\n[TEST 2] Testing classifier...")
try:
    if classifier_service is None:
        print("❌ Classifier is None!")
        sys.exit(1)
    
    test_result = classifier_service.classify("This is a test")
    print(f"✅ Classification test passed")
    print(f"   Result: {test_result.get('primary_category', 'unknown')}")
except Exception as e:
    print(f"❌ ERROR in classifier: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check all routes
print("\n[TEST 3] Checking Flask routes...")
try:
    routes = []
    for rule in application.url_map.iter_rules():
        routes.append(f"{rule.methods} {rule.rule}")
    print(f"✅ Found {len(routes)} routes:")
    for route in routes:
        print(f"   {route}")
except Exception as e:
    print(f"❌ ERROR checking routes: {e}")
    traceback.print_exc()

# Test 4: Test endpoints with test client
print("\n[TEST 4] Testing endpoints with test client...")
test_client = application.test_client()

# Health check
print("\n   Testing /api/health...")
try:
    response = test_client.get('/api/health')
    if response.status_code == 200:
        print(f"   ✅ Health check: {response.get_json()}")
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in health check: {e}")
    traceback.print_exc()

# Categories
print("\n   Testing /api/categories...")
try:
    response = test_client.get('/api/categories')
    if response.status_code == 200:
        data = response.get_json()
        print(f"   ✅ Categories: {data.get('count', 0)} categories")
    else:
        print(f"   ❌ Categories failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in categories: {e}")
    traceback.print_exc()

# Classify
print("\n   Testing /api/classify...")
try:
    response = test_client.post('/api/classify', 
                              json={'text': 'I will kill him'},
                              content_type='application/json')
    if response.status_code == 200:
        data = response.get_json()
        print(f"   ✅ Classify: {data.get('primary_category', 'unknown')} ({data.get('confidence', 0):.2f})")
    else:
        print(f"   ❌ Classify failed: {response.status_code}")
        print(f"   Response: {response.get_json()}")
except Exception as e:
    print(f"   ❌ ERROR in classify: {e}")
    traceback.print_exc()

# Batch classify
print("\n   Testing /api/batch-classify...")
try:
    response = test_client.post('/api/batch-classify',
                              json={'texts': ['test 1', 'test 2']},
                              content_type='application/json')
    if response.status_code == 200:
        data = response.get_json()
        print(f"   ✅ Batch classify: {data.get('count', 0)} results")
    else:
        print(f"   ❌ Batch classify failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in batch-classify: {e}")
    traceback.print_exc()

# History
print("\n   Testing /api/history...")
try:
    response = test_client.get('/api/history')
    if response.status_code == 200:
        data = response.get_json()
        print(f"   ✅ History: {data.get('total', 0)} entries")
    else:
        print(f"   ❌ History failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in history: {e}")
    traceback.print_exc()

# Clear history
print("\n   Testing /api/history/clear...")
try:
    response = test_client.delete('/api/history/clear')
    if response.status_code == 200:
        data = response.get_json()
        print(f"   ✅ Clear history: {data.get('message', 'success')}")
    else:
        print(f"   ❌ Clear history failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in clear history: {e}")
    traceback.print_exc()

# Stats
print("\n   Testing /api/stats...")
try:
    response = test_client.get('/api/stats')
    if response.status_code == 200:
        data = response.get_json()
        print(f"   ✅ Stats: {data.get('total_classifications', 0)} classifications")
    else:
        print(f"   ❌ Stats failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in stats: {e}")
    traceback.print_exc()

# Test 5: Test error handling
print("\n[TEST 5] Testing error handling...")

# Missing text field
print("\n   Testing missing text field...")
try:
    response = test_client.post('/api/classify', json={}, content_type='application/json')
    if response.status_code == 400:
        print(f"   ✅ Error handling: Correctly returned 400 for missing text")
    else:
        print(f"   ⚠️  Expected 400, got {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in error handling test: {e}")

# Empty text
print("\n   Testing empty text...")
try:
    response = test_client.post('/api/classify', json={'text': ''}, content_type='application/json')
    if response.status_code == 400:
        print(f"   ✅ Error handling: Correctly returned 400 for empty text")
    else:
        print(f"   ⚠️  Expected 400, got {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in error handling test: {e}")

# 404 handler
print("\n   Testing 404 handler...")
try:
    response = test_client.get('/api/nonexistent')
    if response.status_code == 404:
        print(f"   ✅ Error handling: Correctly returned 404 for nonexistent endpoint")
    else:
        print(f"   ⚠️  Expected 404, got {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR in 404 test: {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)


