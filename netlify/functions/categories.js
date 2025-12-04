const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Content-Type': 'application/json'
};

exports.handler = async (event, context) => {
  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: ''
    };
  }

  if (event.httpMethod !== 'GET') {
    return {
      statusCode: 405,
      headers: CORS_HEADERS,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    const BACKEND_URL = process.env.BACKEND_URL;
    
    // Provide default categories if backend not configured
    const defaultCategories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];
    
    if (!BACKEND_URL) {
      console.log('BACKEND_URL not configured - returning default categories');
      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify({ 
          categories: defaultCategories,
          count: defaultCategories.length,
          source: 'default'
        })
      };
    }

    const response = await fetch(`${BACKEND_URL}/api/categories`, {
      method: 'GET'
    });

    const data = await response.json();

    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify(data)
    };
  } catch (error) {
    console.error('Categories fetch error:', error);
    // Return default categories on error
    const defaultCategories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify({ 
        categories: defaultCategories,
        count: defaultCategories.length,
        source: 'fallback',
        note: 'Backend unreachable - using default categories'
      })
    };
  }
};
