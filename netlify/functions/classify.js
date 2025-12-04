const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
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

  // Only allow POST
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: CORS_HEADERS,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    const { text, threshold } = JSON.parse(event.body);

    if (!text) {
      return {
        statusCode: 400,
        headers: CORS_HEADERS,
        body: JSON.stringify({ error: 'Text is required' })
      };
    }

    // Backend URL should be set in Netlify environment variables
    // Go to: Site settings → Environment variables → Add variable
    // Name: BACKEND_URL, Value: https://your-backend-url.com
    const BACKEND_URL = process.env.BACKEND_URL;
    
    if (!BACKEND_URL) {
      return {
        statusCode: 503,
        headers: CORS_HEADERS,
        body: JSON.stringify({ 
          error: 'Backend not configured',
          message: 'Please set BACKEND_URL environment variable in Netlify dashboard'
        })
      };
    }

    console.log(`Calling backend at: ${BACKEND_URL}/api/classify`);
    
    const response = await fetch(`${BACKEND_URL}/api/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, threshold: threshold || 0.5 })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend returned ${response.status}: ${errorText}`);
    }

    const data = await response.json();

    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify(data)
    };
  } catch (error) {
    console.error('Classification error:', error);
    return {
      statusCode: 500,
      headers: CORS_HEADERS,
      body: JSON.stringify({ 
        error: 'Failed to classify text',
        details: error.message 
      })
    };
  }
};
