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
    console.log('Health check called');
    const BACKEND_URL = process.env.BACKEND_URL;
    
    if (!BACKEND_URL) {
      console.log('BACKEND_URL not configured');
      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify({ 
          status: 'frontend_healthy',
          backend_configured: false,
          message: 'Backend URL not configured. Set BACKEND_URL environment variable in Netlify.',
          timestamp: new Date().toISOString()
        })
      };
    }

    console.log('Checking backend at:', BACKEND_URL);

    const response = await fetch(`${BACKEND_URL}/api/health`, {
      method: 'GET'
    });

    const data = await response.json();

    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify({
        frontend_status: 'healthy',
        backend_status: data,
        backend_url: BACKEND_URL
      })
    };
  } catch (error) {
    console.error('Health check error:', error);
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify({
        frontend_status: 'healthy',
        backend_status: 'unreachable',
        error: error.message,
        timestamp: new Date().toISOString()
      })
    };
  }
};
