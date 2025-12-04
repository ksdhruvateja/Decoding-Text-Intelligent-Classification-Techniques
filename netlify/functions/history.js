const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'GET, DELETE, OPTIONS',
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

  const BACKEND_URL = process.env.BACKEND_URL;
  
  if (!BACKEND_URL) {
    console.log('BACKEND_URL not configured - returning empty history');
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify({ 
        history: [],
        total: 0,
        message: 'Backend not configured - history not available'
      })
    };
  }

  try {
    if (event.httpMethod === 'GET') {
      const limit = event.queryStringParameters?.limit || '50';
      const response = await fetch(`${BACKEND_URL}/api/history?limit=${limit}`, {
        method: 'GET'
      });

      const data = await response.json();

      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify(data)
      };
    } else if (event.httpMethod === 'DELETE') {
      const response = await fetch(`${BACKEND_URL}/api/history/clear`, {
        method: 'DELETE'
      });

      const data = await response.json();

      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify(data)
      };
    } else {
      return {
        statusCode: 405,
        headers: CORS_HEADERS,
        body: JSON.stringify({ error: 'Method not allowed' })
      };
    }
  } catch (error) {
    console.error('History operation error:', error);
    // Return empty history on error
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify({ 
        history: [],
        total: 0,
        note: 'Backend unreachable - history not available'
      })
    };
  }
};
