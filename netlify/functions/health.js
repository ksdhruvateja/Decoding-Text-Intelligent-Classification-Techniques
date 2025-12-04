const fetch = require('node-fetch');

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
    
    if (!BACKEND_URL) {
      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify({ 
          status: 'frontend_healthy',
          backend_configured: false,
          message: 'Backend URL not configured. Set BACKEND_URL environment variable in Netlify.'
        })
      };
    }

    const response = await fetch(`${BACKEND_URL}/api/health`, {
      method: 'GET',
      timeout: 10000
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
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify({
        frontend_status: 'healthy',
        backend_status: 'unreachable',
        error: error.message
      })
    };
  }
};
