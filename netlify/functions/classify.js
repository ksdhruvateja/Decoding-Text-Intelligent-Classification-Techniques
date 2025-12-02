const fetch = require('node-fetch');

exports.handler = async (event, context) => {
  // Only allow POST
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    const { text } = JSON.parse(event.body);

    if (!text) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'Text is required' })
      };
    }

    // For Netlify deployment, you'll need to deploy your Python backend separately
    // (e.g., on Render, Railway, or Heroku) and update this URL
    const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';
    
    const response = await fetch(`${BACKEND_URL}/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
      body: JSON.stringify(data)
    };
  } catch (error) {
    console.error('Classification error:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ 
        error: 'Failed to classify text',
        details: error.message 
      })
    };
  }
};
