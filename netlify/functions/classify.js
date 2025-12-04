// Simple rule-based fallback classifier for when backend is not available
function fallbackClassify(text) {
  const lowerText = text.toLowerCase();
  const predictions = [];
  
  // Simple keyword-based classification
  const toxicKeywords = ['hate', 'stupid', 'idiot', 'dumb', 'kill', 'die', 'worthless'];
  const threatKeywords = ['kill', 'hurt', 'attack', 'destroy', 'bomb'];
  const insultKeywords = ['stupid', 'idiot', 'moron', 'fool', 'dummy', 'loser'];
  
  let hasToxic = toxicKeywords.some(word => lowerText.includes(word));
  let hasThreat = threatKeywords.some(word => lowerText.includes(word));
  let hasInsult = insultKeywords.some(word => lowerText.includes(word));
  
  if (hasToxic) predictions.push({ label: 'toxic', confidence: 0.7 });
  if (hasThreat) predictions.push({ label: 'threat', confidence: 0.75 });
  if (hasInsult) predictions.push({ label: 'insult', confidence: 0.65 });
  
  return {
    text: text,
    predictions: predictions.length > 0 ? predictions : [{ label: 'safe', confidence: 0.9 }],
    model_used: 'fallback_rule_based',
    timestamp: new Date().toISOString(),
    note: 'Backend not configured - using simple rule-based classification. Deploy backend and set BACKEND_URL for ML-based results.'
  };
}

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
    // Log incoming request for debugging
    console.log('Received request:', { 
      method: event.httpMethod, 
      bodyLength: event.body?.length,
      hasBody: !!event.body 
    });

    if (!event.body) {
      return {
        statusCode: 400,
        headers: CORS_HEADERS,
        body: JSON.stringify({ error: 'Request body is empty' })
      };
    }

    let parsedBody;
    try {
      parsedBody = JSON.parse(event.body);
    } catch (parseError) {
      console.error('JSON parse error:', parseError);
      return {
        statusCode: 400,
        headers: CORS_HEADERS,
        body: JSON.stringify({ error: 'Invalid JSON in request body' })
      };
    }

    const { text, threshold } = parsedBody;

    if (!text || typeof text !== 'string' || text.trim() === '') {
      return {
        statusCode: 400,
        headers: CORS_HEADERS,
        body: JSON.stringify({ error: 'Text is required and must be a non-empty string' })
      };
    }

    console.log('Processing text classification for:', text.substring(0, 50) + '...');

    const BACKEND_URL = process.env.BACKEND_URL;
    
    // If backend is not configured, use fallback classifier
    if (!BACKEND_URL) {
      console.log('BACKEND_URL not set - using fallback classifier');
      const result = fallbackClassify(text);
      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify(result)
      };
    }

    // Try to use the backend
    console.log(`Calling backend at: ${BACKEND_URL}/api/classify`);
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, threshold: threshold || 0.5 })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Backend error: ${response.status} - ${errorText}`);
        // Fallback to rule-based if backend fails
        const result = fallbackClassify(text);
        result.note = `Backend unreachable (${response.status}) - using fallback classifier. Deploy backend for ML results.`;
        return {
          statusCode: 200,
          headers: CORS_HEADERS,
          body: JSON.stringify(result)
        };
      }

      const data = await response.json();
      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify(data)
      };
    } catch (backendError) {
      console.error('Backend fetch error:', backendError.message);
      // Fallback to rule-based if backend is unreachable
      const result = fallbackClassify(text);
      result.note = `Backend unreachable - using fallback classifier. Deploy backend for ML results.`;
      return {
        statusCode: 200,
        headers: CORS_HEADERS,
        body: JSON.stringify(result)
      };
    }
  } catch (error) {
    console.error('Unexpected classification error:', error);
    console.error('Error stack:', error.stack);
    
    // Return a fallback response even on error to keep the app working
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: JSON.stringify({ 
        text: 'Error occurred',
        predictions: [{ label: 'error', confidence: 0.0 }],
        error: true,
        error_message: error.message,
        note: 'An unexpected error occurred. Using fallback response.'
      })
    };
  }
};
