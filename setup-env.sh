#!/bin/bash
# Setup environment files for frontend

echo "Setting up environment files..."

# Create .env.development
cat > frontend/.env.development << EOF
REACT_APP_API_URL=http://localhost:5000/api
EOF

# Create .env.production
cat > frontend/.env.production << EOF
REACT_APP_API_URL=/api
EOF

echo "✓ Environment files created!"
echo "  - frontend/.env.development (for local development)"
echo "  - frontend/.env.production (for Vercel deployment)"

