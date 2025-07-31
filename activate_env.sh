#!/bin/bash

# HackRX 6.0 Virtual Environment Activation Script
# This script activates the virtual environment and sets up the project

echo "🚀 HackRX 6.0 - Activating Virtual Environment"
echo "=" * 50

# Check if virtual environment exists
if [ ! -d "hackrx-env" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python -m venv hackrx-env
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source hackrx-env/bin/activate

# Check Python location
echo "📍 Python location: $(which python)"
echo "🐍 Python version: $(python --version)"

# Check if key packages are installed
echo ""
echo "📦 Checking key packages..."
python -c "
try:
    import fastapi
    print('✅ FastAPI installed')
except ImportError:
    print('❌ FastAPI not installed')

try:
    import openai
    print('✅ OpenAI installed')
except ImportError:
    print('❌ OpenAI not installed')

try:
    import psycopg2
    print('✅ PostgreSQL driver installed')
except ImportError:
    print('❌ PostgreSQL driver not installed')

try:
    import pinecone
    print('✅ Pinecone installed')
except ImportError:
    print('❌ Pinecone not installed')
"

echo ""
echo "🎯 Environment ready! You can now:"
echo "  • Run the API: uvicorn app:app --reload"
echo "  • Test PostgreSQL: python test_postgresql_setup.py"
echo "  • Run comprehensive tests: python test_accuracy_comprehensive.py"
echo ""
echo "💡 To deactivate later, just type: deactivate"
echo "🚀 Happy coding!"
