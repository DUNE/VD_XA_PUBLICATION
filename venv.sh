# Check if .venv file already exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "Virtual environment created and dependencies installed." 
fi