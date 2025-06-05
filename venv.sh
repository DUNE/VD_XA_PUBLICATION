# Check if .venv file already exists
if [ -d ".venv" ]; then
    echo "-> virtual environment already exists."
    source .venv/bin/activate
else
    echo "creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "installing dependencies..."
    pip install -r requirements.txt
    echo "-> virtual environment created and dependencies installed." 
fi