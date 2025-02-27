echo "Setting up Python virtual environment..."

python3 -m venv .venv
source .venv/bin/activate

echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

python3 main.py