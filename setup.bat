@echo off
echo Setting up Python virtual environment...

python -m venv .venv
call .venv\Scripts\activate

echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

python main.py