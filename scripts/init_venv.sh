python3 -m venv .venv
echo 'PYTHONPATH=./pkg:.' >> .venv/bin/activate
source .venv/bin/activate
pip install -r requirements.txt
