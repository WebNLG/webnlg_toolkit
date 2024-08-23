# Config Poetry to install virtual envs in .venv
poetry config virtualenvs.in-project true

# Install virtual envrionment
poetry install

# Activate
poetry shell

# Install relevant metrics (BLEURT-20, METEOR) (optional)
bash webnlg_toolkit/eval/install_dependencies.sh