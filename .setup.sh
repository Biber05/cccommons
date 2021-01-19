# Add repo for Python3.8 and update
sudo apt-get update -y
sudo apt-get install vim -y

# install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
# shellcheck disable=SC1090 disable=SC2086
source $HOME/.poetry/env

# no virtual environment creation
poetry config virtualenvs.create false --local

# update poetry dependencies
poetry update

