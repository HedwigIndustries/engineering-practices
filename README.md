# engineering-practices-hw-1
Install poetry:
```
pip install poetry
```
First dependencies:
```
poetry install
```
Update dependencies:
```
poetry update
```
Install formatters and linters -> black, flake, isort
```
poetry add --dev black 
```
```
poetry add --dev flake8
```
```
poetry add --dev isort
```
Install pre-commit:
```
poetry install pre-commit
```
[Virtual enviroment]  
```
poetry shell
```
Black: 
```
black <path to your file> # formatting
black --check <path to your file> # check
```
Isort:
```
isort <path to your file> # formatting
isort --check <path to your file> # check
```
Flake8:
```
flake8 <path to your file> # run linter 
```
Run pre-commit:
```
pre-commit run --all-files
```
Build: 
```
poetru build
```
Run code: 
```
poetry run python3 <path to your file>
```
