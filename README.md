# engineering-practices-hw-1
Install poetry:
```
pip install poetry
```
For update dependencies in 'toml'
```
poetry update
```
Install formatters and linters -> black, flake, isort
```
poetry add --dev black
```
```
poetry add --dev flake
```
```
poetry add --dev isort
```
Install pre-commit:
```
poetry add pre-commit
```
[Virtual enviroment]  
```
poetry shell
```
Black: 
```
poetry run black <path to your file> 
```
Isort:
```
poetry run isort <path to your file> 
```
Flake8:
```
poetry run flake8 <path to your file> 
```
