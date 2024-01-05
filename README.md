# robust_recommendation

## Setup
This repository is using poetry + python(^3.9).
If you want to use this repository, please run the following command.

```python
$make env
```

## hyperparameter
This repository is using hydra. So, you can change hyperparameter in command line.
For example, if you want to change the number of epochs, you can run the following command.

```python
$poetry run python3 main.py name=exp001 prediction.epoch=100
```

## Code Formatting
```python
$make format
```

