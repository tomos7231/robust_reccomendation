# Robust portfolio optimization for recommender systems considering uncertainty of estimated statistics
- https://arxiv.org/pdf/2406.10250

## Setup
This repository is using rye.  
If you want to use this repository, please run the following command.

1. install rye
   - install instructions: https://rye-up.com/guide/installation/
2. enable uv to speed up dependency resolution.
```
rye config --set-bool behavior.use-uv=true
```
3. create a virtual environment
```
rye sync
```

## hyperparameter
This repository is using hydra. So, you can change hyperparameter in command line.  
For example, if you want to change the number of epochs, you can run the following command.

```python
$rye run python3 main.py name=exp001 prediction.epoch=100
```

## Code Formatting
```python
$make format
```

