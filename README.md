# Robust portfolio optimization for recommender systems considering uncertainty of estimated statistics
- https://arxiv.org/pdf/2406.10250

This repository contains the code used for the experiments in "Robust portfolio optimization for recommender systems considering uncertainty of estimated statistics" (PRICAI 2024)

Note: This paper is under review.

## Abstract
This paper is concerned with portfolio optimization models for creating high-quality lists of recommended items to balance the accuracy and diversity of recommendations. However, the statistics (i.e., expectation and covariance of ratings) required for mean--variance portfolio optimization are subject to inevitable estimation errors. To remedy this situation, we focus on robust optimization techniques that derive reliable solutions to uncertain optimization problems. Specifically, we propose a robust portfolio optimization model that copes with the uncertainty of estimated statistics based on the cardinality-based uncertainty sets. This robust portfolio optimization model can be reduced to a mixed-integer linear optimization problem, which can be solved exactly using mathematical optimization solvers. Experimental results using two publicly available rating datasets demonstrate that our method can improve not only the recommendation accuracy but also the diversity of recommendations compared with conventional mean--variance portfolio optimization models. Notably, our method has the potential to improve the recommendation quality of various rating prediction algorithms.

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

