# Blender
## Automated Model Blending
Blending is a ensemble learning algorithm that combines the results of base models by training a higher-level learner on the lower level outputs. This idea is closely related to Stacked Generalization which was first introduced by Wolpert (1992) with the basic idea as follows:

1. Split the training set into two disjoint sets.
2. Train several base learners on the first part.
3. Test the base learners on the second part.
4. Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.

Recent interest in model blending and stacking has grown lately as a result of widespread use and success in Kaggle competitions, with winners often combining [over 30 models](https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335#184498) to great sucess. This script simplifies the model selection process by allowing users to choose any number of models, number of folds used for cross-validation, the higher-level stacking model, and data stratification. The automated stacker also accomodates many different forms of machine learning tasks such as classification, regression, and image and text analysis. Additional base models not included here can also be used as long as they support `.fit()` and `.predict()` or `.predict_proba()` methods. 

## Description
Furthermore, although model stacking and blending are extremely similar ideas, the subtle difference between this algorithm and the blending script is described [here](https://github.com/youngrao/blender). 

Functional support for neural networks has also been added in `BlenderN.py` which assumes usage of Keras. 

## Example: Otto Group Product Classification Challenge
The Otto Group Product Classification Challenge has been Kaggle's more popular competition to date with over 3500 teams competing. Here, we show a quick application of the automated stacking model.

Directions: Download `train.csv`, `BlenderN.py`, [the test set](https://www.kaggle.com/c/otto-group-product-classification-challenge/data) and run `stacker-ex.py`.
This example combines the following 12 models:
1. Random Forest with 100 Trees
2. Random Forest with 500 Trees
3. Random Forest with 1500 Trees
4. XGBoost with 200 rounds of boosting
5. XGBoost with 400 rounds of boosting
6. XGBoost with 600 rounds of boosting
7. K-Nearest Neighbors with n=10
8. K-Nearest Neighbors with n=50
9. K-Nearest Neighbors with n=100
10. Neural Network with one 50 node hidden layer
11. Neural Network with one 100 node hidden layer
12. Neural Network with one 200 node hidden layer

with a XGBoost as a higher-level learner. 

This achieves a 0.41783 score on the private leaderboard, achieving #116th place (top 3.3 percentile)
