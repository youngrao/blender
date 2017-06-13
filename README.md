# Blender
## Automated Model Blending
Blending is a ensemble learning method that combines the results of base models by training a higher-level learner on the lower level outputs. It is closely related to Stacked Generalization which was first introduced by Wolpert (1992).

Recent interest in model blending and stacking has grown lately as a result of widespread use and success in Kaggle competitions, with winners often combining [over 30 models](https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335#184498). This script simplifies the model selection process by allowing users to choose any number of base models, the higher-level stacking model, and data stratification. The automated stacker also accommodates many different forms of machine learning tasks such as classification, regression, and image or text analysis. Additional base models not included here can also be used as long as they support `.fit()` and `.predict()` or `.predict_proba()` methods. 

## Description
Although model stacking and blending are extremely similar ideas, there is a subtle difference between them. An automated stacked generalization pipeline is described [here](https://github.com/youngrao/stackedgeneralizer). In contrast, when blending models one always splits the data into two disjoint sets and trains base learners on only one of the two. The other is used to generate predictions which are then used to train the higher-level learner. Thus, blending models has the benefit that it:
1. Is simpler and faster than stacking because fewer models need to be trained
2. Has no information leak - the base models and generalizers use different data.

However, blending is seen as weaker because: 
1. Less data is used 
2. The model might overfit to the trained subset
3. Cross validation is more solid with stacking because you are using n-folds rather than just two.

Thus, blending models can be seen as a quick way of getting combining the results from many different models. With more time and computing power, stacking is generally preferred. 

This script allows for use of different models in the scikit-learn toolbox, as well as XGBoost. Functional support for neural networks has also been added in `BlenderN.py` which assumes usage of Keras. 

## Example: Otto Group Product Classification Challenge
The Otto Group Product Classification Challenge has been Kaggle's more popular competition to date with over 3500 teams competing. Here, we show a quick application of the automated stacking model.

Directions: Download `train.csv`, `BlenderN.py`, [the test set](https://www.kaggle.com/c/otto-group-product-classification-challenge/data) and run `blender-ex.py`.
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

This achieves a 0.44471 score on the private leaderboard, achieving 633rd place (top 18 percentile)
