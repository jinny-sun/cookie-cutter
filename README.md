---
layout: post
title: Behind the scenes of CookieCutter
subtitle: by Jinny Sun
cover-img: /assets/img/cookie.jpg
gh-repo: jinny-sun/cookie-cutter
gh-badge: [star, fork, follow]
tags: [python, nlp]
---

Do you love baking cookies? Have you ever used a recipe that didn't include important nutritional information, such as the calories per cookie? CookieCutter is a web app that predicts the calories per cookie of a given recipe. It also provides personalized portion control suggestions to help you meet your caloric needs.

This app was created during my fellowship at [Insight Data Science](https://insightfellows.com/) and is currently hosted [here](cookie-cutter.xyz).

## How does CookieCutter work?
CookieCutter uses **bag of words** vectorization to convert ingredient lists into numerical predictors that are fed into a pretrained Gradient Boosting Regressor to estimate how many calories are in each cookie. The algorithm represents each recipe as a high-dimensional vector of ingredient names and the amount of ingredient required for the recipe.

## Text processing

Here is an example recipe.

| Recipe Name | Ingredients | Number of Servings | Calories per Serving |
| :---------- |:----------- | :----------------: | :------------------: |
| Best cookie recipe | ¾ cup White Sugar, granulated<br>8 ounces Butter (softened)<br>2 ½ cups All-purpose Flour<br>1 teaspoon Baking Soda | 12 | 214.3 |

The ingredients strings are pre-processed using RegEx and NLTK library. This includes:
- Converting unicode characters to strings
- Separating numeric values from text
- Removing plurals, punctuation, hypenated words, anything in parentheses, whitespace, and stopwords

## NLP
After the ingredient strings was processed, part-of-speech tagging was used to select for nouns to remove unnecessary words. The 20 most frequent bigrams (where the first token was an adjective or noun, and second token was a noun) were also included in the analysis. The number of features was then reduced from 372 to 79 using frequency-based feature selection to only include tokens that appeared in at least 10 recipes. 

## Model Validation

Several regression models were assessed, including linear regression, random forest, gradient boosting, and XGBoost. 

| Model | Coefficient of Determination (R<sup>2</sup>) | Root Mean Squared Error (RMSE) |
| :---- |:---------------------------: | :---------------------: |
| Linear Regression | 0.79 | 52 |
| Random Forest | 0.82 | 50 |
| Gradient Boosting | 0.86 | 43 |
| XGBoost | 0.86 | 43 |

The linear regression performed the worst, most likely due to the large number of features, and inaccurately assigned the largest coefficients, which tells the model the amount of calories per gram of ingredient, to irrelevant features. For example, the largest coefficient was assigned to salt. 

All tree-based methods accurately identified the most important features contributing to calories, such as sugar, chocolate, and flour. Boosting models performed better than the random forest. While gradient boosting and XGBoost had similar performance metrics based on R<sup>2</sup>, gradient boosting was ultimately chosen as the preferred model since XGBoost had a larger over-estimation of the calories of an individual ingredient. 

| 1 cup (ingredient) | True Calories | Predicted<br>(Lin. Reg.) | Predicted<br>(Random Forest) | Predicted<br>(Gradient Boosting) | Predicted<br>(XGBoost) |
| :---- |:-----------: | :---------------: |:----------------: |:----------------: |:----------------: |
| Sugar | 773 | 2,078 | 2,426 | 1,826 | 2,324 |
| Flour | 445 | 1,730 | 2,478 | 1,777 | 1,955 |
| Chocolate | 805 | 2,260 | 3,287 | 2,652 | 2,971 |
| Salt | 0 | 33,303 | 3,567 | 3,637 | 2,755 |
