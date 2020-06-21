# CookieCutter

Do you love baking cookies? Have you ever used a recipe that didn't include important nutritional information, such as the calories per cookie? CookieCutter is a web app that predicts the calories per cookie of a given recipe. It also provides personalized portion control suggestions to help you meet your caloric needs.

## How does CookieCutter work?
CookieCutter uses **bag of words** vectorization to convert ingredient lists into numerical predictors that are fed into a pretrained Gradient Boosting Regressor to estimate how many calories are in each cookie. The algorithm represents each recipe as a high-dimensional vector of ingredient names and the amount of ingredient required for the recipe.

