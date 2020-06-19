# Import packages
import streamlit as st
import numpy as np
import pandas as pd
import re

# Change web style
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#
# def remote_css(url):
#     st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)
#
# def icon(icon_name):
#     st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)
#
# local_css("style.css")
# remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# Functions
def string_replace(x):
    new_string = re.sub(' {2,}', ' ', x).replace("  ", ';').replace("\n", ";").replace("; ;", ";")
#    new_string = new_string.split(';')
    return(new_string)

def get_ingredients (x):
    ing_regex = ('(\d+/*\d*\s*\d*/*\d*)\s(\w+\s*.*?);')
    all_ing = re.findall(ing_regex, x)
    return(all_ing)

def get_quantity(x):
    quantity = [y[0] for y in x] # use for df
    units_with_ingredient = [y[1] for y in x]
    df_of_units = pd.DataFrame({'quantity':quantity, 'ingredient':units_with_ingredient})
    return (df_of_units)

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()

    def lemmatize(string):
        for word in re.findall(r"[a-z]+", string):
            string = string.replace(word, wnl.lemmatize(word, 'n') if 's' in word[-3:] else word)
        return string

    unit_stopwords = ['dash','pinch','teaspoon','tablespoon','fluid','cup','pint','quart','ounce','oz','pound','rack',
                'small','medium','large','crushed','grated','skinless','boneless','melted','fresh',
                'diced','minced','thinly','dry','dried','halved','taste','frying','lean','drained','jars','grated'
                'clove','slice','eaches','whole','cube','thick','unit','freshly','finely','splash',
                'semisweet','chip','extract','spread','powder','room','temperature','brown','cooking','yolk','ground',
                'package','mix','cake','plain','goody','light','wheat','piece','substitute','mini','kosher','crispy',
                'minature','chunk','dark','bit','square','boiling','bag','crumb','popsicle','stick','zest','cereal',
                'bar','tart','nib','tennessee','turbinado','baking','pack','spice','moist','miniarature','crunchy',
                'morsel','nugget','candy','crisp','super','fine','decoration','sucralose','puree','pureed','rainbow',
                'cut','frozen','broken','round','concentrate','miniature','cooky','virgin','dusting','half','baby',
                'food','jar','seedless','container','box','granule','filling','cold','super','ripe','moisture',
                'packet','instant','mint','ripe','sea','coarse','fun','size','funsize','bulk','chopped','torn']

    # Remove anything in parenthesis
    mess = re.sub(r"\([^\)]+\)", '', mess)
    # Make everything lowercase
    mess = mess.lower()
    # Remove non-word punctuation
    mess =' '.join(re.findall(r"[-,''\w]+", mess)) # This leaves some commas as a character #
    mess = re.sub(r"\,", ' ', mess)
    # Remove hypenated words
    mess = re.sub(r"(?=\S*['-])([a-zA-Z'-]+)",'',mess) # remove hypenated words
    # Remove punctuation and numbers
    mess = ''.join([i for i in mess if not i.isdigit()])
    # Remove plurals
    mess = lemmatize(mess)
    #clean excess whitespace
    mess = re.sub(r"\s+", ' ', mess).strip()
    # Remove stopwords
    mess = [word for word in mess.split() if word.lower() not in stopwords.words('english')]
    mess  = [word for word in mess if word.lower() not in unit_stopwords]
    mess = ' '.join(mess)
    return(mess.split())

def test_noun(tokens):
    import nltk
    tagged = nltk.pos_tag(tokens)
    return([token[0] for token in tagged if token[1] in ['NN',]])

def convert_fractions (quantity):
    from fractions import Fraction
    return float(sum(Fraction(s) for s in quantity.split()))

# App

st.title("CookieCutter")
st.subheader('Cut the calories from your cookie recipe!')

X_train = pd.read_csv('X_train.csv')
# X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
# y_test = pd.read_csv('y_test.csv')

# This includes the list of ingredients before cleaning

ingredient_string = st.text_input('Input the ingredient list here:', '1 cup packed brown sugar; 1 cup white sugar; 1 cup butter; 2 eggs; 1 teaspoon baking soda; 1 teaspoon salt; 1 teaspoon vanilla extract; 2 1/2 cups sifted all-purpose flour; 1/2 cup chopped walnuts; 2 cups semisweet chocolate chips')
if ingredient_string:
    st.write('Ingredients',ingredient_string)

serving_size = st.number_input('How many cookies will be made using this recipe?', 24)
if ingredient_string:
    st.write('This recipe will make',serving_size,'cookies')

desiredcal = st.number_input('What is the maximum number of calories per cookie you desire?', 200)
if ingredient_string:
    st.write('I want',desiredcal,'calories or less')

button = st.button('Get this recipe!')

if button:
    # Process ingredient_string
    serving_size = serving_size
    ingredient_string = ingredient_string + ';' # add semicolon to end of ingredient list for regex
    ingredient_string = string_replace(ingredient_string) # remove white space
    ingredient_string_tuple = get_ingredients(ingredient_string) # separate ingredients into list of tuples
    testdf = get_quantity(ingredient_string_tuple) # separate quantity from words
    testdf['quantity'] = [convert_fractions(x) for x in testdf['quantity']]
    testdf['unit'] = np.where(testdf.ingredient.str.contains("dash"), .3,
                np.where(testdf.ingredient.str.contains("pinch"), .6,
                np.where(testdf.ingredient.str.contains("teaspoon"), 5,
                np.where(testdf.ingredient.str.contains("tablespoon"), 3,
                np.where(testdf.ingredient.str.contains("fluid"), 30,
                np.where(testdf.ingredient.str.contains("cup"), 240,
                np.where(testdf.ingredient.str.contains("pint"), 473,
                np.where(testdf.ingredient.str.contains("quart"), 980,
                np.where(testdf.ingredient.str.contains("ounce"), 28,
                np.where(testdf.ingredient.str.contains("oz"), 28,
                np.where(testdf.ingredient.str.contains("pound"), 454,
                np.where(testdf.ingredient.str.contains("rack"), 908,
                np.where(testdf.ingredient.str.contains("small"), 50,
                np.where(testdf.ingredient.str.contains("medium"), 60,
                np.where(testdf.ingredient.str.contains("large"), 70,
                1)))))))))))))))

    # Total quantity of each ingredient needed for recipe (grams* quantity) and condense into a list.
    testdf['norm_quant'] = round(testdf['unit']*testdf['quantity'])
    testdf['norm_quant'] = testdf['norm_quant'].astype(int)

    st.subheader('Ingredients')
    testdf[['quantity','ingredient']]

    # Tokenization = convert text string into list of tokens, or words, we want (i.e., cleaned version of words).
    import string
    from nltk.corpus import stopwords
    testdf['ingredient']=[text_process(x) for x in testdf['ingredient']]

    # One word per ingredient - keep only nouns, join multiple words as one string
    testdf['ingredient'] = [test_noun(tokens) for tokens in testdf['ingredient']]
    testdf['ingredient'] = [''.join(tokens) for tokens in testdf['ingredient']]

    # Repeat word by normalized quantity
    testdf['ingredient'] = testdf['ingredient'].astype(str) + ' '
    zipped = list(zip(testdf['ingredient'], testdf['norm_quant']))
    inglist = [t[0]*t[1] for t in zipped]
    inglist = ''.join(inglist)
    inglist = [inglist]

    # Calorie Prediction v2
    import pickle
    bow_transformer = pickle.load(open('bow_transformer.sav','rb'))
    ingredient_bow_train = pickle.load(open('ingredient_bow_train.sav','rb'))
    inglist_bow_test =  bow_transformer.transform(inglist)

    # Linear Regression
    # from sklearn.linear_model import LinearRegression
    # linreg = LinearRegression()
    # linreg.fit(ingredient_bow_train,y_train['totalCal'])
    # predictions = linreg.predict(inglist_bow_test)

    # Gradient Boosting
    from sklearn.ensemble import GradientBoostingRegressor
    gboost = GradientBoostingRegressor(loss="ls", learning_rate=0.03, n_estimators=1500, max_depth=7, min_samples_split=950, min_samples_leaf=6, subsample=0.8, max_features=21, random_state=10)
    gboost.fit(ingredient_bow_train, y_train['totalCal'])
    predictions = gboost.predict(inglist_bow_test)

    st.subheader('Calorie Predictor')
    calPerServing = round(predictions[0]/serving_size,1)
    st.write()

    if calPerServing < desiredcal:
        'If you make ', serving_size, 'cookies with this recipe, each cookie is', calPerServing, "calories. That's less than", desiredcal,'calories per cookie :grin:'
    else:
        'If you make ', serving_size, 'cookies with this recipe, each cookie is', calPerServing, "calories. That's more than", desiredcal,'calories per cookie :cry:'
        import math
        new_servings = math.ceil(predictions[0]/desiredcal)
        new_calories = round(predictions[0]/new_servings,1)
        'If you make', new_servings, "cookies instead using the same recipe, each cookie is only", new_calories, "calories. That's less than", desiredcal,'calories per cookie :grin:'
