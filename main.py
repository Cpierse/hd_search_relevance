# -*- coding: utf-8 -*-
"""
The final version of our algorithm to predict the relevance of home depot's 
search results. This version includes all function created to explore the data,
extract predict features, and make predictions.

In the future, I would use Google's Python convention and break this code into
multiple python files, each focusing on a different aspect of the problem.
@author: Cpierse
"""
############################## IMPORTS ###################################
# Math and database handling:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, json
# Word-related algorithms:
#import enchant # PyEnchant is a spellchecking library for Python.
import re
from nltk.stem.snowball import SnowballStemmer # Breaks words down to root words... somehow.
from nltk.corpus import stopwords # we want to remove these words
#from nltk.stem import WordNetLemmatizer # FOR WHEN LIFE GIVES YOU LEMMAS
## PREPROCESSING AND MODEL TESTING:
#from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler # Processing Data for algorithms
from sklearn.feature_extraction.text import TfidfVectorizer # Processing term frequency
from sklearn.cross_validation import cross_val_score#, KFold # Evaluating our methods
#from sklearn.pipeline import Pipeline #Used to implement several things in a row to data such as scale -> fit 
from sklearn.feature_selection import SelectKBest#, f_classif
from sklearn.decomposition import TruncatedSVD
## POTENTIAL MACHINE LEARNING MODELS:
#from sklearn.naive_bayes import MultinomialNB # A model for bayesian probability - assumes no coorelation in terms
#from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor # BUNCH OF TREES
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV, LarsCV #, Lars, Lasso

######################### KEY VARIABLES ################################
FIRST_RUN = False

# TODO: There appears to be a bug with processing the unicode from scratch.
# This obviously wasn't an issue earlier, so what changed?

############################# DATA ####################################
if FIRST_RUN:
    ## The raw data - Uncomment and load as needed ##
    train = pd.read_csv('train.csv', encoding="ISO-8859-1")
    test = pd.read_csv('test.csv', encoding="ISO-8859-1")
    desc = pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1")
    attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")
else:
    ## Processed Data ##
    attr_vals = pd.read_json('attr_vals.json')
    desc = pd.read_json('desc.json')
    brands = pd.read_json('brands.json')
    combo = pd.read_json('combo.json')
    combo_feats = pd.read_json('combo_feats.json')
    combo_ness = pd.read_json('combo_ness.json')
    combo_tt_feats = pd.read_json('combo_tt_feats.json')
    print('Loading Jsons Complete')

## Spell checked dictionary ##
with open('googleSpellCheckedDict.json', 'r') as fp:
    spell_check_dict = json.load(fp)
    
############################## KEY FUNCTIONS TO REF ###########################

def DirectLeftoverList( WordList, CheckList ):
    # Checks if Wordlist is in Checklist. One to One
    # Example: DirectLeftoverList( combo['search_list'], combo['product_title'] )
    ResultList = list(map(lambda x: [1]*len(x), WordList))
    word = ''
    for i in range(0,len(WordList)):
        for j in range(0,len(WordList[i])):
            word = WordList[i][j]
            if word in CheckList[i]:
                ResultList[i][j] = 0
    return ResultList
    
def IndirectLeftoverList( WordList, CheckList , MapW, MapC):
    # WordList = combo['search_list']; CheckList = desc["product_description"]
    # MapW =  combo["product_uid"]; MapC =  desc["product_uid"];
    ResultList = list(map(lambda x: [1]*len(x), WordList) )
    Frac = pd.DataFrame({'frac':[-1.0]*len(WordList)}) #map(lambda x: -1, WordList)  # Checks if any W map to C. Default is no = -1
    for i in range(0,len(CheckList)):
        r_items = (MapW == MapC[i])# relevant items
        Frac.frac[r_items] = 0.0
        indices = Frac.frac[r_items].index.get_values() # Note some are empty. These belong to the test set.
        desc_now = CheckList[i]
        index = 0
        for search_lists in WordList[r_items]:
            sub_index = 0
            for search_word in search_lists:
                if search_word in desc_now:
                    Frac.frac[indices[index]] += 1.0/len(search_lists)
                    ResultList[indices[index]][sub_index] = 0
                sub_index += 1
            index += 1             
    return ResultList, Frac

stemmer = SnowballStemmer("english")
stopset = set(stopwords.words('english'))
def StopAndStem(listOfWords): # Remove stop words and stem a list of individual words
    newList = []    
    for x in listOfWords:
        if x not in stopset:
            newList.append(str(stemmer.stem(x)))
    return newList

def FindNaN(data):
    return  np.where(np.isnan(np.array(data)))

def CheckTheBest(predictors,train, k =6):
    #selector = SelectKBest(f_classif, k=k)
    from sklearn.feature_selection import f_regression
    selector = SelectKBest(f_regression, k=k)
    selector.fit(train[predictors], train["relevance"])
    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)
    # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)),predictors, rotation='vertical')
    plt.show()
    feats = [predictors[i] for i,x in enumerate(selector.get_support()) if x]
    return feats

# Edit this one as needed to include new models:
def CrossValStuff(model_number, predictors, train):
    # Try a model and cross-validate that shit:
    if model_number == 0:
        alg = RandomForestRegressor(random_state = 777)
    elif model_number == 1:
        alg = GradientBoostingRegressor(random_state=777)
    elif model_number == 2:
        alg = ExtraTreesRegressor(random_state=777)
    elif model_number == 3:
        alg = SVR(kernel = 'linear')
    elif model_number == 4:
        alg = SVR(kernel = 'rbf')
    elif model_number == 5:
        b_alg = RandomForestRegressor(random_state=777)
        alg = BaggingRegressor(b_alg, random_state=7777)
    elif model_number == 6:
        b_alg = GradientBoostingRegressor(max_depth = 6, random_state=777)
        alg = BaggingRegressor(b_alg, random_state=7777)
    elif model_number==7:
        alg = LassoCV()
    elif model_number == 8:
        alg = LarsCV()
    scores = cross_val_score(alg, train[predictors],train.relevance, cv=3, scoring='mean_squared_error')
    print(scores)
    print( 'Model: %d Score: %f' %(model_number,scores.mean()))  
    # Note the fucking score sign is flipped.

def SplitCombo(combo_feats,combo):
    train = combo_feats.loc[combo.istrain,:]
    test = combo_feats.loc[~combo.istrain,:]
    return train,test

# Format everything like Nuo:
def LowercaseAndReplace(column):
    for ind, line in enumerate(column):
        #if type(line)==unicode:
        #    line = line.encode('utf-8')
        #    line = re.sub(r'[^\x00-\x7F]+',' ', line)
        #else:
        line = str(line)
        new_line = re.sub('[^a-zA-Z0-9\n\~\=\_\-\,\;\:\!\?\/\.\'\(\)\[\]\$\*\\\&\#\%\+\"]', ' ', line)
        new_line = new_line.lower().replace("&#39;", ' ')
        new_line = new_line.replace("&quot;", ' ')
        new_line = new_line.replace("&amp;", ' ')
        new_line = new_line.replace('  ', ' ')
        column[ind] = new_line
    return column

def Strip(column): # This does not help at all. It makes things worse. Do not use.
    for ind, line in enumerate(column):
        line = str(line)
        pattern = re.compile('([^\s\w]|_)+')
        new_line = pattern.sub('', line)
        column[ind] = new_line
    return column

def FixUnits(column):
    for ind, line in enumerate(column):
        new_line = str(line)
        # Replacing units from NUMBER UNIT and NUMBER UNITx:
        # Removed the space between number and unit. Also added the by. If this decreases accuracy, change it back.
        # Score dropped. Put the space back. The issue may really be with the spellchecker fucking things up.
        # Or maybe I don't unify all units
        new_line = re.sub('(\s?)([0-9]+)\s?\(?((ft|feet|foot|\')\.?)\)?s?x?(by)?', '\g<1>\g<2> foot \g<5>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?([aA]|[aA]mp)\)?s?x?(by)?', '\g<1>\g<2> amp \g<4>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?(gal(lon|\.))\)?s?x?(by)?', '\g<1>\g<2> gallon \g<5>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?(in(\.|ches|ch|))\)?s?x?(by)?', '\g<1>\g<2> inch \g<5>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?(cm)\)?s?x?(by)?', '\g<1>\g<2> centimeter \g<4>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?(m)\)?s?x?(by)?', '\g<1>\g<2> meter \g<4>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?(ounce|oz)\)?s?x?(by)?', '\g<1>\g<2> ounce \g<4>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?((lb|pound)\.?)\)?s?x?(by)?', '\g<1>\g<2> pound \g<5>', new_line )
        new_line = re.sub('(\s?)([0-9]+)\s?\(?((sq\s?\.?\s?ft\s?\.?)|(square feet)|(square foot))\)?x?((by)?)', '\g<1>\g<2> sqft \g<6>', new_line ) 
        new_line = re.sub('(\s?)([0-9]+)\s?\(?([vV]olt|[vV])\)?s?x?(by)?', '\g<1>\g<2> volt \g<4>', new_line ) 
        new_line = re.sub('(\s?)([0-9]+)\s?\(?([Ww]att|[wW])\)?s?x?(by)?', '\g<1>\g<2> watt \g<4>', new_line ) 
        # Replacing some other units:
        new_line = re.sub('(\s?)(feet)', '\g<1> foot ', new_line ) # Make singular
        #new_line = re.sub('(square feet)|(square foot)|(sq. ft.)', 'sqft ', new_line ) # Universalize square feet
        new_line = re.sub('(\s?)(by|x)\s?([0-9])', ' by \g<3>', new_line ) 
        new_line = re.sub('(\s?)(qt.?)|(quarts?)', ' quart', new_line ) # no need for special stuff with quart
        new_line = new_line.replace('  ', ' ') # remove any double spaces
        column[ind] = new_line
    return column

def AttrUnits(column):
    # Height Width Depth (units): number -> number units
    for ind,line in enumerate(column):
        new_line = re.sub('\s\((.{0,5})\): ([0-9]+\.?[0-9]*)',': \g<2> \g<1>',str(line))
        new_line = new_line.replace('Bullet','')        
        column[ind] = new_line
    return column

def longest_common_substring(s1, s2):
    # From Wikibooks:
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


######################## MAKING THE SUBMISSIONS ############################
    
def ChooseFeatures(normal = False, stem = False, shared = True, ness = False, extras = False, extra_ness=False, new=False):
    predictors = []
    normal_predictors = ['frac_in_title','frac_in_desc','frac_in_attr','leftover_frac_td','num_search'] 
    stem_predictors = ['stem_frac_title','stem_frac_desc','stem_frac_attr','stem_leftover_frac_td','num_stem'] #last is new
    shared_predictors = ['shortest_word_length','longest_word_length']
    ness_feats = GetNessFeats()
    extra_stem = ['long_len_stem_title','long_len_stem_desc','long_len_stem_attr']
    p_ness_feats = ['p_' + x for x in ness_feats]
    s_ness_feats = ['s_' + x for x in ness_feats]    
    extras = ['has_attr', 'num_prod_appear','num_search_appear'] 
    #unused = ['long_len_stem_brand'] # so far doesn't contribute, but also doesn't hurt much
    ness_left = list(map(lambda x: 's_left_' + x, ness_feats))
    new_feats = ['stem_leftover_frac_tda','num_units','frac_units_found','num_numbers','frac_numbers_found']
    if normal:
        predictors = predictors + normal_predictors
    if stem:
        predictors = predictors + stem_predictors + extra_stem
    if shared:
        predictors = predictors + shared_predictors
    if ness:
        predictors = predictors + p_ness_feats + s_ness_feats
    if extras:
        predictors = predictors + extras
    if extra_ness:
        predictors = predictors + ness_left
    if new:
        predictors = predictors + new_feats
    print('Features Ready!')
    return predictors


def MakeSubmission(train, test, features, n_tree = 100, max_depth = 9, name = "MehCurrentTry.csv"):
    print('Bagged gradient boosted trees, n_tree = ' + str(n_tree) + ', max_depth = ' + str(max_depth) )
    b_alg = GradientBoostingRegressor(max_depth = max_depth, random_state=777, n_estimators = n_tree)
    alg = BaggingRegressor(b_alg, random_state=7777)
    alg.fit(train[features],train.relevance)
    predictions = alg.predict(test[features])
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    predictions[predictions>3] = 3
    submission = pd.DataFrame({
            "id": test["id"],
            "relevance": predictions
        })
    submission.to_csv(name, index=False)
    # Best so far is max_depth = 11, actually, 9 is slightly better
    print('Submission Ready! Go Submit ' + name)
    # 100 to 200 trees gives an improvement of ~ 0.00070

def Gridsearch_GBR(train, test, features, n_tree = 100):
    from sklearn.grid_search import  GridSearchCV
    start = time.time()
    param_grid = {'max_depth': [x for x in range(3,20)]}#int(len(features)/2))]}
    param_grid = {'learning_rate': [0.03, 0.01, 0.03, 0.1, 0.3 ]}#int(len(features)/2))]}
    param_grid = {'learning_rate': [ x for x in np.arange(0.09,0.2,0.01) ]}#int(len(features)/2))]}
    b_alg = GradientBoostingRegressor(max_depth = 9, random_state=777, n_estimators = n_tree)
    #alg = BaggingRegressor(b_alg, random_state=7777)
    alg_search = GridSearchCV(b_alg,param_grid,n_jobs=4)  
    alg_search.fit(train[features],train.relevance)
    print(time.time()-start)
    predictions = alg_search.predict(test[features])
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    predictions[predictions>3] = 3
    submission = pd.DataFrame({
            "id": test["id"],
            "relevance": predictions
        })
    submission.to_csv("MehGridTry.csv", index=False)
    print(alg_search.best_score_)
    print(alg_search.best_estimator_)
    print(alg_search.grid_scores_)
    return alg_search    

def MakeRandomSearchSubmission(train, test, features, model = 'RF', n_iter=100):
    from sklearn.grid_search import  RandomizedSearchCV #GridSearchCV
    from scipy.stats import randint as sp_randint
    if model == 'RF':
        param_dist = {"max_depth": sp_randint(3,len(features)),
                  "max_features": sp_randint(3,len(features)),
                  "min_samples_split": sp_randint(10,np.int(np.sqrt(len(train)))),
                  "min_samples_leaf": sp_randint(10,np.int(np.sqrt(len(train))/2)),
                  "bootstrap": [True, False]}
        alg = RandomForestRegressor(random_state = 777, n_estimators = 25, n_jobs = 2)
    # Else another model?
    alg_search = RandomizedSearchCV(estimator=alg, param_distributions=param_dist, n_iter=n_iter)
    alg_search.fit(train[features],train.relevance)
    print(alg_search.best_score_)
    print(alg_search.best_estimator_)
    predictions = alg_search.predict(test[features])
    predictions[predictions>3] = 3
    submission = pd.DataFrame({
            "id": test["id"],
            "relevance": predictions
        })
    submission.to_csv("MehCurrentTry.csv", index=False)
    print('Submission Ready!')

def BestFeatsMaybe(train, test, features, model='lasso', submit=False):
    # Lasso and Lars both try to find the best features and do a fit similar to linear regression
    # Maybe add Ridge    
    if model=='lasso':  alg = LassoCV(); model_cv = 7;
    elif model == 'lars':   alg = LarsCV(); model_cv = 8;
    # Fit the Model:    
    alg.fit(train[features],train.relevance)
    # Print the non-zero coefficients:
    print(np.sum(alg.coef_ != 0))
    CrossValStuff(model_cv,features,train)
    mask = alg.coef_ != 0
    mask = [i for i, elem in enumerate(mask, 1) if elem]
    a_feat = np.array(features)
    bm_features = list(a_feat[mask])
    if submit: MakeSubmission(test, train, bm_features)
    return bm_features


####################### CLEANING THE DATA ###############################
##### Combining the attributes for one product #####
def CleanAttrVals(strip = False): # Combine the attr name and val
    attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")
    attr_vals = pd.DataFrame({
            "product_uid": np.array(np.unique(attr.product_uid[~np.isnan(attr.product_uid)]))
        })
    attr_vals.loc[:,'product_uid'] = attr_vals.loc[:,'product_uid'].astype(int)
    attr_vals["value"] = ""
    for ind,prod in enumerate(attr_vals.product_uid):
        df = attr.loc[attr.product_uid == prod,['name','value']]
        attr_vals.loc[ind,'value'] = " // ".join(map(str,df['name'] + ': ' + df['value']))
    # Fixing the unit references in these values:
    print('Cleaning time')
    attr_vals['value'] = AttrUnits(attr_vals['value'][:])
    if strip: attr_vals['value'] = Strip(attr_vals['value'][:]) 
    attr_vals['value'] = FixUnits(attr_vals['value'][:])
    #attr_vals['value'] = attr_vals.value.apply(lambda x: str.lower(str(x)))
    #attr_vals.to_csv('attr_vals2.csv', index=False)
    #attr_vals['value'] = LowercaseAndReplace(attr_vals['value'][:])    
    attr_vals["stem_value"] = attr_vals.value.apply(lambda x: " ".join(StopAndStem(str.split(x))))    
    #attr_vals.to_hdf('attr_vals.h5','df')
    attr_vals.to_json('attr_vals.json',orient='records')
    print('Attributes Cleaned')
    return attr_vals

def Brands(strip = False): # A dataframe of product_uid, brand, stem_brand
    attr = pd.read_csv('attributes.csv', encoding="ISO-8859-1")
    brands = attr.loc[(attr.name == 'MFG Brand Name'),['product_uid','value']]
    brands = brands.loc[brands['value'] != 'Unbranded']
    if strip: brands['value'] = Strip(brands['value'][:])
    brands['stem_value'] = brands.value.apply(lambda x: " ".join(StopAndStem(str.split(str(x)))))    
    brands.to_json('brands.json',orient='records')
    print('Brands Prepped')

def CleanDesc(strip = False):
    desc= pd.read_csv('product_descriptions.csv', encoding="ISO-8859-1")
    if strip: desc['product_description'] = Strip(desc['product_description'][:]) 
    desc['product_description'] = FixUnits(desc['product_description'][:])
    desc['product_description'] = desc['product_description'].apply(lambda x: str.lower(str(x)))
    desc["stem_desc"] =  desc["product_description"].apply(lambda x: " ".join(StopAndStem(str.split(x))))
    # list(map(lambda x: " ".join(StopAndStem(str.split(x))), desc["product_description"])
    #desc.to_hdf('desc.h5','df')
    desc.to_json('desc.json',orient='records')
    print('Descriptions Cleaned')
    return desc

def CreateComboPlus(convert = False, strip = False):
    start = time.time()
    train = pd.read_csv('train.csv', encoding="ISO-8859-1")
    test = pd.read_csv('test.csv', encoding="ISO-8859-1")
    train['istrain'] = True
    test['istrain'] = False
    combo = pd.concat((train, test), axis=0, ignore_index=True)
    print('Combo created!')
    if convert:  ## Conrad's corrections are the google ones with all ratios to fractions
        sp = pd.read_csv('Conrad//new_search_terms.csv',usecols=['term','corrected_term'])    
        sp = sp.set_index('term')
        combo['search_list'] = list(map(lambda x: sp.at[x,'corrected_term'].replace(';;',' '),combo['search_term']))
    else:
        combo['search_list'] = combo['search_term']
        for key,value in spell_check_dict.items():
            combo.loc[combo['search_term']==key,'search_list']=value
    print('Combo spellchecked!')
    if strip: combo['search_list'] = Strip(combo['search_list'][:]) 
    else: combo['search_list'] = LowercaseAndReplace(combo['search_list'][:]) # Leaves the search term in-tact
    combo['search_list'] = FixUnits(combo['search_list'][:])
    combo['search_list'] =  list(map(lambda x: str.split(x), combo['search_list'][:]))
    print(time.time()-start)
    combo['product_title'] = LowercaseAndReplace(combo['product_title'][:]) # Left product_uid in-tact
    combo['product_title'] = FixUnits(combo['product_title'][:])
    print(time.time()-start)
    # The Stemmed Features:
    combo['stem_search'] = list(map(lambda x: StopAndStem(x), combo['search_list'][:]))
    combo['stem_title'] = list(map(lambda x: " ".join(StopAndStem(str.split(x))), combo['product_title'][:]))
    combo['stem_search_term'] = list(map(lambda x: " ".join(x), combo['search_list'][:]))
    combo = LongestCombo(combo)    
    #combo.to_hdf('combo.h5','df')
    combo.to_json('combo.json',orient='records')
    print(time.time()-start)
    return combo
    

def LongestCombo(combo): # Find longest common substring between stemmed combo and stuff.
    # I could make this more effient by comparing only the unique search terms, then filling in combo.
    desc = pd.read_json('desc.json')
    desc = desc.set_index('product_uid')
    combo['longest_stem_desc'] = list(map(lambda x,y: longest_common_substring(x, y),combo['stem_search_term'],desc.ix[combo.product_uid,'stem_desc']))
    combo['longest_stem_title'] = list(map(lambda x,y: longest_common_substring(x, y),combo['stem_search_term'],combo['stem_title']))
    #attr_vals = pd.read_hdf('attr_vals.h5','df')
    pids = np.unique(combo.product_uid); np_pids = pids[~np.in1d(pids,attr_vals.product_uid)]
    attr_add = pd.DataFrame({
            "product_uid": np_pids,
            "value":'',
            "stem_value":''
        })
    attr_vals_p = pd.concat([attr_vals,attr_add])
    attr_vals_p = attr_vals_p.set_index('product_uid')
    combo['longest_stem_attr'] = list(map(lambda x,y: longest_common_substring(x, y),combo['stem_search_term'],attr_vals_p.ix[combo.product_uid,'stem_value']))    
    np_pids = pids[~np.in1d(pids,brands.product_uid)]
    brands_add = pd.DataFrame({
            "product_uid": np_pids,
            "value":'',
            "stem_value":''
        })
    brands_p = pd.concat([brands,brands_add])
    brands_p = brands_p.set_index('product_uid')
    brands_p.loc[:,'stem_value'] = brands_p.loc[:,'stem_value'].astype(str)
    combo['longest_stem_brand'] = list(map(lambda x,y: longest_common_substring(x, y),combo['stem_search_term'],brands_p.ix[combo.product_uid,'stem_value']))    
    return combo

def StartUp():
    attr_vals = CleanAttrVals()
    print('Attr Cleaned')
    desc = CleanDesc()
    print('Desc Cleaned')    
    combo = CreateComboPlus()
    print('Combo Complete')
    brands = Brands()


####################### CREATING THE FEATURES ###############################
# Start with some basic features
def BasicFeatures(combo):
    print('Starting Basic Features')
    start = time.time()
    combo_feats = pd.DataFrame({
        "id": combo['id'],
        "relevance": combo['relevance']
    })
    combo_feats["num_search"] = list(map(lambda x: len(x), combo["search_list"][:]))
    combo_feats["shortest_word_length"] =  0
    combo_feats["longest_word_length"] =  0
    print(time.time()-start)
    for i in range(0,len(combo)):
        combo_feats.at[i,"shortest_word_length"] = np.min(list(map(lambda x: len(x), combo["search_list"][i]) ))
        combo_feats.at[i,"longest_word_length"]  = np.max(list(map(lambda x: len(x), combo["search_list"][i]) ))
    print(time.time()-start)
    combo_feats['num_stem'] =  list(map(lambda x: len(x), combo['stem_search'][:]) )
    combo_feats['long_len_stem_title'] =  list(map(lambda x: len(x), combo['longest_stem_title'][:]) )
    combo_feats['long_len_stem_desc'] =  list(map(lambda x: len(x), combo['longest_stem_desc'][:]) )
    combo_feats['long_len_stem_attr'] =  list(map(lambda x: len(x), combo['longest_stem_attr'][:]) )
    combo_feats['long_len_stem_brand'] =  list(map(lambda x: len(x), combo['longest_stem_brand'][:]) )
    #combo_feats.to_hdf('combo_feats.h5','df')
    combo_feats.to_json('combo_feats.json',orient='records')
    print(time.time()-start)
    print('Mission comprete!')
    return combo_feats
#BasicFeatures(combo)
    
def FindLeftovers(combo,combo_feats):
    start = time.time()
    combo["title_leftovers"] = DirectLeftoverList( combo['search_list'][:], combo['product_title'][:] )
    combo_feats["frac_in_title"] = list(map(lambda x: 1-np.mean(x), combo['title_leftovers'][:]))
    print(time.time()-start)
    combo["desc_leftovers"], combo_feats["frac_in_desc"] = IndirectLeftoverList( combo['search_list'][:], desc['product_description'][:] , combo['product_uid'][:], desc['product_uid'][:])
    print(time.time()-start)
    combo["attr_leftovers"], combo_feats["frac_in_attr"] = IndirectLeftoverList( combo['search_list'][:], attr_vals['value'][:] , combo['product_uid'][:], attr_vals['product_uid'][:])
    print(time.time()-start)
    combo_feats['leftover_frac_td'] = list(map(lambda x,y: np.mean(np.multiply(x,y)), combo['title_leftovers'][:],combo['desc_leftovers'][:])) # Ignores attributes, customers don't see them.
    print('Mission comprete!')
    #combo.to_hdf('combo.h5','df')
    #combo_feats.to_hdf('combo_feats.h5','df')
    combo_feats.to_json('combo_feats.json',orient='records')    
    ## Stemmed:
    print('Onto Stemmed!')
    combo["stem_title_leftovers"] = DirectLeftoverList( combo['stem_search'][:], combo['stem_title'][:] )
    combo_feats["stem_frac_title"] = list(map(lambda x: 1-np.mean(x), combo['stem_title_leftovers'][:]))
    print(time.time()-start)
    combo["stem_desc_leftovers"], combo_feats["stem_frac_desc"] = IndirectLeftoverList( combo['stem_search'][:], desc['stem_desc'][:] , combo['product_uid'][:], desc['product_uid'][:])
    print(time.time()-start)
    #print('Mission comprete!')
    combo["stem_attr_leftovers"], combo_feats["stem_frac_attr"] = IndirectLeftoverList( combo['stem_search'][:], attr_vals['stem_value'][:] , combo['product_uid'][:], attr_vals['product_uid'][:])
    print(time.time()-start)
    print('Mission comprete!')
    combo_feats['stem_leftover_frac_td'] = list(map(lambda x,y: np.mean(np.multiply(x,y)), combo['stem_title_leftovers'][:],combo['stem_desc_leftovers'][:])) # Ignores attributes, customers don't see them.
    # There are some Na's from the null stem searches that came from shit, e.g. "to":
    stem_predictors = ['stem_frac_title','stem_frac_desc','stem_frac_attr','stem_leftover_frac_td','num_stem']    
    combo_feats[stem_predictors] = combo_feats[stem_predictors].fillna(0)    
    combo['stem_search_term'] = list(map(lambda x: " ".join(x), combo['stem_search'][:]))
    #combo.to_hdf('combo.h5','df')
    #combo_feats.to_hdf('combo_feats.h5','df')
    combo['stem_all_leftovers'] = list(map(lambda x,y,z: np.multiply(x,np.multiply(y,z)),combo.stem_title_leftovers[:], combo.stem_attr_leftovers[:], combo.stem_desc_leftovers[:] ))
    combo_feats['stem_leftover_frac_tda'] = list(map(lambda x: np.mean(x), combo['stem_all_leftovers'][:]))
    combo_feats.loc[np.isnan(combo_feats['stem_leftover_frac_td']),'stem_leftover_frac_td'] = 0
    combo_feats.loc[np.isnan(combo_feats['stem_leftover_frac_tda']),'stem_leftover_frac_td'] = 0
    combo.to_json('combo.json',orient='records')
    combo_feats.to_json('combo_feats.json',orient='records')
    print(time.time()-start)
    print('Mission comprete!')
    return combo, combo_feats
#FindLeftovers(combo,combo_feats)

def GetNessFeats():
    ness_feats = ['adhesiveness', 'bathness',  'cleanness', 'doorness',\
        'fanness', 'fasteness', 'floorness', 'gardenness',\
       'heatness',  'kitchenness', 'lightness', 'lumberness',\
       'paintness', 'paperness', \
       'roofness',  'sheetrockness',\
       'stainness', 'storageness', 'tileness', 'toolness', 'waterness',\
       'windowness', 'wireness', 'woodness']
    return ness_feats

## Processing Conrad's *Ness files
def NessFeatures(combo_feats, method = 'diff'):
    #combo_feats_plus = combo_feats
    con_prod = pd.read_csv('Conrad//products.csv',encoding="ISO-8859-1")
    #con_search = pd.read_csv('Conrad//search_terms.csv',encoding="ISO-8859-1")
    #con_search = pd.read_csv('Conrad//google_search_terms.csv',encoding="ISO-8859-1")
    con_search = pd.read_csv('Conrad//new_search_terms.csv',encoding="ISO-8859-1")
    con_search =  con_search.set_index('term')
    con_prod = con_prod.set_index('product_uid')
    ness_feats = GetNessFeats()
    p_ness_feats = ['p_' + x for x in ness_feats]
    s_ness_feats = ['s_' + x for x in ness_feats]
    combo_ness = pd.DataFrame({
        "id": combo['id'],
    })
    start = time.time()
    #for feat in ness_feats:
    #    combo_feats_plus[feat] = 0
    for feat in ness_feats + p_ness_feats + s_ness_feats:
        combo_ness[feat] = 0
    sts = con_search.index.values
    pids = con_prod.index.values
    print(time.time()-start)
    # Search first
    vals = np.array(con_search[ness_feats])
    st_d = dict(zip(sts,vals));
    st_nf = []
    for st in combo.search_term:
        st_nf.append( list(st_d[st].tolist() ))
    combo_ness[s_ness_feats] = st_nf
    # Prods next
    vals = np.array(con_prod[ness_feats])
    prod_d = dict(zip(pids,vals));
    prod_nf = []
    for pid in combo.product_uid:
        prod_nf.append( list(prod_d[pid].tolist() ))
    combo_ness[p_ness_feats] = prod_nf
    if method == 'diff':
        #combo_feats_plus.loc[:,ness_feats] = abs(np.array(st_nf)-np.array(prod_nf))
        combo_ness.loc[:,ness_feats] = abs(np.array(st_nf)-np.array(prod_nf))
    if method == 'prod':
        #combo_feats_plus.loc[:,ness_feats] = np.multiply(np.array(st_nf),np.array(prod_nf))
        combo_ness.loc[:,ness_feats] = np.multiply(np.array(st_nf),np.array(prod_nf))
    #combo_feats_plus.to_hdf('combo_feats_plus.h5','df')
    #combo_feats_plus.to_json('combo_feats_plus.json',orient='records')
    combo_ness.to_json('combo_ness.json',orient='records')
    print(time.time()-start)
    nessfeatsplus = ness_feats + p_ness_feats + s_ness_feats
    return combo_ness, nessfeatsplus #combo_feats_plus



def FindTsvdTfidf(column, n_tsvd):
    tfidf = TfidfVectorizer(ngram_range=(1,1))
    tsvd = TruncatedSVD(n_components = n_tsvd, random_state = 777)
    tcols = tfidf.fit_transform(column)
    column = tsvd.fit_transform(tcols)
    return np.transpose(column)
    

# Use this to replace any of our -1's with the average (not including the -1's)
def NegOneToAvg(db, features):
    for feat in features:
        avg = np.mean(db.loc[db[feat]!=-1,feat])
        db.loc[db[feat]==-1,feat] = avg
    return db



def MakeTtFeats(n_tsvd, title = True, diff=True):
    tt_search = ['tt_search_vec' + str(i) for i in range(0,n_tsvd)]
    tt_desc = ['tt_desc_vec' + str(i) for i in range(0,n_tsvd)]
    tt_attr = ['tt_attr_vec' + str(i) for i in range(0,n_tsvd)]
    tt_smd = ['tt_smd_vec' + str(i) for i in range(0,n_tsvd)]
    tt_sma = ['tt_sma_vec' + str(i) for i in range(0,n_tsvd)]
    tt_title = ['tt_title_vec' + str(i) for i in range(0,n_tsvd)]
    ttfeats = tt_search + tt_desc + tt_attr
    if title: ttfeats += tt_title
    if diff: ttfeats += tt_sma + tt_smd
    return ttfeats, tt_search, tt_desc, tt_attr, tt_title, tt_smd, tt_sma
    
    
def MakeTsvdTfidfVectors(combo,n_tsvd, diff = False):
    combo_tt_feats = pd.DataFrame({
        "id": combo['id']
    })    
    ttfeats, tt_search, tt_desc, tt_attr, tt_title, tt_smd, tt_sma = MakeTtFeats(n_tsvd,title=True,diff=diff)
    if diff: n_len = n_tsvd*6
    else: n_len = n_tsvd*4
    for i in range(0,n_len):
        combo_tt_feats[ttfeats[i]] = -1
    #future_tt_search = FindTsvdTfidf(combo.stem_search_term[:], n_tsvd)
    future_tt_desc = FindTsvdTfidf(desc.stem_desc[:], n_tsvd)
    future_tt_attr = FindTsvdTfidf(attr_vals.stem_value[:], n_tsvd)
    # First the search terms:
    sts = np.unique(combo.stem_search_term[:]).tolist()
    future_tt_search = FindTsvdTfidf(sts, n_tsvd)
    vals = np.transpose(future_tt_search)
    st_d = dict(zip(sts,vals));
    st_tt = []
    for st in combo.stem_search_term:
        st_tt.append( list(st_d[st].tolist() ))
    combo_tt_feats.loc[:,tt_search] = st_tt
    pids = np.unique(combo.product_uid).tolist()
    keys = attr_vals.product_uid; vals = np.transpose(future_tt_attr)
    attr_d = dict(zip(keys,vals)); 
    no_attr = list(set(pids)-set(keys)); no_attr = dict(zip(no_attr,[np.array([-1]*n_tsvd)]*len(no_attr)))
    attr_d.update(no_attr)
    keys = desc.product_uid; vals = np.transpose(future_tt_desc)
    desc_d = dict(zip(keys,vals)); 
    attr_desc_tt = []
    for pid in combo.product_uid:
        attr_desc_tt.append( list(attr_d[pid].tolist() + desc_d[pid].tolist()))
    combo_tt_feats.loc[:,tt_attr + tt_desc] = attr_desc_tt
    #combo_tt_feats.to_hdf('combo_tt_feats.h5','df')
    # Add in the titles:
    titles = np.unique(combo.product_title[:]).tolist()
    future_tt_title = FindTsvdTfidf(titles, n_tsvd)
    vals = np.transpose(future_tt_title)
    title_d = dict(zip(titles,vals));
    title_tt = []
    for t in combo.product_title:
        title_tt.append( list(title_d[t].tolist() ))
    combo_tt_feats.loc[:,tt_title] = title_tt
    # Improvement is only 0.00036
    if diff:
        # Merge the search terms, product descriptions, and attributes to evaluate all 3:
        all_3_col = sts + desc.stem_desc[:].tolist() + attr_vals.stem_value[:].tolist()
        future_tt_all = FindTsvdTfidf(all_3_col, n_tsvd)
        fa_tt_search = future_tt_all[:,0:len(sts)]
        fa_tt_desc = future_tt_all[:,len(sts):len(sts)+len(desc)]
        fa_tt_attr = future_tt_all[:,len(sts)+len(desc):len(future_tt_all[0])]
        vals = np.transpose(fa_tt_search); st_d = dict(zip(sts,vals));
        st_tt = np.zeros((len(combo_tt_feats),len(fa_tt_attr)))
        for i,st in enumerate(combo.stem_search_term):
            st_tt[i,:]  = st_d[st]
        keys = attr_vals.product_uid; vals = np.transpose(fa_tt_attr)
        attr_d = dict(zip(keys,vals)); attr_d.update(no_attr)
        keys = desc.product_uid; vals = np.transpose(fa_tt_desc)
        desc_d = dict(zip(keys,vals)); 
        attr_desc_tt = np.zeros((len(combo_tt_feats),n_tsvd*2))
        for pid in combo.product_uid:
            attr_desc_tt[i,0:n_tsvd] = attr_d[pid]
            attr_desc_tt[i,n_tsvd:n_tsvd*2] = desc_d[pid]
        combo_tt_feats[tt_sma] = abs(st_tt-attr_desc_tt[:,0:n_tsvd])
        combo_tt_feats[tt_smd] = abs(st_tt-attr_desc_tt[:,n_tsvd:n_tsvd*2])
    #combo_tt_feats.to_hdf('combo_tt_feats.h5','df')
    combo_tt_feats.to_json('combo_tt_feats.json',orient='records')
    return combo_tt_feats, ttfeats



# This function converts our 30 or so *ness values to the top 10 SVD vectors.
# Unfortunately, the result only reduces the score.
def NessTSVD(combo_feats_plus,n_tsvd = 10): 
    ness_feats = GetNessFeats()
    Ness = combo_feats_plus[ness_feats][:]
    t_ness = ['t_ness' + str(i) for i in range(0,n_tsvd)]    
    tsvd = TruncatedSVD(n_components = n_tsvd, random_state = 777)
    NessTSVD = np.transpose(tsvd.fit_transform(Ness))
    for i in range(0,n_tsvd):
        combo_feats_plus[t_ness[i]] =  NessTSVD[i]
    #combo_feats_plus.drop(ness_feats)
    return combo_feats_plus, t_ness


def ExtraFeatures(combo,combo_feats):
    combo_feats['has_attr'] = 0
    combo_feats['num_prod_appear'] = 0
    combo_feats['num_search_appear'] = 0
    
    # Calculate whether or not the product has an attribute
    pids = np.unique(combo.product_uid).tolist()
    keys = attr_vals.product_uid; vals = [1]*len(keys)
    attr_d = dict(zip(keys,vals)); 
    no_attr = list(set(pids)-set(keys)); no_attr = dict(zip(no_attr,[0]*len(no_attr)))
    attr_d.update(no_attr)
    attr_list = []
    for pid in combo.product_uid:
        attr_list.append( attr_d[pid])
    combo_feats.loc[:,'has_attr'] = attr_list
    print('Has_Attr Complete')
    
    # Calculate how often the product appears when searched
    cpids = np.array(combo.product_uid)
    keys = pids; vals = [np.sum(cpids==pid) for pid in pids]
    pap_d = dict(zip(keys,vals))
    pap_list = []
    for pid in combo.product_uid:
        pap_list.append( pap_d[pid])
    combo_feats.loc[:,'num_prod_appear'] = pap_list
    print('Num_Prod Complete')
    
    # Calculate how often the original search term appears:
    osts = np.unique(combo.search_term[:]).tolist()
    c_osts = np.array(combo.search_term)
    keys = osts; vals = [np.sum(c_osts==ost) for ost in osts]
    osts_d = dict(zip(keys,vals))
    osts_list = []
    for ost in combo.search_term:
        osts_list.append( osts_d[ost])
    combo_feats.loc[:,'num_search_appear'] = osts_list
    print('Num_search complete')

    # Calculate how often the original search term appears:
    osts = np.unique(combo.search_term[:]).tolist()
    c_osts = np.array(combo.search_term)
    osts_num_d = dict(zip( osts,list(range(0,len(osts)))) )
    c_nums = np.array([osts_num_d[x] for x in c_osts])
    keys = osts; vals = [np.sum(c_nums==x) for x in range(0,len(osts))]
    osts_d = dict(zip(keys,vals))
    osts_list = []
    for ost in combo.search_term:
        osts_list.append( osts_d[ost])
    combo_feats.loc[:,'num_search_appear'] = osts_list
    
    combo['corrected_search_term'] = list(map(lambda x: ' '.join(x), combo.search_list[:]))
    # Has Units? How about the same units?
    units = '(foot|amp|gallon|inch|centimeter|meter|ounce|pound|sqft|volt|watt|quart)'
    combo['search_units'], combo_feats['num_units'] = ExtractPattern(combo.corrected_search_term[:],units)
    combo_feats['frac_units_found'] = CheckUnitsInDA(combo['search_units'][:])
    
    # Has numbers in search?
    numbers = '([0-9]+\.?[0-9]*)';
    numbers = '([0-9]+\.?[0-9]*-? ?[0-9]*\/?[0-9]*)'
    combo['search_numbers'], combo_feats['num_numbers'] = ExtractPattern(combo.corrected_search_term[:],numbers)
    combo_feats['frac_numbers_found'] = CheckUnitsInDA(combo['search_numbers'][:])
    
    combo_feats.to_json('combo_feats.json',orient='records')    
    return combo_feats
    




# For when combo_feats gets messed up, but combo is alll gooood.
def ReconstructingComboFeats(combo):
    combo_feats = BasicFeatures(combo)
    combo_feats['frac_in_attr'] = list(map(lambda x: 1-np.mean(x), combo['attr_leftovers'][:]))
    combo_feats['frac_in_desc'] = list(map(lambda x: 1-np.mean(x), combo['desc_leftovers'][:]))
    combo_feats['frac_in_title'] = list(map(lambda x: 1-np.mean(x), combo['title_leftovers'][:]))
    combo_feats['leftover_frac_td'] = list(map(lambda x,y: np.mean(np.multiply(x,y)), combo['title_leftovers'][:],combo['desc_leftovers'][:])) # Ignores attributes, customers don't see them.
    print('Doing Stems')    
    combo_feats['stem_frac_title'] = list(map(lambda x: 1-np.mean(x), combo['stem_title_leftovers'][:]))
    combo_feats['stem_frac_attr'] = list(map(lambda x: 1-np.mean(x), combo['stem_attr_leftovers'][:]))
    combo_feats['stem_frac_desc'] = list(map(lambda x: 1-np.mean(x), combo['stem_desc_leftovers'][:]))
    combo_feats['stem_leftover_frac_td'] = list(map(lambda x,y: np.mean(np.multiply(x,y)), combo['stem_title_leftovers'][:],combo['stem_desc_leftovers'][:])) # Ignores attributes, customers don't see them.
    combo_feats['stem_leftover_frac_tda'] = list(map(lambda x: np.mean(x), combo['stem_all_leftovers'][:]))
    # Removing the NaN's
    combo_feats.loc[np.isnan(combo_feats['stem_leftover_frac_td']),'stem_leftover_frac_td'] = 0
    combo_feats.loc[np.isnan(combo_feats['stem_leftover_frac_tda']),'stem_leftover_frac_tda'] = 0
    combo_feats.loc[np.isnan(combo_feats['stem_frac_attr']),'stem_frac_attr'] = 0
    combo_feats.loc[np.isnan(combo_feats['stem_frac_desc']),'stem_frac_desc'] = 0
    combo_feats.loc[np.isnan(combo_feats['stem_frac_title']),'stem_frac_title'] = 0
    print('Starting Extra Feats')
    combo_feats = ExtraFeatures(combo,combo_feats)
    return combo_feats
    


## Using all of these functions:
def CalculateFeats(combo):
    #combo = CreateComboPlus()
    combo_feats = BasicFeatures(combo)
    combo, combo_feats = FindLeftovers(combo,combo_feats)
    combo_ness, nessfeatsplus = NessFeatures(combo_feats)
    combo_tt_feats, ttfeats = MakeTsvdTfidfVectors(combo,10, diff = True)
    combo_feats = ExtraFeatures(combo,combo_feats)

    return combo, combo_feats, combo_ness, combo_tt_feats


def LeftoverSearchToNess(combo, combo_ness):
    leftovers = list(map( lambda x :[True if y==1 else False for y in x],combo['stem_all_leftovers']))
    for i,x in enumerate(combo.stem_search):
        a = []
        for j,b in enumerate(leftovers[i]):
            if b: a.append(x[j])
        leftovers[i] = a
    word_sim = pd.read_csv('Conrad//wordsim.csv',encoding="ISO-8859-1")
    word_sim['stem_word'] = list(map(lambda x:  ''.join(StopAndStem([x])) , word_sim['word']))
    ness_feats = GetNessFeats()
    keys = word_sim['stem_word']; vals = np.array(word_sim[ness_feats]);
    word_d = dict( zip(keys,vals))
    left_ness = np.zeros( (len(combo),len(ness_feats)) )
    for i,lefties in enumerate(leftovers):
        a = []
        for word in lefties:
            if word in word_d:
                a.append(word_d[word])
        if len(a)==0:
            left_ness[i,:] =-1*np.ones((1,len(ness_feats)))[0]
        elif len(a)==1:
            left_ness[i,:] = a[0]
        elif len(a)>1:
            left_ness[i,:] = np.mean(a,axis=0)
    ness_left = list(map(lambda x: 's_left_' + x, ness_feats))
    left_ness = np.transpose(left_ness)
    for i,feat in enumerate(ness_left):
        combo_ness[feat] = left_ness[i]
    return combo_ness
    
def ExtractPattern(column,pattern): # Looks for pattern like units or numbers
    count_pattern = [0]*len(column)
    res_col= [0]*len(column)
    for ind, line in enumerate(column):
        line = str(line)
        result = re.findall(pattern,line)
        if len(result)>0: count_pattern[ind] = len(result);
        else:  count_pattern[ind] = 0;
        res_col[ind] = result
    return res_col, count_pattern


def CheckUnitsInDA(column): 
    #column is the search units in same order as combo
    desc_attr = CombinedDescAttr()
    desc_attr = MakeComboLikeList(desc_attr,'desc_attr')
    count = [0]*len(column)
    for ind, units in enumerate(column):
        if len(units)==0: count[ind] = -1; continue;
        for i,unit in enumerate(units):
            if unit in desc_attr[ind]: count[ind]+=1;
        count[ind] = count[ind]/len(units)
        return count

def CombinedDescAttr(stem = False):
    attr_vals = pd.read_json('attr_vals.json')
    attr_vals = attr_vals.set_index('product_uid')
    desc = pd.read_json('desc.json')
    desc = desc.set_index('product_uid')
    deat = pd.concat( [desc,attr_vals], axis=1)
    if stem: col = list(map(lambda x,y: str(x) + ' //// ' + str(y), deat['stem_desc'][:],deat['stem_value'][:]   ))
    else: col = list(map(lambda x,y: str(x) + ' //// ' + str(y), deat['product_description'][:],deat['value'][:]   ))
    result = pd.DataFrame({
        "product_uid": deat.index.values,
        "desc_attr":col
    })    
    return result


# This should have been made earlier: converts unique dataframe with product_uid's
# to those of the combo file. This makes using pandas SO MUCH FASTER
def MakeComboLikeList(df,col,pid_name = 'product_uid'):
    pids = np.unique(combo['product_uid']).tolist()
    keys = df[pid_name]; vals = df[col];
    df_d = dict(zip(keys,vals)); 
    no_df = list(set(pids)-set(keys)); no_df = dict(zip(no_df,[0]*len(no_df)))
    df_d.update(no_df)
    df_list = []
    for pid in combo.product_uid:
        df_list.append( df_d[pid])
    return df_list


####
def IntegrateConradFeats(combo_feats):
    # Use conrad's data to compare the numbers in his search terms vs attributes
    # All numbers are converted to fractions.
    con_attr_u = pd.read_csv('Conrad//unit_Attributes.csv',encoding="ISO-8859-1")
    
    # How many dimensions in attributes for each item?
    combo_feats['num_attr_dims'] = MakeComboLikeList(con_attr_u,'num_dims','pid');    
    
    # Using the data set with corrected numbers, compare number of tokens found
    #con_st['unit_strings'] = list(map(lambda x: str(x).split(';;'),con_st['unit_strings'][:] ))
    

    con_attr_u['pid'] = [int(x) for x in con_attr_u['pid']]
    con_attr_u = con_attr_u.set_index('pid')
    con_st = pd.read_csv('Conrad//new_search_terms.csv',encoding="ISO-8859-1",usecols=['term','corrected_term','unit_strings'])
    con_st = con_st.set_index('term')

    def numMatching(s1,s2):
        l1 = set(s1.split(';;'))
        l2 = s2.split(';;')
        num = 0
        for unitphrase in l1:
            num += len([x for x in l2 if x == unitphrase])
        return num    

    def numMatchingTokens(product_uid,term):
        #units is unit_Attributes.csv, sterms is new_search_terms.csv
        s1 = con_st.at[term,'unit_strings']
        if product_uid in con_attr_u.index and not isinstance(s1,float):
            s2 = con_attr_u.at[product_uid,'unit_strings']
            if not isinstance(s2,float):
                return numMatching(s1,s2)
            else:
                return 0
        else:
            return 0  
    combo_feats['num_numberunit_in_attr'] = list(map(lambda x,y: numMatchingTokens(x,y),combo['product_uid'][:],combo['search_term'][:]))
####


# Add longest list of strings found
# Add number of times each word is found
#%% Running the code above! 
########################## RUNNING THE CODE ##############################
if FIRST_RUN:
    # Processing the data:
    print ('Starting Up')
    StartUp()
    print ('Making Feats')
    combo, combo_feats, combo_feats_plus = CalculateFeats()
    combo_tt_feats, ttfeats = MakeTsvdTfidfVectors(10, diff = True)



##################### MAKING THE SUBMISSION ###################################
### The fomula for the current top submission submission:
combo_feats_plus = pd.concat( (combo_feats,combo_ness.drop('id',axis=1),combo_tt_feats.drop('id',axis=1)),axis=1, ignore_index=False)
features = ChooseFeatures(True, True, True, ness = True, extras = True, extra_ness = False)
ttfeats = MakeTtFeats(10,False,False)[0]
features = features + ttfeats 
combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
train,test = SplitCombo(combo_feats_plus, combo)
CheckTheBest(features,train)
MakeSubmission(train, test, features)#,name = 'MehBestTry.csv')






# Try Setting Up non-attr files as the avg!
#combo_feats_plus =NessFeatures()
## When trimming feattures
## feats = CheckTheBest(features,train,k=90)
#
#
## Try without the Negative to Avg
#combo_ness = LeftoverSearchToNess(combo, combo_ness)
#combo_feats_plus = pd.concat( (combo_feats,combo_ness.drop('id',axis=1),combo_tt_feats.drop('id',axis=1)),axis=1, ignore_index=False)
#features = ChooseFeatures(True, True, True, ness = True, extras = True, extra_ness = True)
#ttfeats = MakeTtFeats(10,False,False)[0]
#features = features + ttfeats 
##combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
#train,test = SplitCombo(combo_feats_plus)
#CheckTheBest(features,train)
#MakeSubmission(train, test, features,n_tree = 100, max_depth = 9, name = 'MehAltTry.csv')
#
#
#
#
#
#
#
#
#
#
## Adding in the new features
#combo_feats_plus = pd.concat( (combo_feats,combo_ness.drop('id',axis=1),combo_tt_feats.drop('id',axis=1)),axis=1, ignore_index=False)
#features = ChooseFeatures(True, True, True, ness = True, extras = True, extra_ness = False, new = True)
#ttfeats = MakeTtFeats(10,False,False)[0]
#features = features + ttfeats 
#combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
#train,test = SplitCombo(combo_feats_plus,combo)
#CheckTheBest(features,train)
#MakeSubmission(train, test, features)
#
#
#
## Adding in Conrad's features
#combo_feats_plus = pd.concat( (combo_feats,combo_ness.drop('id',axis=1),combo_tt_feats.drop('id',axis=1)),axis=1, ignore_index=False)
#features = ChooseFeatures(True, True, True, ness = True, extras = True, extra_ness = False, new = True)
#ttfeats = MakeTtFeats(10,False,False)[0]
#features = features + ttfeats + ['num_attr_dims','num_numberunit_in_attr']
#combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
#train,test = SplitCombo(combo_feats_plus,combo)
#CheckTheBest(features,train)
#MakeSubmission(train, test, features,name='MehAltTry.csv')
#
#
#






#
## Negative removed, include conrad
#combo_feats_plus = pd.concat( (combo_feats,combo_ness.drop('id',axis=1),combo_tt_feats.drop('id',axis=1)),axis=1, ignore_index=False)
#features = ChooseFeatures(True, True, True, ness = True, extras = True, extra_ness = False, new = True)
#ttfeats = MakeTtFeats(10,False,False)[0]
#features = features + ttfeats + ['num_attr_dims','num_numberunit_in_attr']
##combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
#train,test = SplitCombo(combo_feats_plus,combo)
#CheckTheBest(features,train)
#MakeSubmission(train, test, features,name='MehAltNegTry.csv')










# Perhaps number of dimensions in attributes?

# Does the search contain words?














#
## Just throw more at it
#combo_ness = LeftoverSearchToNess(combo, combo_ness)
#combo_feats_plus = pd.concat( (combo_feats,combo_ness.drop('id',axis=1),combo_tt_feats.drop('id',axis=1)),axis=1, ignore_index=False)
#features = ChooseFeatures(True, True, True, ness = True, extras = True, extra_ness = True)
#ttfeats = MakeTtFeats(10,False,False)[0]
#features = features + ttfeats 
#combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
#train,test = SplitCombo(combo_feats_plus)
#CheckTheBest(features,train)
#MakeSubmission(train, test, features,n_tree = 500, max_depth = 11, name = 'MehBestTry.csv')


##################### LINUX BASED WORK: IGNORE FOR NOW ###################################
#
#from sknn.mlp import Layer, Regressor
#features = ChooseFeatures(True, True, True, False)
#combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
#train,test = SplitCombo(combo_feats_plus)
#nn = Regressor(
#    layers=[
#        Layer("Rectifier", units=100),
#        Layer("Linear")],
#    learning_rate=0.02,
#    n_iter=5)
#nn.fit(np.array(train[features]),np.array(train['relevance']))
#
#y_valid = nn.predict(test[features])
#
#
#
#
#from sklearn.neural_network import MLPRegressor
#alg = MLPRegressor(hidden_layer_sizes=(500,250,100), max_iter = 500)
#alg.fit(train[features],train.relevance)
#predictions = alg.predict(test[features])
#raw_pred = predictions
## Create a new dataframe with only the columns Kaggle wants from the dataset.
#predictions[predictions>3] = 3
#predictions[predictions<1] = 1
#submission = pd.DataFrame({
#        "id": test["id"],
#        "relevance": predictions
#    })
#submission.to_csv("MehNNTry.csv", index=False)
#print('Submission Ready!')


#
#
#
#
#
#
#
## Run a CrossVal and check for best model
#CrossValStuff(1,features,train)
#CrossValStuff(0,features,train)
#CrossValStuff(2,features,train)
#CrossValStuff(3,features,train)
#
#CrossValStuff(7,features,train)
#CrossValStuff(8,features,train)
#
#
### Try a random forest and a gridsearch. RESULT: Was not beter, but now a function.
### Try normalizing all of the data, normalize(combo_feats[features], norm='l1'). RESULT: Lower Score
### Turn -1's into the average. RESULT: Higher Score
#
#
#
#
#
#####################
## Current Best:
# Adding the tt vectors, no processingcombo_
#features = ChooseFeatures(False, False, True, True, extras = False)
#ttfeats = MakeTtFeats(10,False)[0]
#features = features + ttfeats 
#for i in range(0,len(ttfeats)):
#    combo_feats_plus[ttfeats[i]] = combo_tt_feats[ttfeats[i]]
#combo_feats_plus = NegOneToAvg(combo_feats_plus,features)
#train,test = SplitCombo(combo_feats_plus)
##CheckTheBest(ChooseFeatures(True, True, True, False),train)
#CheckTheBest(features,train)
#MakeSubmission(train, test, features)
    
#    

#################### Cannot get this to work
#
#from sknn.mlp import Regressor, Layer
#
#nn = Regressor(
#    layers=[
#        Layer("Rectifier", units=100),
#        Layer("Linear")],
#    learning_rate=0.02,
#    n_iter=10)
#nn.fit(np.array(train[features]),np.array(train.relevance))
#predictions = nn.predict(np.array(test[features]))
## Create a new dataframe with only the columns Kaggle wants from the dataset.
#predictions[predictions>3] = 3
#submission = pd.DataFrame({
#        "id": test["id"],
#        "relevance": predictions
#    })
#submission.to_csv("MehCurrentTry.csv", index=False)
#print('Submission Ready!')