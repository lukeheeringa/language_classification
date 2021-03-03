#### Capstone: Language Classification
Luke Heeringa

Data Science Immersive Remote (DSIR-113020)

March 4, 2021

#### Problem Statement

Can machine learning be used to identify the language being spoken in an audio clip? 

#### Executive Summary

The scope of this project is intended to cover the use of web APIs, natural language processing (NLP), and the comparison of classification models. To begin, the Pushshift reddit API was used to collect 5000 posts from the subreddits r/Coffee and r/tea. After cleaning, the remaining 4607 post titles and associated subreddit labels were split into a training set (75% of the data) and a test set (25% of the data). Multiple NLP techniques and classification models were then tested to find the overall model setup that was best equipped to accurately classify titles that it had not yet seen. 

The ultimate model chosen incorporated a CountVectorizer that transformed 1 and 2 word tokens included in a minimum of 2 post titles into numeric data by counting their occurences in each post title. Tokens were formed with WordNet lemmatization to standardize conjugations and remove irrelevant "stop words." The count vector features were then standard scaled and fed into a logistic regression model with an l1 penalty term, a C value of 0.05, and a maximum of 100 iterations. This model was able to correctly classify 92.1% of the titles it had not previously been exposed to. 

#### Key Findings

While the types of words that were most important in classifying post titles as coming from one subreddit or the other were not particularly surprising ("coffee", "grinder", or "espresso" for r/Coffee or "tea", "teapot", and "matcha" for r/tea), their inclusion was still important. In the future, a model like this could easily be deployed across other subreddits or even other social media platforms like Facebook, Twitter, or Instagram in order to identify users as clearly discussing coffee or tea. By classifying what kind of product someone is already talking about, advertisers could best target them with products specific to their interests. 

In addition, the enumeration of key terms for r/Coffee and r/tea did reveal a split between the two: 
coffee consumers are much more likely to be talking about specific brands of coffee makers or related products. Words like "aeropress", "v60", and "moccamaster" being commonplace in coffee discussion circles show coffee drinkers to be exhibiting brand loyalty and word of mouth advertising. In comparison, tea consumers have few such commonly discussed products. This discrepency shows that there could be a rich market opportunity for tea makers or even coffee product makers to expand and develop more recognizable, branded products for tea. 

Finally, digging into the prediction probabilities given by the model showed where its major weaknesses are. Despite the use of lemmatization with a pre-existing library and the removal of stop words, the processing was unable to account for certain punctuation formattings (like a "/" directly in between two words) or misspellings (like "coffe" instead of "coffee" or "chamomille" instead of "chamomile"). This indicates that there is room for improvement in the NLP techniques being used, as more robust expression parsing and word recognition could lead to even higher accuracy models in the future. 

#### Sources

- [Audio Lingua](https://www.audio-lingua.eu/?lang=en) (note: site flagged for potential risk)
- [EveryTongue](http://www.everytongue.com/)
- [Omniglot](https://omniglot.com/soundfiles/)
- [VoxForge](http://www.voxforge.org/home/downloads)
- [Common Voice](https://commonvoice.mozilla.org/en/datasets)


#### File Directory
- README.md

- presentation_slides.pdf : a pdf version of the Google Slides presentation given on January 22, 2021


- code 
    - 01_data_collection.ipynb : process and functions used for the collection of data via the Pushshift reddit API
    - 02_cleaning.ipynb : code transforming the raw data into useable form
    - 03_EDA.ipynb : exploratory data analysis, including several visualizations of the data
    - 04_modeling.ipynb : walkthrough of the modeling techniques attempted, including various vectorization and classification models 
    - 05_findings.ipynb : a deeper dive into the final selected model's features and classification limitations


- data
    - cleaned_posts.csv : the cleaned dataset of reddit posts
    - posts.csv : the data as pulled from the Pushshift reddit API


- grid_search
    - logisticregression.csv : saved results of hyperparameter searching on a logisitic regression model
    - naivebayes.csv : saved results of hyperparameter searching on a naive bayes model
    - randomforest.csv : saved results of hyperparameter searching on a random forest model
    - svc.csv : saved results of hyperparameter searching on a support vector classification model
    - vectorizers.csv : saved results of vectorizer and hyperparameter searching (using a naive bayes classifier)
