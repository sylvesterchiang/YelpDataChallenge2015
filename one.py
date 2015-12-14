# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 11:57:21 2015

Exerpts taken from Kaggle NLP introduction
"""

#Import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
import re

#Function to remove stopwords 
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove non-letters      
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   
    
#Returns the accuracy of the classifier
def evaluate( prediction, truth ):
    print np.sum(prediction == truth)
    return np.sum(prediction == truth) / float(len(prediction))
    
os.chdir('C:\Users\schiang\Documents\yelp')
reviews = pd.read_csv('yelp_academic_dataset_review.csv', header=0, delimiter=',')
businesses = pd.read_csv('yelp_academic_dataset_business.csv', header=0, delimiter=',')

reviews = reviews.head(20000)

num_reviews, _ = reviews.shape
num_businesses, _ = businesses.shape

print 'cleaning and parsing data\n'

clean_reviews = []
stars = []
for i in range(num_reviews):
    #if index is divisible by 1000 print, for logging purposes
    if ((i+1)%1000 == 0):
        print "review %d of %d\n" % (i+1, num_reviews)
    if (type(reviews['text'][i]) == str):
        clean_reviews.append(review_to_words(reviews['text'][i]))
        stars.append(reviews['stars'][i])
    
print "Creating the bag of words...\n"
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 500) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

#==============================================================================
# print 'saving bag of words features'
# write_data = pd.DataFrame(train_data_features)
# write_data.to_csv('bow_features.csv')
#==============================================================================

#==============================================================================
# print "Training the random forest..."
# 
# forest = RandomForestClassifier(n_estimators = 100)
# forest = forest.fit(train_data_features, stars) 
#==============================================================================

#==============================================================================
# features = pd.read_csv('bow_features.csv')
# print features.shape
#==============================================================================

from sklearn.cross_validation import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(train_data_features, stars, test_size = 0.2, random_state = 0)

print 'Training random forest'

forest = RandomForestClassifier(n_estimators = 1000)
forest.fit(X_train, y_train)

result = forest.predict(X_test)

#answer = pd.DataFrame(data = np.column_stack((result, y_test)), columns = ['prediction', 'y'])
#answer.to_csv('prediction.csv')

print evaluate(result, y_test)



