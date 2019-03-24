
# coding: utf-8

# # Predict tags on StackOverflow with linear models

# In this assignment you will learn how to predict tags for posts from [StackOverflow](https://stackoverflow.com). To solve this task you will use multilabel classification approach.
# 
# ### Libraries
# 
# In this task you will need the following libraries:
# - [Numpy](http://www.numpy.org) — a package for scientific computing.
# - [Pandas](https://pandas.pydata.org) — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
# - [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
# - [NLTK](http://www.nltk.org) — a platform to work with natural language.

# ### Data
# 
# The following cell will download all data required for this assignment into the folder `week1/data`.

# In[212]:


import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()


# ### Grading
# We will create a grader instance below and use it to collect your answers. Note that these outputs will be stored locally inside grader and will be uploaded to platform only after running submitting function in the last part of this assignment. If you want to make partial submission, you can run that cell any time you want.

# In[213]:


from grader import Grader


# In[214]:


grader = Grader()


# ### Text preprocessing

# For this and most of the following assignments you will need to use a list of stop words. It can be downloaded from *nltk*:

# In[215]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In this task you will deal with a dataset of post titles from StackOverflow. You are provided a split to 3 sets: *train*, *validation* and *test*. All corpora (except for *test*) contain titles of the posts and corresponding tags (100 tags are available). The *test* set is provided for Coursera's grading and doesn't contain answers. Upload the corpora using *pandas* and look at the data:

# In[216]:


from ast import literal_eval
import pandas as pd
import numpy as np


# In[217]:


def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


# In[218]:


train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')


# In[219]:


train.head()


# As you can see, *title* column contains titles of the posts and *tags* column contains the tags. It could be noticed that a number of tags for a post is not fixed and could be as many as necessary.

# For a more comfortable usage, initialize *X_train*, *X_val*, *X_test*, *y_train*, *y_val*.

# In[220]:


X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values


# One of the most known difficulties when working with natural data is that it's unstructured. For example, if you use it "as is" and extract tokens just by splitting the titles by whitespaces, you will see that there are many "weird" tokens like *3.5?*, *"Flip*, etc. To prevent the problems, it's usually useful to prepare the data somehow. In this task you'll write a function, which will be also used in the other assignments. 
# 
# **Task 1 (TextPrepare).** Implement the function *text_prepare* following the instructions. After that, run the function *test_test_prepare* to test it on tiny cases and submit it to Coursera.

# In[221]:


import re


# In[222]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, "", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " " + text + " "  # delete stopwords from text
    for sw in STOPWORDS:
        text = text.replace(" "+sw+" ", " ") # delete stopwors from text
    # print(text)
    text = re.sub('[ ][ ]+', " ", text)
    if text[0] == ' ':
        text = text[1:]
    if text[-1] == ' ':
        text = text[:-1]
        
    # text = text[1:-1]
    # print(text)    
    return text


# In[223]:


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# In[224]:


print(test_text_prepare())


# 
# 
# Run your implementation for questions from file *text_prepare_tests.tsv* to earn the points.

# In[225]:


prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

grader.submit_tag('TextPrepare', text_prepare_results)


# Now we can preprocess the titles using function *text_prepare* and  making sure that the headers don't have bad symbols:

# In[226]:


X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]


# In[227]:


X_train[:3]


# For each tag and for each word calculate how many times they occur in the train corpus. 
# 
# **Task 2 (WordsTagsCount).** Find 3 most popular tags and 3 most popular words in the train data and submit the results to earn the points.

# In[228]:


# Dictionary of all tags from train corpus with their counts.
from collections import Counter
tags_counts = Counter() #{}
# Dictionary of all words from train corpus with their counts.
words_counts =  Counter() #{}
for sentence in X_train:
    for word in sentence.split():
        # print(word)
        words_counts[word] += 1

for l in y_train:
    for tag in l:
        tags_counts[tag] += 1

######################################
######### YOUR CODE HERE #############
######################################


# We are assuming that *tags_counts* and *words_counts* are dictionaries like `{'some_word_or_tag': frequency}`. After applying the sorting procedure, results will be look like this: `[('most_popular_word_or_tag', frequency), ('less_popular_word_or_tag', frequency), ...]`. The grader gets the results in the following format (two comma-separated strings with line break):
# 
#     tag1,tag2,tag3
#     word1,word2,word3
# 
# Pay attention that in this assignment you should not submit frequencies or some additional information.

# In[229]:


most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags), 
                                                ','.join(word for word, _ in most_common_words)))


# ### Transforming text to a vector
# 
# Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is". There are many ways to transform text data to numeric vectors. In this task you will try to use two of them.
# 
# #### Bag of words
# 
# One of the well-known approaches is a *bag-of-words* representation. To create this transformation, follow the steps:
# 1. Find *N* most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
# 2. For each title in the corpora create a zero vector with the dimension equals to *N*.
# 3. For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.
# 
# Let's try to do it for a toy example. Imagine that we have *N* = 4 and the list of the most popular words is 
# 
#     ['hi', 'you', 'me', 'are']
# 
# Then we need to numerate them, for example, like this: 
# 
#     {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
# 
# And we have the text, which we want to transform to the vector:
# 
#     'hi how are you'
# 
# For this text we create a corresponding zero vector 
# 
#     [0, 0, 0, 0]
#     
# And iterate over all words, and if the word is in the dictionary, we increase the value of the corresponding position in the vector:
# 
#     'hi':  [1, 0, 0, 0]
#     'how': [1, 0, 0, 0] # word 'how' is not in our dictionary
#     'are': [1, 0, 0, 1]
#     'you': [1, 1, 0, 1]
# 
# The resulting vector will be 
# 
#     [1, 1, 0, 1]
#    
# Implement the described encoding in the function *my_bag_of_words* with the size of the dictionary equals to 5000. To find the most common words use train data. You can test your code using the function *test_my_bag_of_words*.

# In[230]:


most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:6000]
DICT_SIZE = 5000
WORDS_TO_INDEX = {p[0]:i for i,p in enumerate(most_common_words[:DICT_SIZE])} ####### YOUR CODE HERE #######
INDEX_TO_WORDS =  {WORDS_TO_INDEX[k]:k for k in WORDS_TO_INDEX} ####### YOUR CODE HERE #######
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    for word in text.split():
        if word in words_to_index:
            result_vector[words_to_index[word]] += 1
    return result_vector


# In[231]:


def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# In[232]:


print(test_my_bag_of_words())


# Now apply the implemented function to all samples (this might take up to a minute):

# In[236]:


from scipy import sparse as sp_sparse


# In[237]:


X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)


# As you might notice, we transform the data to sparse representation, to store the useful information efficiently. There are many [types](https://docs.scipy.org/doc/scipy/reference/sparse.html) of such representations, however sklearn algorithms can work only with [csr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix) matrix, so we will use this one.

# **Task 3 (BagOfWords).** For the 11th row in *X_train_mybag* find how many non-zero elements it has. In this task the answer (variable *non_zero_elements_count*) should be a number, e.g. 20.

# In[238]:


row = X_train_mybag[10].toarray()[0]
non_zero_elements_count = np.count_nonzero(row)####### YOUR CODE HERE #######

grader.submit_tag('BagOfWords', str(non_zero_elements_count))


# #### TF-IDF
# 
# The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora. It helps to penalize too frequent words and provide better features space. 
# 
# Implement function *tfidf_features* using class [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from *scikit-learn*. Use *train* corpus to train a vectorizer. Don't forget to take a look into the arguments that you can pass to it. We suggest that you filter out too rare words (occur less than in 5 titles) and too frequent words (occur more than in 90% of the titles). Also, use bigrams along with unigrams in your vocabulary. 

# In[239]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[240]:


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', min_df=5, max_df=0.9, ngram_range=(1,2))   ############# YOUR CODE HERE #######
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


# Once you have done text preprocessing, always have a look at the results. Be very careful at this step, because the performance of future models will drastically depend on it. 
# 
# In this case, check whether you have c++ or c# in your vocabulary, as they are obviously important tokens in our tags prediction task:

# In[241]:


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}


# In[242]:


######### YOUR CODE HERE #############
print('c++' in tfidf_vocab)
print('c#' in tfidf_vocab)
print('java' in tfidf_vocab)
# print(tfidf_vocab)


# If you can't find it, we need to understand how did it happen that we lost them? It happened during the built-in tokenization of TfidfVectorizer. Luckily, we can influence on this process. Get back to the function above and use '(\S+)' regexp as a *token_pattern* in the constructor of the vectorizer.  

# Now, use this transormation for the data and check again.

# In[243]:


######### YOUR CODE HERE #############


# ### MultiLabel classifier
# 
# As we have noticed before, in this task each example can have multiple tags. To deal with such kind of prediction, we need to transform labels in a binary form and the prediction will be a mask of 0s and 1s. For this purpose it is convenient to use [MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) from *sklearn*.

# In[244]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[245]:


mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)


# Implement the function *train_classifier* for training a classifier. In this task we suggest to use One-vs-Rest approach, which is implemented in [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) class. In this approach *k* classifiers (= number of tags) are trained. As a basic classifier, use [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). It is one of the simplest methods, but often it performs good enough in text classification tasks. It might take some time, because a number of classifiers to train is large.

# In[246]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier


# In[247]:


def train_classifier(X_train, y_train, C=1.0, penalty='l2'):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    lr = LogisticRegression(C=C, penalty=penalty)
    # lr.fit(X_train, y_train)
    ovr = OneVsRestClassifier(lr)
    ovr.fit(X_train, y_train)
    return ovr
    ######################################
    ######### YOUR CODE HERE #############
    ######################################    


# Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.

# In[248]:


classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)


# Now you can create predictions for the data. You will need two types of predictions: labels and scores.

# In[249]:


y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


# Now take a look at how classifier, which uses TF-IDF, works for a few examples:

# In[250]:


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))


# Now, we would need to compare the results of different predictions, e.g. to see whether TF-IDF transformation helps or to try different regularization techniques in logistic regression. For all these experiments, we need to setup evaluation procedure. 

# ### Evaluation
# 
# To evaluate the results we will use several classification metrics:
#  - [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
#  - [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
#  - [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
#  - [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) 
#  
# Make sure you are familiar with all of them. How would you expect the things work for the multi-label scenario? Read about micro/macro/weighted averaging following the sklearn links provided above.

# In[251]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


# Implement the function *print_evaluation_scores* which calculates and prints to stdout:
#  - *accuracy*
#  - *F1-score macro/micro/weighted*
#  - *Precision macro/micro/weighted*

# In[252]:


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    print(#accuracy,
          #f1_score(y_val, predicted, average='macro'),
          #f1_score(y_val, predicted, average='micro'),
          f1_score(y_val, predicted, average='weighted')#,
          #average_precision_score(y_val, predicted, average='macro'),
          #average_precision_score(y_val, predicted, average='micro'),
          #average_precision_score(y_val, predicted, average='weighted')
         )
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################


# In[253]:


print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


# You might also want to plot some generalization of the [ROC curve](http://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc) for the case of multi-label classification. Provided function *roc_auc* can make it for you. The input parameters of this function are:
#  - true labels
#  - decision functions scores
#  - number of classes

# In[254]:


from metrics import roc_auc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[255]:


n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)


# In[256]:


n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)


# **Task 4 (MultilabelClassification).** Once we have the evaluation set up, we suggest that you experiment a bit with training your classifiers. We will use *F1-score weighted* as an evaluation metric. Our recommendation:
# - compare the quality of the bag-of-words and TF-IDF approaches and chose one of them.
# - for the chosen one, try *L1* and *L2*-regularization techniques in Logistic Regression with different coefficients (e.g. C equal to 0.1, 1, 10, 100).
# 
# You also could try other improvements of the preprocessing / model, if you want. 

# In[257]:


def evaluate(C, penalty):
    classifier_mybag = train_classifier(X_train_mybag, y_train, C, penalty)
    classifier_tfidf = train_classifier(X_train_tfidf, y_train, C, penalty)
    y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
    y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

    y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
    y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
    print('Bag-of-words')
    print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
    print('Tfidf')
    print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
######################################
######### YOUR CODE HERE #############
######################################


# When you are happy with the quality, create predictions for *test* set, which you will submit to Coursera.

# In[258]:


classifier_tfidf = train_classifier(X_train_tfidf, y_train, C=3.0, penalty='l1')
y_test_predicted_labels_tfidf = classifier_tfidf.predict(X_test_tfidf)
test_predictions =y_test_predicted_labels_tfidf ######### YOUR CODE HERE #############
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
grader.submit_tag('MultilabelClassification', test_predictions_for_submission)


# ### Analysis of the most important features

# Finally, it is usually a good idea to look at the features (words or n-grams) that are used with the largest weigths in your logistic regression model.

# Implement the function *print_words_for_tag* to find them. Get back to sklearn documentation on [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) and [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) if needed.

# In[259]:


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))
    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator. 
    idx = tags_classes.index(tag)
    coef=classifier_tfidf.coef_[idx]
    cd = {i:coef[i] for i in range(len(coef))}
    scd=sorted(cd.items(), key=lambda x: x[1], reverse=True)
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    top_positive_words = [index_to_words[k[0]] for k in scd[:5]]# top-5 words sorted by the coefficiens.
    top_negative_words = [index_to_words[k[0]] for k in scd[-5:]] # bottom-5 words  sorted by the coefficients.
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))


# In[260]:


print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)


# ### Authorization & Submission
# To submit assignment parts to Cousera platform, please, enter your e-mail and token into variables below. You can generate token on this programming assignment page. <b>Note:</b> Token expires 30 minutes after generation.

# In[261]:


grader.status()


# In[262]:


STUDENT_EMAIL = "mishraabhi433@gmail.com" 
STUDENT_TOKEN = "EdUTMLqbANPuaXAf"
grader.status()


# If you want to submit these answers, run cell below

# In[263]:


grader.submit(STUDENT_EMAIL, STUDENT_TOKEN)

