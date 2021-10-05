import collections
import matplotlib.pyplot as plt
import statistics
import warnings
import tensorflow as tf
#import arg as arg
import nltk as nl
import numpy as np
import string
import re
from keras.layers.merge import concatenate
from sklearn import preprocessing
from gensim.models import word2vec
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import LSTM, Embedding, SpatialDropout1D, Dense
from tensorflow.python.keras.tests.model_architectures import lstm
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from wordcloud import WordCloud
import nltk
import textblob
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from setuptools.command.test import test
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji as emoji
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import multilabel_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, \
    roc_auc_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from nltk.corpus import words
import validators
import gensim
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from itertools import islice
#nltk.download()
from keras.preprocessing import sequence

#test

cachedStopWords = stopwords.words('english')
dictionary = dict.fromkeys(words.words(), None)

def check_valid_word(word):
    if word not in cachedStopWords:
        '''try:
            x = dictionary[word]
            return True
        except KeyError:
            return False'''
        return True

    else:
        return False
def is_english(word):
    try:
        x = dictionary[word]
        return True
    except KeyError:
        return False

def textBlob_sentiment_score(x):
    return textblob.TextBlob(x).polarity

def vader_sentiment_score(tweet):
    return SentimentIntensityAnalyzer().polarity_scores(tweet)['compound']

def convert_emoticons(sentence):
    sentence = sentence.replace(":'(", " crying ")
    sentence = sentence.replace("(y)", " thumbs up ")
    sentence = sentence.replace(":x", " kiss ")
    sentence = sentence.replace(":3", " goofy ")
    sentence = sentence.replace(":)", " happy ")
    sentence = sentence.replace(":(", " sad ")
    return sentence
def remove_valid_links(tweet):
    links = re.findall(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', tweet)
    for link in links:
        if validators.url(link):
            tweet = re.sub(link, ' ', tweet)
    return tweet

def sentence_preprocessor(sentence):

    sentence = sentence.lower()
    sentence = convert_emoticons(sentence)
    sentence = emoji.demojize(sentence)
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = remove_valid_links(sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub('[0-9]+', '', sentence)
    new_sentence = ""
    words = sentence.split()
    for word in words:
        word =  PorterStemmer().stem(word)

        if (check_valid_word(word)):
            new_sentence += word + " "

    new_sentence = new_sentence.strip()
    new_sentence = remove_uninformative_words(new_sentence)
    return new_sentence




def perform_preprocessing(vaccine_data):
    vaccine_data["text"] = vaccine_data["text"].apply(lambda s: sentence_preprocessor(s) if type(s) == str else s)
    #print(vaccine_data["text"])
    return vaccine_data

def remove_list_of_words(list, sentence):
    for word in list:
        sentence = sentence.replace(word,"")
    return sentence



def remove_uninformative_words(sentence):
    #sentence = remove_list_of_words([" antivaxx ", " vaccin ", " vaccineswork ", " antivax ", " thi ", " covid "," peopl ", " get ", " amp ",
                                    # " ha ", " wa ", " make ", " go ", " coronaviru ", " think ", " whi ", " like ", " need ", " think ", " one ", " us ", " say ", " ita ", " take ", " right " ],sentence)
    sentence = remove_list_of_words(["antivax","antivaxx","covid", "vaccin","vaccineswork","vaccinesarenottheansw","coronavirus",
                                     "just","get","via","much","like","now","one","use","also","tell","lot","look","live","will","see",
                                     "see","dont","tri","say","amp","theyr","cant","even","anti","take","can","want","shes","well","thing",
                                     "group","come","actual","still","actual","give","care","way","show","new","put","read","make","call",
                                     "peopl","fufuu","isnt","feel","yall","didnt","got","need","test","know","immunizedotca","time","year",
                                     "right","mani","big","sure","claim","back","healthscience","talk","system","antivaccine","hes","day","date",
                                     "let","bet","guy","far","sinc","page","link","guess","mom","forc","hey","dog","condit",
                                     "provid","thegoodgodabove","marcusblimi","immunizedotca",
                                    "elonjam","couldnt","gullibi","fransrech","gonna",
                                    "cameronwilson","forc","said","watch","away",
                                    "therickwilson","kathmarv","franldelia",
                                    "chrisjohnsonmd","mamadeb","ipaworldorg","click","yet",
                                    "doesnt","whole","ever","find","near","yes","mcfunni",
                                    "must","full","ear","without","ask","due","agr","pjmoor",
                                    "macbarid","jestrbob","yeah","krebiozen",
                                    "vaccinesarenottheans","vaccineagenda","found","home",
                                    "frankdelia","melindafirst","https","tco","vaxxer","fufuufufuufufuu",
                                    "kevinroos","zabotxdyjz","kbghyyh","clccalala","ajpollard","vaxx",
                                    "fltvqjqkmm","fltvqjnm","tytvftwq","asovtgqf","soezdwsc","vnouhhpyl",
                                    "kmolgehqu","sjhjifx","vnouhhpyl","tuesdaythough","psguqvu",
                                    "themelyssak","karinagould","other","wa","ha","think","trump"], sentence)
    return sentence


def get_text_from_data(tweets):
    return ' '.join(tweets['text'])

def create_wordcloud(preprocessed_data):
    pro_tweets = preprocessed_data.loc[preprocessed_data['Label'] == 0]
    anti_tweets = preprocessed_data.loc[preprocessed_data['Label'] == 1]

    pro_wordcloud = WordCloud().generate(get_text_from_data(pro_tweets))
    anti_wordcloud = WordCloud().generate(get_text_from_data(anti_tweets))
    overall_wordcloud = WordCloud().generate(get_text_from_data(preprocessed_data))

    plt.imshow(overall_wordcloud)
    plt.title("Wordcloud for All Classes")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(pro_wordcloud)
    plt.title("Positive Class Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    plt.imshow(anti_wordcloud)
    plt.title("Negative Class Wordcloud")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()



def get_top_ten_words(preprocessed_data, labels):
    preprocessed_unique_words = []
    pro_words = []
    anti_words = []
    for index, row in preprocessed_data.iterrows():
        sentence = row["text"]
        labels = row["Label"]
        words_in_sentence = sentence.split()
        for word in words_in_sentence:
            preprocessed_unique_words.append(word)
            if labels == 0:
                pro_words.append(word)
            elif labels == 1:
                anti_words.append(word)

    pro_count = collections.Counter(pro_words)
    anti_count = collections.Counter(anti_words)
    '''print("Pro top 10: ", pro_count.most_common(10))
    print("Anti top 10: ", anti_count.most_common(10))'''

def social_features(vaccine_data):
    display_text_width = []
    is_quote_list = []
    is_tweet_list = []
    followers_count_list = []
    friends_count_list = []
    listed_count_list = []
    statuses_count_list = []
    favourites_count_list = []
    verifieds_list = []
    quoted_favorite_count_list = []
    quoted_retweet_count_list = []
    quoted_followers_count_list = []
    quoted_friends_count_list = []
    quoted_statuses_count_list = []
    retweet_count_list = []
    Y = vaccine_data.loc[:, 'Label']
    for index, row in vaccine_data.iterrows():
        text_width = row['display_text_width']
        display_text_width.append(text_width)
        followers_count = row['followers_count']
        followers_count_list.append(followers_count)
        friends_count = row['friends_count']
        friends_count_list.append(friends_count)
        listed_count = row['listed_count']
        listed_count_list.append(listed_count)
        statuses_count = row['statuses_count']
        statuses_count_list.append(statuses_count)
        favourites_count = row['favourites_count']
        favourites_count_list.append(favourites_count)
        retweet_count = row['retweet_count']
        retweet_count_list.append(retweet_count)
        quoted_favorite_count = row['quoted_favorite_count']
        quoted_favorite_count_list.append(quoted_favorite_count)
        quoted_retweet_count = row['quoted_retweet_count']
        quoted_retweet_count_list.append(quoted_retweet_count)
        quoted_followers_count = row['quoted_followers_count']
        quoted_followers_count_list.append(quoted_followers_count)
        quoted_friends_count = row['quoted_friends_count']
        quoted_friends_count_list.append(quoted_friends_count)
        quoted_statuses_count = row['quoted_statuses_count']

        quoted_statuses_count_list.append(quoted_statuses_count)


    social_feature_data = {#'display_text_width': display_text_width,
            'retweet_count': retweet_count_list,
            'followers_count_list':followers_count_list,
                   #'listed_count': listed_count_list,
                   'friends_count_list': friends_count_list,
                    'statuses_count_list':statuses_count_list,
                   'favourites_count_list': favourites_count_list,

                    #'quoted_favorite_count_list': quoted_favorite_count_list,
                    'quoted_retweet_count_list': quoted_retweet_count_list,
                    'quoted_followers_count': quoted_followers_count_list,
                    'quoted_friends_count': quoted_friends_count_list,
                    'quoted_statuses_count': quoted_statuses_count_list

    }
    Y = Y.astype('int')
    #print(Y)
    df1 = pd.DataFrame(social_feature_data)
    #df1 = df1.fillna(0)
    #print(df1)
    # apply SelectKBest class to extract top 10 best features
    #print(df1)
    x = df1.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df1 = pd.DataFrame(x_scaled)
    print(".......minmax scaler..")
    print(df1)
    #print(df1)
    #saving the dataframe


    return df1


def get_metadata(vaccine_data):
    number_of_questionmark = []
    number_of_exclamationmark = []
    number_of_periodmark = []
    number_of_quotationmark = []
    number_of_quotationmark1 = []
    hashtag_counts = []
    mention_counts = []
    punctuation_counts = []
    link_counts = []
    capital_word_counts = []
    vader_scores = []
    textblob_scores = []
    Y = vaccine_data.loc[:, 'Label']

    for index, row in vaccine_data.iterrows():
        tweet = row['text']
        links = re.findall(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', tweet)
        link_counts.append(len(links))

        count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        punctuation_count = count(tweet, string.punctuation)
        punctuation_counts.append(punctuation_count)

        question_count = count(tweet,"?")
        number_of_questionmark.append(question_count)

        exclamation_count = count(tweet, "!")
        number_of_exclamationmark.append(exclamation_count)

        period_count = count(tweet, "?")
        number_of_periodmark.append(period_count)

        quotation_count = count(tweet, "\'")
        number_of_quotationmark.append(quotation_count)

        quotation_count1 = count(tweet, "\"")
        number_of_quotationmark1.append(quotation_count1)

        textBlob_score = textBlob_sentiment_score(tweet)
        textblob_scores.append(textBlob_score)

        vader_score = vader_sentiment_score(tweet)
        vader_scores.append(vader_score)


        words = tweet.split()
        hashtag_count = 0
        mention_count = 0
        capital_word = 0
        for word in words:
            if word.startswith('#'):
                hashtag_count += 1
            elif word.startswith('@'):
                mention_count += 1
            elif word.isupper():
                capital_word += 1


        hashtag_counts.append(hashtag_count)
        mention_counts.append(mention_count)
        capital_word_counts.append(capital_word)

    data = {'hashtag_counts': hashtag_counts,
            'mention_counts': mention_counts,
            'punctuation_counts': punctuation_counts,
            'link_counts': link_counts,
            'capital_word_counts': capital_word_counts,
            'number_of_questionmark': number_of_questionmark,
            'number_of_exclamationmark': number_of_exclamationmark,
            'number_of_periodmark' : number_of_periodmark,
            #'number_of_quotationmark' : number_of_quotationmark,
            #'number_of_quotationmark1' : number_of_quotationmark1,
            #'vader_scores': vader_scores,
            #'textblob_score': textblob_scores
            }
    df = pd.DataFrame(data)
    return df, hashtag_counts

def tf_idf_vectorize(tweet_dataframe):
    corpus = vaccine_data.loc[:, "text"]
    Y = vaccine_data.loc[:, "Label"]
    unigram_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=2000)
    unigram_tf_idf = unigram_vectorizer.fit_transform(corpus)
    return unigram_tf_idf, Y



def word_embedding_2(preprocessed_data):
    train = []
    # getting only the first 4 columns of the file
    for index, row in preprocessed_data.iterrows():
        sentence = row["text"]
        train.append(sentence)

    # Create an array of tokens using nltk
    tokens = [nl.word_tokenize(sentences) for sentences in train]

    model = gensim.models.Word2Vec(tokens, size=10, min_count=1, workers=4)
    print("\n Training the word2vec model...\n")
    # reducing the epochs will decrease the computation time
    model.train(tokens, total_examples=len(tokens), epochs=40)
    first_value_means = []
    second_value_means = []
    third_value_means = []
    fourth_value_means = []
    fifth_value_means = []
    sixth_value_means = []
    seventh_value_means = []
    eights_value_means = []
    ninth_value_means = []
    tenth_value_means = []


    for index, row in preprocessed_data.iterrows():
        sentence = row["text"]
        word_array = []
        first_value = []
        second_value = []
        third_value = []
        fourth_value = []
        fifth_value = []
        sixth_value = []
        seventh_value = []
        eights_value = []
        ninth_value = []
        tenth_value = []
        for word in nl.word_tokenize(sentence):
            word_2_vec_value = model[word]
            #print(word_2_vec_value)
            first_value.append(word_2_vec_value[0])
            second_value.append(word_2_vec_value[1])
            third_value.append(word_2_vec_value[2])
            fourth_value.append(word_2_vec_value[3])
            fifth_value.append(word_2_vec_value[4])
            sixth_value.append(word_2_vec_value[5])
            seventh_value.append(word_2_vec_value[6])
            eights_value.append(word_2_vec_value[7])
            ninth_value.append(word_2_vec_value[8])
            tenth_value.append(word_2_vec_value[9])
        # word_array = [statistics.mean(first_values), statistics.mean(second_values)]
        if len(first_value) > 0:
            first_value_means.append(statistics.mean(first_value))
        else:
            first_value_means.append(0.0)
        if len(second_value) > 0:
            second_value_means.append(statistics.mean(second_value))
        else:
            second_value_means.append(0.0)
        if len(third_value) > 0:
            third_value_means.append(statistics.mean(third_value))
        else:
            third_value_means.append(0.0)
        if len(fourth_value) > 0:
            fourth_value_means.append(statistics.mean(fourth_value))
        else:
            fourth_value_means.append(0.0)
        if len(fifth_value) > 0:
            fifth_value_means.append(statistics.mean(fifth_value))
        else:
            fifth_value_means.append(0.0)
        if len(sixth_value) > 0:
            sixth_value_means.append(statistics.mean(sixth_value))
        else:
            sixth_value_means.append(0.0)
        if len(seventh_value) > 0:
            seventh_value_means.append(statistics.mean(seventh_value))
        else:
            seventh_value_means.append(0.0)
        if len(eights_value) > 0:
            eights_value_means.append(statistics.mean(eights_value))
        else:
            eights_value_means.append(0.0)
        if len(ninth_value) > 0:
            ninth_value_means.append(statistics.mean(ninth_value))
        else:
            ninth_value_means.append(0.0)
        if len(tenth_value) > 0:
            tenth_value_means.append(statistics.mean(tenth_value))
        else:
            tenth_value_means.append(0.0)


    data = {'first_value_wordembedding': first_value_means,
            'second_value_wordembedding': second_value_means,
            'third_value_wordembedding': third_value_means,
            'fourth_value_wordembedding': fourth_value_means,
            'fifth_value_wordembedding': fifth_value_means,
            'sixth_value_wordembedding': sixth_value_means,
            'seventh_value_wordembedding': seventh_value_means,
            'eighth_value_wordembedding': eights_value_means,
            'ninth_value_wordembedding': ninth_value_means,
            'tenth_value_wordembedding': tenth_value_means

            }
    #print(len(data))
    df = pd.DataFrame(data)
    #print(df)
    return df



def get_distribution_of_unique_words(vaccine_data, labels):
    pro_tweets = vaccine_data.loc[vaccine_data['Label'] == 0]
    anti_tweets = vaccine_data.loc[vaccine_data['Label'] == 1]
    preprocessed_unique_words = set()
    pro_unique_words = set()
    anti_unique_words = set()

    total_words_counts = [0, 0, 0]

    for index, row in vaccine_data.iterrows():

        sentence = row["text"]
        label = row["Label"]
        words_in_sentence = sentence.split()
        number_of_words = len(words_in_sentence)
        total_words_counts[0] += number_of_words
        if label == 0:
            total_words_counts[1] += number_of_words
        elif label == 1:
            total_words_counts[2] += number_of_words
        for word in words_in_sentence:
            if label == 0 and not (word in pro_unique_words):
                pro_unique_words.add(word)

            elif label == 1 and not (word in anti_unique_words):
                anti_unique_words.add(word)

            preprocessed_unique_words.add(word)

    print("Total number of words in " + labels[0] + " class", (total_words_counts[1]))
    print("Total number words in " + labels[1] + " class", (total_words_counts[2]))
    print("Total number of total words: ", (total_words_counts[0]))

    print("Average number of words in " + labels[0] + " class", (total_words_counts[1]) / len(pro_tweets))
    print("Average number of  words: in " + labels[1] + " class", (total_words_counts[2]) / len(anti_tweets))
    print("Average number of total words: ", (total_words_counts[0]) / len(vaccine_data))

    print("Total unique words after preprocessing: ", len(preprocessed_unique_words))
    print("Total unique words in " + labels[0] + " class: ", len(pro_unique_words))
    print("Total unique words in " + labels[1] + " class: ", len(anti_unique_words))
    #print(preprocessed_unique_words)'''

def most_common_hashtag(vaccine_data):
    pro_hashtag = []
    anti_hashtag = []

    for index, row in vaccine_data.iterrows():
        sentence = str(row["hashtags"])
        hashtags = sentence.split()
        sentiment = row["Label"]
        for word in hashtags:
            if sentiment == 0 and word != "nan":
                pro_hashtag.append(word)
            elif sentiment == 1 and word != "nan":
                anti_hashtag.append(word)
    pro_counter = collections.Counter(pro_hashtag)
    anti_counter = collections.Counter(anti_hashtag)
    print("Top 10 most common pro_vaccine hashtags: ", pro_counter.most_common(10))
    print("Top 10 most common anti_vaccine hashtags: ", anti_counter.most_common(10))

def get_vaccine_data():
    table = pd.read_csv("vaccinesMay20labeled.csv", sep=',', engine='python')
    sentiment_mapper = {"Label": {"P": 0, "A": 1}}
    labels = ['Pro-vaccine', 'Anti-vaccine']
    table.replace(sentiment_mapper, inplace=True)
    pro_vaccine = table.loc[table['Label'] == 0]
    anti_vaccine = table.loc[table['Label'] == 1]
    #print(len(pro_vaccine))
    #print(len(anti_vaccine))
    label_tweets = pd.concat([pro_vaccine, anti_vaccine])
    return label_tweets, labels


def run_supervised_ML(all_data, Y, labels):
    Y = Y.astype('int')


    X_train, X_test, y_train, y_test = train_test_split(all_data, Y,
                                                        stratify=Y,
                                                        test_size=0.25)

    '''X_train, X_test, y_train, y_test = train_test_split(
        all_data, Y, test_size=0.2, random_state=42)'''


    print("MLP")
    warnings.filterwarnings('ignore')
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(10, 8, 5, 2), max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    print("Accuracy: " + str(round(sklearn.metrics.accuracy_score(y_test, y_pred), 2)))
    print("Precision : " + str(round(sklearn.metrics.precision_score(y_test, y_pred), 4)))

    print("Recall/Sensitivity : " + str(round(sklearn.metrics.recall_score(y_test, y_pred), 4)))

    print("f1_score: " + str(round(sklearn.metrics.recall_score(y_test, y_pred, average='macro'), 4)))
    print("Feature Importances: ")
    #print(clf.feature_importances_[-10:])

    print("\nRF")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    '''print("Feature Importances: ")
    print("Meta data importance: %f",sum(clf.feature_importances_[-20:-9]))
    print("Social data importance: %f", sum(clf.feature_importances_[-8:]))'''

    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
    print("Recall/Sensitivity : " + str(sklearn.metrics.recall_score(y_test, y_pred)))
    print("Specificity : " + str(round(tn / (tn + fp), 2)))
    print("Feature Importances: ")
    #print(clf.feature_importances_[-10:])
    # get importance
    importance = clf.feature_importances_
    # summarize feature importance
    tfidf_importance = sum(importance[0:2000])
    metadata_importance = sum(importance[2000:2008])
    social_importance = sum(importance[2008:2017])
    wordembedding_importance = sum(importance[2017:2027])
    labels = ['TF-IDF', 'Word Embeddings', 'Auxiliary', 'Social']
    sizes = [tfidf_importance, wordembedding_importance, metadata_importance, social_importance]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=180)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.tight_layout()
    plt.show()


    rf_precision = sklearn.metrics.precision_score(y_test, y_pred)
    rf_recall = sklearn.metrics.recall_score(y_test, y_pred)
    rf_f1_score = 2 * rf_recall * rf_precision / (rf_precision + rf_recall + 0.000001)
    print("precision: " "%.3f" % rf_precision)
    print("recall: " "%.3f" % rf_recall)
    print("f1_score: " "%.3f" % rf_f1_score)

    print("\n Gradient Boosting")
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=1, random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Accuracy: " + str(round((tp + tn) / (tn + fp + tp + fn), 2)))
    print("Recall/Sensitivity : " + str(round(tp / (tp + fn), 2)))
    gb_precision = sklearn.metrics.precision_score(y_test, y_pred)
    print("precision: " "%.3f" % gb_precision)
    print("Specificity : " + str(round(tn / (tn + fp), 2)))
    gb_precision = sklearn.metrics.precision_score(y_test, y_pred)
    gb_recall = sklearn.metrics.recall_score(y_test, y_pred)
    gb_f1_score = 2 * gb_recall * gb_precision / (gb_precision + gb_recall + 0.000001)
    print("f1_score: " "%.3f" % gb_f1_score)
    print("Feature Importances: ")
    #print(clf.feature_importances_[-10:])


    print("\nSVM")
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))
    print("Recall/Sensitivity : " + str(sklearn.metrics.recall_score(y_test, y_pred)))
    print("Specificity : " + str(round(tn / (tn + fp), 2)))

    svm_precision = sklearn.metrics.precision_score(y_test, y_pred)
    svm_recall = sklearn.metrics.recall_score(y_test, y_pred)
    svm_f1_score = 2 * svm_precision * svm_recall / (svm_precision + svm_recall + 0.000001)

    print("precision: %.3f" % svm_precision)
    print("recall: %.3f" % svm_recall)
    print("f1_score: %.3f" % svm_f1_score)
    print("Feature Importances: ")
    #print(clf.feature_importances_[-10:])



def LSTM_ML(vaccine_data, social_feature, meta_data):
    print('...LSTM...')
    #print(social_feature)
    social_feature.fillna(0, inplace=True)
    vocabulary_size = 20000
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(vaccine_data['text'])
    sequences = tokenizer.texts_to_sequences(vaccine_data['text'])
    text_data = pad_sequences(sequences, maxlen=360)
    print('...text...')
    #combine_data = np.concatenate((text_data,meta_data.values), axis=1)
    Y = vaccine_data['Label'].map(lambda x: 1 if int(x) > 0 else 0)
    X_train, X_test, y_train, y_test = train_test_split(text_data, Y, test_size=0.25, random_state=5000)
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=360))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, np.array(y_train),  epochs=10)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall/sensitivity: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    matrix = confusion_matrix(y_test, yhat_classes)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()
    print("Specificity : " + str(round(tn / (tn + fp), 4)))

    print('...text, social feature...')
    combine_data = np.concatenate((text_data, social_feature.values), axis=1)
    Y = vaccine_data['Label'].map(lambda x: 1 if int(x) > 0 else 0)
    X_train, X_test, y_train, y_test = train_test_split(combine_data, Y, test_size=0.2, random_state=5000)
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=360))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, np.array(y_train), epochs=10)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall/sensitivity: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    matrix = confusion_matrix(y_test, yhat_classes)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()
    print("Specificity : " + str(round(tn / (tn + fp), 4)))

    print('...text, metadata...')
    combine_data = np.concatenate((text_data, meta_data.values), axis=1)
    Y = vaccine_data['Label'].map(lambda x: 1 if int(x) > 0 else 0)
    X_train, X_test, y_train, y_test = train_test_split(combine_data, Y, test_size=0.2, random_state=5000)
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=360))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, np.array(y_train), epochs=10)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall/sensitivity: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    matrix = confusion_matrix(y_test, yhat_classes)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()
    print("Specificity : " + str(round(tn / (tn + fp), 4)))

    print('...all...')
    combine_data = np.concatenate((text_data, meta_data.values, social_feature.values), axis=1)
    Y = vaccine_data['Label'].map(lambda x: 1 if int(x) > 0 else 0)
    X_train, X_test, y_train, y_test = train_test_split(combine_data, Y, test_size=0.2, random_state=5000)
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=360))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, np.array(y_train), epochs=10)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall/sensitivity: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    matrix = confusion_matrix(y_test, yhat_classes)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()
    print("Specificity : " + str(round(tn / (tn + fp), 4)))




vaccine_data, labels = get_vaccine_data()
#get_distribution_of_unique_words(vaccine_data, labels)
#storing the metadata and social features
social_feature = social_features(vaccine_data)
meta_data, hashtags = get_metadata(vaccine_data)
'''meta_data.to_pickle('metaData.pkl')
meta_data = pd.read_pickle('metaData.pkl')'''
preprocessed_data = perform_preprocessing(vaccine_data)
word_embedding_2(preprocessed_data)
#get_distribution_of_unique_words(preprocessed_data, labels)
word_embedding_dataframe = word_embedding_2(preprocessed_data)
'''word_embedding_dataframe.to_pickle('word_embedding_dataframe.pkl')
word_embedding_dataframe = pd.read_pickle('word_embedding_dataframe.pkl')'''
TF_IDF, Y = tf_idf_vectorize(preprocessed_data)
TF_IDF = TF_IDF.todense()
tf_idf_frame = pd.DataFrame(TF_IDF)
'''tf_idf_frame.to_pickle('tf_idf_frame.pkl')
tf_idf_frame = pd.read_pickle('tf_idf_frame.pkl')'''


print("  all")
dfs = [tf_idf_frame, meta_data, social_feature, word_embedding_dataframe ]
nan_value = 0
combined_data = pd.concat(dfs, axis=1).fillna(nan_value)
run_supervised_ML(combined_data,Y,labels)
LSTM_ML(preprocessed_data, social_feature, meta_data)
