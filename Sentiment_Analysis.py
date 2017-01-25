######################################################################################## SCRIPT TO PREDICT SENTIMENT OF 25,000 MOVIE REVIEWS ###
#######################################################
## READING THE FILES FROM CSV FORMAT
train = pd.read_csv("train.csv", sep=",")
test = pd.read_csv("test.csv", sep=",")

## WRITING FUNCTION TO PROCESS THE TEXT AND GET CLEAN WORDS AT THE END
Import re
from bs4 import BeautifulSoup
def text_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw review), and 
    # the output is a single string (a preprocessed text)
    # 1. Remove HTML using BeautifulSoup
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters i.e. punctuations, numbers and symbols using regular expression
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
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

### APPLYING FUNCTION ON THE RAW REVIEW TO GET PROCESSED TEXT

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
num_review = train["review"].size
# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_review ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    if( (i+1)%10 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_review ))
    clean_train_reviews.append( text_to_words( train["review"].iloc[i] ) )


## AFTER WE HAVE PROCESSED THE REVIEW’S WE WILL CONVERT THEM INTO NUMERICAL FORM SO ##THAT THEY CAN BE USED TO FIT A MODEL
### USING COUNT VECTORIZER TO FIT THE MODEL ON WORDS’ FREQUENCY
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word', Tokenizer = None, preprocessor = None,  stop_words = None, max_features = 5000)
## fit_transform() FIRST FITS THE MODEL AND LEARNS THE VOCABULARY AND SECOND TRANSFORM  
## OUR TRAIN DATA INTO FEATURE VECTORS
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

#### PERFORMING PREPROCESSING ON TEST DATA
# Create an empty list and append the clean reviews one by one
num_reviews = test["review"].size
clean_test_reviews = [] 
print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"].iloc[i] )
    clean_test_reviews.append( clean_review )

## USING COUNT VECTORIZER
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

### APPLYING RANDOM FOREST FOR CLASSIFICATION
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100, max_features=2000, verbose=2) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest = forest.fit( train_data_features, train["sentiment"] )
##### USING THE FOREST TO FIT TO TEST DATA TO CHECK ACCURACY
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Id":test["Category1"], "Cat":result} )
output.to_csv( "forest_100.csv", index=False, quoting=3 )

##### ANALYZE THE PERFORMANCE OF YOUR CLASSIFIER BY CREATING A CONFUSION MATRIX
y_actu = pd.Series(test["sentiment"], name='Actual')
y_pred = pd.Series(result, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion

### APPLYING RANDOM FOREST FOR CLASSIFICATION
# Initialize a Random Forest classifier with 150 trees
forest = RandomForestClassifier(n_estimators = 150, max_features=2000, verbose=2) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest = forest.fit( train_data_features, train["sentiment"] )
##### USING THE FOREST TO FIT TO TEST DATA TO CHECK ACCURACY
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Id":test["Category1"], "Cat":result} )
output.to_csv( "forest_150.csv", index=False, quoting=3 )

##### ANALYZE THE PERFORMANCE OF YOUR CLASSIFIER BY CREATING A CONFUSION MATRIX
y_actu = pd.Series(test["sentiment"], name='Actual')
y_pred = pd.Series(result, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion

####################################################################################
## USING TF IDF OF DOCUMENTS TO EXTRACT FEATURES
## USING TF-IDF VECTORIZER
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = None, max_features = 5000)
train_data_features = tf.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = tf.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

### APPLYING RANDOM FOREST FOR CLASSIFICATION
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100, max_features=2000, verbose=2) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest = forest.fit( train_data_features, train["sentiment"] )
##### USING THE FOREST TO FIT TO TEST DATA TO CHECK ACCURACY
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Id":test["Category1"], "Cat":result} )
output.to_csv( "forest_100.csv", index=False, quoting=3 )

##### ANALYZE THE PERFORMANCE OF YOUR CLASSIFIER BY CREATING A CONFUSION MATRIX
y_actu = pd.Series(test["sentiment"], name='Actual')
y_pred = pd.Series(result, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion

### APPLYING RANDOM FOREST FOR CLASSIFICATION
# Initialize a Random Forest classifier with 150 trees
forest = RandomForestClassifier(n_estimators = 150, max_features=2000, verbose=2) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest = forest.fit( train_data_features, train["sentiment"] )
##### USING THE FOREST TO FIT TO TEST DATA TO CHECK ACCURACY
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Id":test["Category1"], "Cat":result} )
output.to_csv( "forest_150.csv", index=False, quoting=3 )

##### ANALYZE THE PERFORMANCE OF YOUR CLASSIFIER BY CREATING A CONFUSION MATRIX
y_actu = pd.Series(test["sentiment"], name='Actual')
y_pred = pd.Series(result, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion

############## BIGRAM VECTORIZER

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, max_features = 4000)

analyze = bigram_vectorizer.build_analyzer()

## converting train and test data to bigram model
train_data_features = bigram_vectorizer.fit_transform (clean_train_reviews)
train_data_features = train_data_features.toarray()

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = bigram_vectorizer.fit_transform (clean_test_reviews)
test_data_features = test_data_features.toarray()

### APPLYING RANDOM FOREST FOR CLASSIFICATION
# Initialize a Random Forest classifier with 150 trees
forest = RandomForestClassifier(n_estimators = 150, max_features=2000, verbose=2) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest = forest.fit( train_data_features, train["sentiment"] )

##### USING THE FOREST TO FIT TO TEST DATA TO CHECK ACCURACY
result = forest.predict(test_data_features)

##### ANALYZE THE PERFORMANCE OF YOUR CLASSIFIER BY CREATING A CONFUSION MATRIX
y_actu = pd.Series(test["sentiment"], name='Actual')
y_pred = pd.Series(result, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion



