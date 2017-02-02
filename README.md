# Bag_of_Words_Meets_Bags_of_Popcorn_Kaggle

Prediction of sentiment of movie reviews
- Used BeautifulSoup and Regular Expressions in Python to remove HTML tags, numbers, punctuation to predict sentiment of movie reviews

- Used NLTK package to remove stop words and created Bag of Words of the clean review dataset using CountVectorizer from scikit-learn

- Applied Random Forest Classifier with 150 trees to predict the sentiment of 25,000 movie reviews with an accuracy of 84% on test dataset

- Improved the accuracy of random forest with 150 trees by 0.83% by using TF-IDF value of each word instead of their frequency on test data
