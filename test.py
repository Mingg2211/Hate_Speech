import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sid = SentimentIntensityAnalyzer()

def sentiment_analysis(text):
    scores = sid.polarity_scores(text)
    del scores["compound"]
    return scores

print(sentiment_analysis("This is wonderful!"))