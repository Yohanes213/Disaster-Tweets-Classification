import re
import emoji
import contractions
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocesses a tweet text for further analysis.

    Args:
        text: The text of the tweet to be preprocessed.

    Returns:
        str: The preprocessed tweet text.
    """

    # Expand contractions
    text = contractions.fix(text)

    # Remove URLs
    url_pattern = r"(https?://)?(www\.)?(\S+\.\S+)(/\S*)?"
    text = re.sub(url_pattern, "", text)

    # Replace emojis with their meaning
    text = emoji.demojize(text)

    # Remove punctuations
    punctuation_pattern = r"[^\w\s]"
    text = re.sub(punctuation_pattern, "", text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_word = [word for word in words if word.lower() not in stop_words]
    text = " ".join(filtered_word)

    # Lowercase the text
    text = text.lower()

    return text