from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

import sys

def ngrams(w_input, n):
    output = []

    for i in range(len(w_input) - n + 1):
        output.append(w_input[i: i + n])
    return [''.join(x) for x in output]

def get_sentences_from_files(files):
    """ Read files line by line into a list """
    sentences = []
    for file_name in files:
        with open(file_name, 'r') as fp:
            sentences.extend(fp.readlines())
    return sentences


def bag_of_words(sentences):
    """
    Build a bag of words from all the words in the dataset.
    returns: Top 1500 words are chosen by frequency
    """
    word_bag = []
    stop = stopwords.words('english')
    lmtzr = WordNetLemmatizer()

    for phrase in sentences:
        token_words = word_tokenize(phrase)
        # Filter out stop words, punctuation and numbers
        filtered_words = [lmtzr.lemmatize(w.lower()) for w in token_words if w not in stop and w.isalpha() and len(w) >= 3]
        bi_grams = ngrams(filtered_words, 2)
        word_bag.extend(filtered_words)
        word_bag.extend(bi_grams)

    freq = FreqDist(word_bag)
    return list(freq.keys())[:1500]


def export_to_file(word_bag, file_name='./data/vocab.txt'):
    with open(file_name, 'w') as fp:
        fp.write("\n".join(word_bag))


if __name__ == '__main__':
    raw_sentences = get_sentences_from_files(sys.argv[1:])
    word_bag = bag_of_words(raw_sentences)
    export_to_file(word_bag)
