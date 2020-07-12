import string
import nltk
import re
from nltk.corpus import stopwords
from gensim.models.phrases import Phraser, Phrases


def tokenize_text(plain_text) -> list:
    sentences = nltk.sent_tokenize(plain_text, language="russian")

    tokenized_sentences = []
    for sent in sentences:
        words = nltk.word_tokenize(sent, language="russian")
        tokenized_sentences.append(words)

    tokenized_sentences = clean_words(tokenized_sentences)

    return tokenized_sentences


def clean_words(tokenized_sentences) -> list:
    tokenized_sentences = [[word[1:] if word.startswith('.')
                            else word
                            for word in sent]
                           for sent in tokenized_sentences]

    return tokenized_sentences


def normalization(sentences) -> list:
    normalized_sentences = []

    extended_stopwords = list(stopwords.words('russian'))
    extended_stopwords.extend(stopwords.words('english'))
    extended_stopwords.extend(
        ["это", "являться", "такой", "образ", 'глава', 'точка', 'зрение', 'который', 'часть'])

    extended_punctuation = list(string.punctuation)
    extended_punctuation.extend(["“", "„", "—", "«", "»", '//', "''"])

    low_case_sentences = [[word.lower() for word in sent]
                          for sent in sentences]

    for sent in low_case_sentences:
        new_list = [word for word in sent if word
                    not in extended_punctuation
                    and word not in extended_stopwords]
        normalized_sentences.append(new_list)

    return normalized_sentences


def add_bigrams(sentences_array, min_count=1, threshold=10.0) -> list:
    phrases = Phrases(sentences_array, min_count=min_count,
                      threshold=threshold)
    bigram = Phraser(phrases)
    return bigram[sentences_array]


def pos_tagging(sentences) -> dict:
    tagged_sentences = nltk.pos_tag_sents(sentences, lang='rus')

    tagged_words = {}
    for sent in tagged_sentences:
        for word in sent:
            tagged_words[word[0]] = word[1]
    return tagged_words


def get_name_from_IRI(path) -> str:
    name = str(path).split(".")
    name = "_".join(re.findall("[а-яa-zА-ЯA-Z][^А-ЯA-Z]*", name[1])).lower()
    return name


def show_most_frequent_words(sentences, amount=10) -> None:
    words = [word for sent in sentences for word in sent]
    print("Unique words:", len(set(words)))
    freq = nltk.FreqDist(words)
    for word in freq.most_common(amount):
        print(word)
