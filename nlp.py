import nltk
import re
from nltk.corpus import stopwords
from gensim.models.phrases import Phraser, Phrases
import string


def parse_text_to_words(text):
    """
    Функция принимает на вход обычный текст. На выходе получаем
    двумерный массив из предложений и слов, эти предложения составляющих. 
    """

    sentences = nltk.sent_tokenize(text, language="russian")

    tokenized_sentences = []
    for sent in sentences:
        sent = nltk.word_tokenize(sent, language="russian")
        tokenized_sentences.append(sent)

    return tokenized_sentences


def normalization(sentences):
    """
    На вход принимаем выход из предыдущей функции. На выходе получаем
    двумерный массив из предложений и слов, из которого были удалены 
    знаки пунктуации и стоп-слова. Все слова приводятся к нижнему регистру.
    """
    new_sentences = []
    extended_stopwords = []
    for word in stopwords.words('russian'):
        extended_stopwords.append(word)

    extended_stopwords.extend(["это", "является", "таким", "образом", "—", "“", "„",
                               "«", "»", 'глава', '//', "''"])

    low_case_sentences = []
    for sent in sentences:
        new_list = [word.lower() for word in sent]
        low_case_sentences.append(new_list)

    for sent in low_case_sentences:
        new_list = [word for word in sent if word not in string.punctuation
                    and word not in extended_stopwords]
        new_sentences.append(new_list)

    return new_sentences


def pos_tagging(sentences):
    '''Returns dictionary'''
    tagged_sentences = nltk.pos_tag_sents(sentences, lang='rus')

    tagged_words = {}
    for sent in tagged_sentences:
        for word in sent:
            tagged_words[word[0]] = word[1]
    return tagged_words


def add_bigrams(sentences_array, min_count=1, treshhold=10.0):
    """
    Добавляет биграммы в массив предложений. 
    """
    phrases = Phrases(sentences_array, min_count=min_count,
                      threshold=treshhold)
    bigram = Phraser(phrases)
    return bigram[sentences_array]


def get_name_from_IRI(path):
    """
    Извлекает название класса из IRI. Названия, состоящие
    из нескольких слов, объединяются в одну строку через
    нижнее подчеркивание.
    """
    name = str(path).split(".")
    name = "_".join(re.findall("[а-яa-zА-ЯA-Z][^А-ЯA-Z]*", name[1])).lower()
    return name


def show_most_frequent_words(sentences, amount=10):
    words = [word for sent in sentences for word in sent]
    print("Unique words:", len(set(words)))
    freq = nltk.FreqDist(words)
    for word in freq.most_common(amount):
        print(word)
