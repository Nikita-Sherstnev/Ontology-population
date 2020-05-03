# -*- coding: utf-8 -*-
import nltk
import re
from nltk.corpus import stopwords
from gensim.models.phrases import Phraser, Phrases


def parse_text_to_words(text):
    """
    Функция принимает на вход обычный текст. На выходе получаем
    двумерный массив из предложений и слов, эти предложения составляющих. 
    """

    sentences = nltk.sent_tokenize(text, language="russian")

    new_sentences = []
    for sent in sentences:
        sent = nltk.word_tokenize(sent, language="russian")
        new_sentences.append(sent)

    return new_sentences


def normalization(sentences):
    """
    На вход принимаем выход из предыдущей функции. На выходе получаем
    двумерный массив из предложений и слов, из которого были удалены 
    знаки пунктуации и стоп-слова. Все слова приводятся к нижнему регистру.
    """
    new_sentences = []
    stopw = []
    for word in stopwords.words('russian'):
        stopw.append(word)
    stopw.append(["в", "это"])

    for sent in sentences:
        for word in sent:
            word.lower()

    for sent in sentences:
        new_list = [word.lower() for word in sent if word.isalnum()
                    and word not in stopw]
        new_sentences.append(new_list)

    return new_sentences


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


def print_most_frequent_words(sentences):
    words = [word for word in sent for sent in word]
