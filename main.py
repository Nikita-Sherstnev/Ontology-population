# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import io
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import time
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from owlready2 import *

import wikireader
import nlp

# %%

text_to_parse = wikireader.collect_articles(
    'Объектно-ориентированное_программирование',
    'Тип_данных')

print(text_to_parse[:500])
# %%


sentences = nlp.parse_text_to_words(text_to_parse)

print("Количество предложений:" + str(len(sentences)))
for sent in sentences[:5]:
    print(sent)


# %%

sentences_array = nlp.normalization(sentences)

print("Количество предложений:" + str(len(sentences_array)))
words_amount = 0
for s in sentences_array:
    words_amount += len(s)
print("Количество слов: " + str(words_amount))
for sent in sentences_array[:5]:
    print(sent)

# %%


def add_bigrams(sentences_array):
    """
    Добавляет биграммы в массив предложений. 
    """
    phrases = Phrases(sentences_array, min_count=8, threshold=10)
    bigram = Phraser(phrases)
    return bigram[sentences_array]


sentences = add_bigrams(sentences_array)
for sent in sentences[:100]:
    # Вывод биграмм из первых 100 предложений.
    for word in sent:
        for symbol in word:
            if symbol == "_":
                print(word)

# %%


def train_word2vec(sentences):
    """
    Принимает на вход массив предложений и выдает обученную модель
    Word2Vec.
    """
    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(min_count=1,
                         window=7,
                         size=250,
                         sample=0.01,
                         workers=cores,
                         sg=1,
                         hs=1,
                         negative=0)

    w2v_model.build_vocab(sentences)

    start_time = time.time()

    w2v_model.train(
        sentences,
        total_examples=w2v_model.corpus_count,
        epochs=300)

    end_time = time.time()
    print(end_time-start_time)

    w2v_model.init_sims(replace=True)  # saves memory

    return w2v_model


w2v_model = train_word2vec(sentences)
print("Trained")

# %%

world = World()
onto = world.get_ontology(
    "file://C:\Dev\Python\course_project\OOPOntology.owl").load()

classes = onto.classes()

for c in classes:
    """
    Извлекаем индивиды каждого класса. Находим потенциальных индивидов
    этого класса и записываем их в онтологию.
    """
    instances = []
    instances = [nlp.get_name_from_IRI(inst)
                 for inst in onto.get_instances_of(c)]

    # Удаляем индивиды, если их нет в модели.
    for inst in instances:
        if inst not in w2v_model.wv.vocab.keys():
            instances.remove(inst)

    similar = []
    if(instances != []):
        similar = w2v_model.wv.most_similar(positive=instances,
                                            topn=3)

    print(c)
    print(similar)
    for s in similar:
        # Сохранение трех найденных экземпляров класса.
        eval("onto." + str(c).split(".")[1] + "(s[0])")

onto.save(file=r"C:\Dev\Python\course_project\test.owl")
# %%
w2v_model['наследование']
