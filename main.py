# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import io
import multiprocessing
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import time
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from IPython import get_ipython
from nltk.corpus import stopwords
from owlready2 import *
from sklearn.decomposition import PCA
import wikipediaapi

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")

# %%
wiki_wiki = wikipediaapi.Wikipedia('ru')

text_to_parse = ""


def add_sections(sections, text, level=0):
    ignored_sections = ["См. также", "Примечания", "Литература", "Ссылки"]
    clean_sections = [s for s in sections if s.title not in ignored_sections]
    for s in clean_sections:
        text += s.text
        add_sections(s.sections, text, level=level + 1)

    return text


def collect_articles(*titles):
    text = ""
    for title in titles:
        page = wiki_wiki.page(title)
        text += page.summary + "\n"
        text += add_sections(page.sections, text) + "\n"

    return text


text_to_parse = collect_articles(
    'Объектно-ориентированное_программирование',
    'Java',
    'C_Sharp')

with io.open('article.txt', 'w', encoding="utf-8") as file:
    file.write(str(text_to_parse))
# %%


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


sentences = parse_text_to_words(text_to_parse)
print("Количество предложений:" + str(len(sentences)))
for sent in sentences[:5]:
    print(sent)


# %%
def normalization(sentences):
    """
    На вход принимаем выход из предыдущей функции. На выходе получаем
    двумерный массив из предложений и слов, из которого были удалены 
    знаки пунктуации и стоп-слова. Все слова приводятся к нижнему регистру.
    """
    new_sentences = []
    for sent in sentences:
        new_list = [word.lower() for word in sent if word.isalnum()
                    and word not in stopwords.words('russian')]
        new_sentences.append(new_list)

    return new_sentences


sentences_array = normalization(sentences)
print("Количество предложений:" + str(len(sentences_array)))
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

    w2v_model = Word2Vec(min_count=2,
                         window=7,
                         size=250,
                         sample=0.001,
                         workers=cores,
                         sg=1,
                         hs=0,
                         negative=10)

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


def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]

    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x+0.03, y+0.03, word)


display_pca_scatterplot(w2v_model.wv,
                        ["класс", "интерфейс", "инкапсуляция", "наследование",
                         "java", "private", "protected", "public",
                         "язык", "программирование"])
display_pca_scatterplot(w2v_model.wv, sample=20)

# %%


def get_name_from_IRI(path):
    name = str(path).split(".")
    name = "_".join(re.findall("[а-яa-zА-ЯA-Z][^А-ЯA-Z]*", name[1])).lower()
    return name


onto = get_ontology("file://C:\OOPOntology.owl").load(reload_if_newer=True)

classes = onto.classes()

for c in classes:
    instances = []

    instances = [get_name_from_IRI(inst) for inst in onto.get_instances_of(c)]

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
    # for s in similar:
    # eval("onto." + str(c).split(".")[1] + "(s[0])")

# onto.save(file=r"C:\Users\Nik\PycharmProjects\test\test.owl")


# %%
