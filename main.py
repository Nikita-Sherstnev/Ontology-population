# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from gensim.models import Word2Vec
import ontology
import graphics
import model
import nlp
import wikireader
import itertools


# %%

# text_to_parse = wikireader.collect_wiki_articles(
# 'Объектно-ориентированное_программирование')

# print(text_to_parse[:500])

with open('cbook.txt', 'r', encoding='utf-8') as f:
    text_to_parse = f.read()


# %%
sentences = nlp.parse_text_to_words(text_to_parse)

print(f'Количество предложений: {str(len(sentences))}')
words_amount = sum([len(sent) for sent in sentences])
print(f'Количество слов: {str(words_amount)}')
# for sent in sentences[:5]:
#    print(sent)


# %%

sentences_array = nlp.normalization(sentences)

print(f'Количество предложений: {str(len(sentences_array))}')
words_amount = sum([len(sent) for sent in sentences_array])
print(f'Количество слов: {str(words_amount)}')
for sent in sentences_array:
    print(sent)

# %%

sentences = nlp.add_bigrams(sentences_array, 10, 15.0)
for sent in sentences:
    # Вывод биграмм
    for word in sent:
        for symbol in word:
            if symbol == '_':
                print(word)

nlp.show_most_frequent_words(sentences, amount=20)

# %%

w2v_model = model.train_word2vec(sentences,
                                 min_count=2,
                                 window=7,
                                 size=250,
                                 sample=0.001,
                                 epochs=300)

print('Trained')
print('Размер словаря: ', len(w2v_model.wv.vocab))

# %%
w2v_model = Word2Vec.load('models/2_model_on_ricter.model')
tagged_words = nlp.pos_tagging(sentences)
print(dict(itertools.islice(tagged_words.items(), 5)))
ontology.populate_ontology(w2v_model,
                           tagged_words,
                           input_onto='file://C:\Dev\Python\course_project\oop.owl',
                           output_onto=r'C:\Dev\Python\course_project\test.owl',
                           topn=30)


# %%
# graphics.display_pca_scatterplot(w2v_model.wv,
#                                  ['класс', 'интерфейс', 'инкапсуляция', 'наследование',
#                                   'java', 'private', 'protected', 'public',
#                                   'язык', 'полиморфизм'])

# graphics.display_pca_scatterplot(w2v_model.wv, sample=20)


# %%
#import docx

# %%
