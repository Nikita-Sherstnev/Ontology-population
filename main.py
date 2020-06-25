# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import correct_instances
from gensim.models import Word2Vec
import ontology
import graphics
import model
import nlp
import wikireader
import itertools
from gensim.models.keyedvectors import KeyedVectors

INPUT_ONTO = '/home/sher/Dev/Python/course_project/ontologies/ooponto.owl'
OUTPUT_ONTO = '/home/sher/Dev/Python/course_project/ontologies/test.owl'

# %%

with open('texts/richter_types.txt', 'r', encoding='utf-8') as f:
    text_to_parse = f.read()


# %%
sentences = nlp.parse_text_to_words(text_to_parse)

print(f'Количество предложений: {str(len(sentences))}')
words_amount = sum([len(sent) for sent in sentences])
print(f'Количество слов: {str(words_amount)}')


# %%

sentences_array = nlp.normalization(sentences)

print(f'Количество предложений: {str(len(sentences_array))}')
words_amount = sum([len(sent) for sent in sentences_array])
print(f'Количество слов: {str(words_amount)}')
for sent in sentences_array:
    print(sent)

# %%

sentences = nlp.add_bigrams(sentences_array, 20, 15.0)
for sent in sentences:
    # Вывод биграмм
    for word in sent:
        for symbol in word:
            if symbol == '_':
                print(word)

nlp.show_most_frequent_words(sentences, amount=20)

# %%

# w2v_model = model.train_word2vec(sentences,
#                                  min_count=2,
#                                  window=7,
#                                  size=250,
#                                  sample=0.001,
#                                  epochs=300,
#                                  sg=0,
#                                  hs=0,
#                                  negative=20)

# print('Размер словаря: ', len(w2v_model.wv.vocab))

# %%

tagged_words = nlp.pos_tagging(sentences)

word_vectors = KeyedVectors.load('vectors/ric_types.kv')
#word_vectors = w2v_model.wv
# w2v_model.wv.save('vectors/sch_types.kv')

# %%

#print(dict(itertools.islice(tagged_words.items(), 5)))
ontology.populate_ontology(word_vectors,
                           tagged_words,
                           correct_instances=correct_instances.instances,
                           input_onto=INPUT_ONTO,
                           output_onto=OUTPUT_ONTO,
                           topn=30)


# %%
# graphics.display_pca_scatterplot(word_vectors,
#                                  ['класс', 'интерфейс', 'инкапсуляция', 'наследование',
#                                   'java', 'private', 'protected', 'public',
#                                   'язык', 'полиморфизм'])

# graphics.display_pca_scatterplot(word_vectors, sample=20)
