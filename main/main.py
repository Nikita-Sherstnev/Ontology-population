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

INPUT_ONTO = '../ontologies/oopOnto.owl'
OUTPUT_ONTO = '../ontologies/oopOntoPopulated.owl'
CORRECT_INSTANCES_FILE = '../instances/oop_instances.txt'
CORRECT_INSTANCES = correct_instances.parse_instances_from_txt(CORRECT_INSTANCES_FILE)

# %%

# with open('texts/richter_types.txt', 'r', encoding='utf-8') as f:
#     text_to_parse = f.read()

text_to_parse = wikireader.collect_wiki_articles(['Объектно-ориентированное программирование'],
["Основные понятия", "Определение ООП и его основные концепции",
    "Особенности реализации", "Объектно-ориентированные языки"])

print(text_to_parse)

# %%
sentences = nlp.parse_text_to_words(text_to_parse)

words_amount = sum([len(sent) for sent in sentences])
print(f'Кол-во предложений: {str(len(sentences))}')
print(f'Кол-во слов: {str(words_amount)}')


# %%

sentences_array = nlp.normalization(sentences)

words_amount = sum([len(sent) for sent in sentences_array])
print('После нормализации:')
print(f'Кол-во предложений: {str(len(sentences_array))}')
print(f'Кол-во слов: {str(words_amount)}')
for sent in sentences_array[:10]:
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
# Train new model

w2v_model = model.train_word2vec(sentences,
                                 min_count=2,
                                 window=7,
                                 size=250,
                                 sample=0.001,
                                 epochs=400,
                                 sg=0,
                                 hs=0,
                                 negative=20)

print('Размер словаря: ', len(w2v_model.wv.vocab))

word_vectors = w2v_model.wv

# Save new vectors
# w2v_model.wv.save('../vectors/sch_types.kv')

# %%
# Reuse vectors

# word_vectors = KeyedVectors.load('../vectors/ric_types.kv')
# word_vectors = w2v_model.wv
# w2v_model.wv.save('../vectors/sch_types.kv')

# %%

tagged_words = nlp.pos_tagging(sentences)
# %%

#print(dict(itertools.islice(tagged_words.items(), 5)))
ontology.populate_ontology(word_vectors,
                           tagged_words,
                           correct_instances=CORRECT_INSTANCES,
                           input_onto=INPUT_ONTO,
                           output_onto=OUTPUT_ONTO,
                           topn=30)

print(CORRECT_INSTANCES)

# %%
# graphics.display_pca_scatterplot(word_vectors,
#                                  ['класс', 'интерфейс', 'инкапсуляция', 'наследование',
#                                   'java', 'private', 'protected', 'public',
#                                   'язык', 'полиморфизм'])

# graphics.display_pca_scatterplot(word_vectors, sample=20)
