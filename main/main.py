# %%
# from gensim.models.keyedvectors import KeyedVectors
import ontology
import model
import nlp
import wikireader
import itertools
import correct_instances
import pymorphy2
from IPython import get_ipython
ipython = get_ipython()

ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

INPUT_ONTO = '../ontologies/oopOnto.owl'
OUTPUT_ONTO = '../ontologies/oopOntoPopulated.owl'
CORRECT_INSTANCES_FILE = '../instances/oop_instances.txt'
CORRECT_INSTANCES = correct_instances.parse_instances_from_txt(
    CORRECT_INSTANCES_FILE)

# %%

# with open('../texts/richter_types_wo_p.txt', 'r', encoding='utf-8') as f:
#     text_to_parse = f.read()

text_to_parse = (wikireader
                 .collect_wiki_articles
                 (['Объектно-ориентированное программирование'],
                  ["Основные понятия", "Определение ООП и его основные концепции",
                   "Особенности реализации", "Объектно-ориентированные языки"]))

print(text_to_parse[:500])

# %%
sentences = nlp.tokenize_text(text_to_parse)

words_amount = sum([len(sent) for sent in sentences])
print(f'Кол-во предложений: {str(len(sentences))}')
print(f'Кол-во слов: {str(words_amount)}')


# %%

morph = pymorphy2.MorphAnalyzer()

sentences = [[morph.parse(word)[0].normal_form
              for word in sent]
             for sent in sentences]

# %%

sentences_list = nlp.normalization(sentences)

words_amount = sum([len(sent) for sent in sentences_list])
print('После нормализации:')
print(f'Кол-во предложений: {str(len(sentences_list))}')
print(f'Кол-во слов: {str(words_amount)}')
for sent in sentences_list[:5]:
    print(sent)

# %%

sentences = nlp.add_bigrams(sentences_list, 4, 10.0)

for sent in sentences:
    for word in sent:
        for symbol in word:
            if symbol == '_':
                print(word)

nlp.show_most_frequent_words(sentences, amount=20)

# %%
# Train new model

w2v_model = model.train_word2vec(sentences,
                                 min_count=1,
                                 window=7,
                                 size=256,
                                 sample=0.001,
                                 epochs=400,
                                 sg=1,
                                 hs=1,
                                 negative=0)

print('Размер словаря: ', len(w2v_model.wv.vocab))

w2v_vectors = w2v_model.wv

# w2v_model.wv.save('../vectors/ric_types_wo_p.kv')

# %%
# Reuse vectors

# w2v_vectors = KeyedVectors.load('../vectors/ric_types.kv')
# w2v_vectors = w2v_model.wv

# %%

tagged_words = nlp.pos_tagging(sentences)
# %%

candidate_instances = ontology.find_candidate_instances(
    w2v_vectors, tagged_words, INPUT_ONTO, 30)

ontology.populate_ontology(candidate_instances,
                           input_onto=INPUT_ONTO,
                           output_onto=OUTPUT_ONTO)

ontology.calculate_metrics(candidate_instances, CORRECT_INSTANCES)


# %%
print(dict(itertools.islice(tagged_words.items(), 50)))


# %%
