# %%
from gensim.models.keyedvectors import KeyedVectors
import confuse
import ontology
import model
import nlp
import input_reader
import itertools
import correct_instances
import pymorphy2
from IPython import get_ipython

ipython = get_ipython()
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

config = confuse.Configuration('course_project')

ONTO_BASE = '../ontologies/'
CORRECT_INSTANCES_BASE = '../instances/'
CORRECT_INSTANCES = correct_instances.parse_instances_from_txt(
    CORRECT_INSTANCES_BASE + config['correct_instances'].get())

# %%

text_to_parse = input_reader.read(config)

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

sentences = nlp.add_bigrams(sentences_list,
                            config['gensim_bigrams']['min_count'].get(),
                            config['gensim_bigrams']['threshold'].get())

for sent in sentences:
    for word in sent:
        for symbol in word:
            if symbol == '_':
                print(word)

nlp.show_most_frequent_words(sentences, amount=20)

# %%

w2v_model = model.train_word2vec(sentences,
                                 config['word2vec']['min_count'].get(),
                                 config['word2vec']['window'].get(),
                                 config['word2vec']['size'].get(),
                                 config['word2vec']['sample'].get(),
                                 config['word2vec']['epochs'].get(),
                                 config['word2vec']['sg'].get(),
                                 config['word2vec']['hs'].get(),
                                 config['word2vec']['negative'].get())

print('Размер словаря: ', len(w2v_model.wv.vocab))

w2v_vectors = w2v_model.wv

if config['save_vectors_path'] != 'None':
    w2v_model.wv.save('../vectors/' + config['save_vectors_path'].get())

if config['reuse_vectors_path'] != 'None':
    w2v_vectors = KeyedVectors.load('../vectors/' + config['reuse_vectors_path'].get())
    w2v_vectors = w2v_model.wv

# %%

tagged_words = nlp.pos_tagging(sentences)
# %%

candidate_instances = ontology.find_candidate_instances(
    w2v_vectors, tagged_words,
    ONTO_BASE + config['input_onto'].get(),
    config['similarity_top'].get())

ontology.populate_ontology(candidate_instances,
                           input_onto=ONTO_BASE + config['input_onto'].get(),
                           output_onto=ONTO_BASE + config['output_onto'].get())

ontology.print_metrics(candidate_instances, CORRECT_INSTANCES)
# %%
print(dict(itertools.islice(tagged_words.items(), 50)))


# %%
