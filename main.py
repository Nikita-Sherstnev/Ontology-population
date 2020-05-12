# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import ontology
import graphics
import model
import nlp
import wikireader
import itertools


# %%

text_to_parse = wikireader.collect_articles(
    'Объектно-ориентированное_программирование')

# print(text_to_parse[:500])


# %%
sentences = nlp.parse_text_to_words(text_to_parse)

print("Количество предложений:" + str(len(sentences)))
words_amount = sum([len(sent) for sent in sentences])
print("Количество слов: " + str(words_amount))
# for sent in sentences[:5]:
#    print(sent)


# %%

sentences_array = nlp.normalization(sentences)

print("Количество предложений:" + str(len(sentences_array)))
words_amount = sum([len(sent) for sent in sentences_array])
print("Количество слов: " + str(words_amount))
for sent in sentences_array:
    print(sent)
nlp.show_most_frequent_words(sentences_array, 15)

# %%

sentences = nlp.add_bigrams(sentences_array, 3, 10.0)
for sent in sentences[:200]:
    # Вывод биграмм из первых 200 предложений.
    for word in sent:
        for symbol in word:
            if symbol == "_":
                print(word)

# %%

w2v_model = model.train_word2vec(sentences,
                                 min_count=1,
                                 window=7,
                                 size=250,
                                 sample=0.003,
                                 epochs=200)

print("Trained")
print("Размер словаря: ", len(w2v_model.wv.vocab))

# %%

tagged_words = nlp.pos_tagging(sentences)
print(dict(itertools.islice(tagged_words.items(), 5)))
ontology.populate_ontology(w2v_model,
                           tagged_words,
                           input_onto="file://C:\Dev\Python\course_project\OOPOntology.owl",
                           output_onto=r"C:\Dev\Python\course_project\test.owl",
                           topn=20)

# %%
print(w2v_model['язык'])
#w2v_model.wv.most_similar(positive="содержит", topn=5)


# %%
graphics.display_pca_scatterplot(w2v_model.wv,
                                 ["класс", "интерфейс", "инкапсуляция", "наследование",
                                  "java", "private", "protected", "public",
                                  "язык", "полиморфизм"])

graphics.display_pca_scatterplot(w2v_model.wv, sample=20)


# %%
