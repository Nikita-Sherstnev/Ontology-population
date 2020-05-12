import multiprocessing
from gensim.models import Word2Vec
import time


def train_word2vec(sentences, min_count, window, size, sample, epochs):
    """
    Принимает на вход массив предложений и выдает обученную модель
    Word2Vec.
    """
    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(min_count=min_count,
                         window=window,
                         size=size,
                         sample=sample,
                         workers=2,
                         sg=1,
                         hs=1,
                         negative=0)

    w2v_model.build_vocab(sentences)

    start_time = time.time()

    w2v_model.train(
        sentences,
        total_examples=w2v_model.corpus_count,
        epochs=epochs)

    end_time = time.time()
    print(end_time - start_time)

    w2v_model.init_sims(replace=True)  # saves memory

    return w2v_model
