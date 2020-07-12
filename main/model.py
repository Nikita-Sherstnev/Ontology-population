import multiprocessing
import time
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        elif(self.epoch % 50 == 0):
            print('Loss after epoch {}: {}'.format(
                self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


def train_word2vec(sentences, min_count, window, size, sample, epochs, sg, hs, negative):
    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(min_count=min_count,
                         window=window,
                         size=size,
                         sample=sample,
                         workers=cores,
                         sg=sg,
                         hs=hs,
                         negative=negative)

    w2v_model.build_vocab(sentences)

    start_time = time.time()

    w2v_model.train(
        sentences,
        total_examples=w2v_model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=[callback()])

    end_time = time.time()
    print("Время обучения: ")
    print(end_time - start_time)

    w2v_model.init_sims(replace=True)  # saves memory

    return w2v_model
