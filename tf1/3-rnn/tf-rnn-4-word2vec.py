"""Concept 3: Word2Vec using RNN with TF.

* Recommendation:
    - gensim (instead of TF)
    - original tutorial
        https://www.tensorflow.org/tutorials/text/word_embeddings

# Foundation:
* Goal
    - word embedding -model-> a vector in n-dimensional space
* Why
    - representation of data
        - audio, image -> dense
        - text ~ 0 0 0 0.2 ... -> sparse
    - continuous vector space
        - can do vector math on words
        - e.g similar words will have similar vectors
    - dimension reduction
        - e.g 150 words | dimensions -> 2 continuous dimension
            - using t-distributed stochastic Neighbor Embedding
* Methods
    - Count-based
        - Frequency of words in corpus
    - Predictive Based
        - Neighboring words are predicted on a vector space
* Models:
    - Skip-gam
        - predict a context given word
        - better if larger data sets
    - CBOW (Continuous Bag of Words)
        - predict a next word given context
        - better for smaller data sets
        - basically noise | target classifier
            ? Target : Bone
            ? vs noise words : [book, ...., ]
        - hence, noise-contrastive training
            (1) draw k words from noise distribution
            (2) which will have high prob of correct word
                and low prob of noise word

Tasks:
    - Prediction Model
    - Generate series given a seed series
"""
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import collections
import os
import random
import zipfile

import numpy as np
import tensorflow as tf

from six.moves import urllib


# mkdir ./data && wget "http://mattmahoney.net/dc/text8.zip" -P ./data
# or using download function below
def fetch_words_data(url, to_dir):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(to_dir, "words.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])

    # Return a list of all the words in the data source
    return data.decode("ascii").split()


def create_counts(words, vocab_size=50000):
    vocab = [] + collections.Counter(words).most_common(vocab_size)
    vocab = np.array([word for word, _ in vocab])

    # Dictionary from vocab
    dictionary = {word: code for code, word in enumerate(vocab)}

    # Convert to size of words
    data = np.array([dictionary.get(word, 0) for word in words])

    return data, vocab


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    batch.fill(0)
    labels.fill(0)

    span = 2 * skip_window + 1  # [skip window target skip_window]
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0

    _v = data[data_index:data_index + span]
    buffer.extend(_v)

    # Next index start
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]

        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)

            _idx = i * num_skips + j
            batch[_idx] = buffer[skip_window]
            labels[_idx, 0] = buffer[target]

        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

        data_index = (data_index + len(data) - span) % len(data)

        return batch, labels


def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embed[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords="offset points",
                     ha="right",
                     va="bottom")


data_url = "http://mattmahoney.net/dc/text8.zip"
data_dir = "./data/word2vec/words"
words = fetch_words_data(data_url, data_dir)
words = [w for w in words if len(w) <= 5]

print(len(words))
print(" ".join(words[9000:9040]))

data, vocabulary = create_counts(words, vocab_size=50000)
print(len(words), data.shape, vocabulary.shape)
print(words[100], data[100], vocabulary[100])

data_index = 0
batch_data, batch_label = generate_batch(data, 10, 1, 2)

# Constants
batch_size = 128
embedding_size = 150
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

num_samples = 64
learning_rate = 0.01

vocabulary_size = 50000


# Place Holder
tf.reset_default_graph()
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Constant
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Variables and init
init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Cost and Lost funciton
_std = 1.0/np.sqrt(embedding_size)
_trucated_norm = tf.truncated_normal([vocabulary_size, embedding_size],
                                     stddev=_std)

nce_weights = tf.Variable(_trucated_norm)
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
nce_loss = tf.nn.nce_loss(nce_weights, nce_biases,
                          train_labels, embed,
                          num_samples, vocabulary_size)
loss = tf.reduce_mean(nce_loss)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
train = optimizer.minimize(loss)

# Compute the cosine similarity between mini batch examples
#   and all embeddings
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings,
                       normalized_embeddings,
                       transpose_b=True)
data_index = 0  # global
init = tf.global_variables_initializer()
num_steps = 5000
with tf.Session() as sess:
    sess.run(init)
    average_loss = 0

    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(data, batch_size,
                                                    num_skips,
                                                    skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([train, loss], feed_dict=feed_dict)

        average_loss += loss_val

        if step % 1000 == 0:
            if step > 0:
                average_loss = average_loss / 1000
            print("Average Loss at step ", step, " is: ", average_loss)
            average_loss = 0
        final_embeddings = normalized_embeddings.eval()

print(final_embeddings.shape)

# convert 2 dimension
tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)

plot_only = 500
low_dim_embed = tsne.fit_transform(final_embeddings[:plot_only, :])
print(low_dim_embed.shape)

labels = [vocabulary[i] for i in range(plot_only)]


plot_with_labels(low_dim_embed, labels)
plt.show()
# if do not see much similarity -> increase vocab_size and steps
