"""
Provides functionality to use pre-trained Word2Vec (Wikipedia 250 dim) from tf.hub
"""
import tensorflow_hub as hub
import tensorflow as tf

embeddings = hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")
embedding_layer = hub.KerasLayer(embeddings)


@tf.function
def embed(x, y):
    """Get embedding for given batch of inputs"""
    t1, t2 = x
    t1 = embedding_layer(t1)
    t2 = embedding_layer(t2)
    return tf.stack([t1, t2]), y


@tf.function
def embed_one(x):
    """Get embedding of one word"""
    return embedding_layer(x)
