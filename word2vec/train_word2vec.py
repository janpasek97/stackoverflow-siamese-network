"""
Train Word2Vec model using Gensim lib
"""
import gensim
import argparse

# definition of output files
MODEL_EXPORT_FN = "word2vec.model"
WV_EXPORT_FN = "wordvectors.kv"


def train_word2vec(ds_file, min_count, emb_size, window):
    """
    Train a Word2Vec Gensim model using the given corpus.

    :param ds_file: text/code corpus for training
    :param min_count: minimum occurrence threshold to involve the token into the dictionary
    :param emb_size: desired size of the embedding vectors
    :param window: window size parameter for the CBOW model
    :return: void
    """
    with open(ds_file, "r", encoding="utf-8") as f:
        sentences = gensim.models.word2vec.LineSentence(f)
        model = gensim.models.Word2Vec(sentences, size=emb_size, window=window, min_count=min_count, workers=5)
        with open(MODEL_EXPORT_FN, "wb") as out_model:
            model.save(out_model)
        with open(WV_EXPORT_FN, "wb") as out_wv:
            model.wv.save(out_wv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train word2vec embeddings for given line dataset")
    parser.add_argument("-s", "--size", required=True, help="Embedding size", type=int)
    parser.add_argument("-f", "--file", required=True, help="Dataset file")
    parser.add_argument("-m", "--mincount", help="Minimum occurence of word to be taken into calculation", type=int,
                        required=True)
    parser.add_argument("-w", "--window", help="Window configuration for training", type=int, required=True)
    args = parser.parse_args()
    train_word2vec(args.file, args.mincount, args.size, args.window)
