"""
Model evaluation
"""
import matplotlib

matplotlib.use('Agg')  # necessary in order to be able to produce plots on grid

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import seaborn as sns
import csv
from datetime import datetime

# include other necessary procedures
from network.metrics.f_scores import F1Score
from network.metrics.confusion_matrix import ConfusionMatrix
from network.utils.train_utils import load_idx2word, load_word2idx, load_word2vec_embeddings
from network.utils.train_utils import TEST_DS_TEXT_FN, EMBEDDING_MATRIX_CUSTOM_CODE_FN, IDX_TO_WORD_CUSTOM_CODE_FN, \
    WORD_TO_IDX_CUSTOM_CODE_FN, TEST_DS_CODE_FN
import network.utils.data_generator_base
from network.utils.file_data_generator import FileDataGenerator
from network.utils import constants

# hyperparameters
BUFFER_SIZE = 1000
BATCH_SIZE = 128
CODE_MAX_SEQ_LENGTH = 500

# hyperparameters values
hparams = {
    constants.HP_FIRST_DROPOUT: 0.6,
    constants.HP_SECOND_DROPOUT: 0.4,
    constants.HP_BATCH_SIZE: 256,
    constants.HP_SEQUENCE_LENGTH: 150,
    constants.HP_L2_REGULARIZATION: 0.05,
}

if __name__ == '__main__':
    """
    Load the model selected by command line arguments, load its weights from a corresponding checkpoint
    and evaluate the accuracy, f1 score and confusion matrix on the test dataset.
    """
    parser = argparse.ArgumentParser("Evaluate model performance on test data and creates confusion matrix.")
    parser.add_argument("--model", required=True, choices=constants.ALL_MODELS_CONFIG.keys(),
                        help="Model to be used.")
    parser.add_argument("--embedding", required=True, choices=constants.TEXT_EMBEDDINGS.keys(),
                        help="Embedding to be used.")
    parser.add_argument("--loss", required=True, choices=["f1_loss", "sparse_categorical_crossentropy"],
                        help="Loss function to be used")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # load idx to word dictionary and it's inversion
    text_word2idx = load_word2idx(constants.TEXT_EMBEDDINGS[args.embedding][1])
    text_idx2word = load_idx2word(constants.TEXT_EMBEDDINGS[args.embedding][2])

    # load pretrained embeddings
    text_embeddings = load_word2vec_embeddings(constants.TEXT_EMBEDDINGS[args.embedding][0])

    # configure rest of the hyperparameters
    hparams[constants.HP_MODEL] = args.model
    hparams[constants.HP_LOSS] = args.loss
    hparams[constants.HP_EMBEDDING] = args.embedding

    # is 2 or 3 class model chosen
    is_3_class_model = True if constants.ALL_MODELS_CONFIG[args.model][1] == 3 else False

    # is selected model using code embedding
    is_code_embedding = constants.ALL_MODELS_CONFIG[args.model][2]
    if is_code_embedding:
        code_word2idx = load_word2idx(WORD_TO_IDX_CUSTOM_CODE_FN)
        code_idx2word = load_idx2word(IDX_TO_WORD_CUSTOM_CODE_FN)
        code_embeddings = load_word2vec_embeddings(EMBEDDING_MATRIX_CUSTOM_CODE_FN)

    # create TEST dataset
    if not is_code_embedding:
        test_generator = FileDataGenerator(TEST_DS_TEXT_FN, text_word2idx, text_idx2word,
                                           max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH],
                                           use_3_cls=is_3_class_model)
        test_ds = tf.data.Dataset.from_generator(test_generator.generate,
                                                 args=[],
                                                 output_types=((tf.int32, tf.int32), tf.int32),
                                                 output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],)), ()))
    else:
        test_generator = FileDataGenerator(TEST_DS_TEXT_FN, text_word2idx, text_idx2word,
                                           max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH],
                                           use_3_cls=is_3_class_model, embed_code=True, code2idx=code_word2idx,
                                           idx2code=code_idx2word, csv_code_file=TEST_DS_CODE_FN,
                                           max_code_length=hparams[constants.HP_SEQUENCE_LENGTH]
                                           )
        test_ds = tf.data.Dataset.from_generator(test_generator.generate,
                                                 args=[],
                                                 output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
                                                 output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],),
                                                                 (hparams[constants.HP_SEQUENCE_LENGTH],)),
                                                                ())
                                                 )

    test_ds = test_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    test_ds = test_ds.batch(hparams[constants.HP_BATCH_SIZE], drop_remainder=True)

    # Create chosen model
    if is_code_embedding:
        model = constants.ALL_MODELS_CONFIG[args.model][0](text_embeddings,
                                                           hparams,
                                                           constants.TEXT_EMBEDDINGS[args.embedding][3],
                                                           code_embeddings)
    else:
        model = constants.ALL_MODELS_CONFIG[args.model][0](text_embeddings,
                                                           hparams,
                                                           constants.TEXT_EMBEDDINGS[args.embedding][3])

    class_nr = 3 if is_3_class_model else 2

    # instantiate chosen loss
    if args.loss == "f1_loss":
        loss = constants.LOSSES[args.loss](class_nr)
    else:
        loss = constants.LOSSES[args.loss]()

    conf_mat = ConfusionMatrix(class_nr)
    # Compile the model with Adam optimizer, crossentropy and macro F1 score + accuracy metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss,
                  metrics=[F1Score(constants.ALL_MODELS_CONFIG[args.model][1], average="macro"),
                           tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                           conf_mat])

    # get latest model checkpoint
    latest_checkpoint = tf.train.latest_checkpoint("network/checkpoints/" + args.model + "_" + args.loss + "/")

    # if latest model checkpoint is availble, continue by loading all weights
    if latest_checkpoint is not None:
        print(latest_checkpoint)

        # It is necessary to run fake evaluation, in order to build the graph. Weights can be restored only
        # after the computational graph is build.
        model.evaluate(test_ds.take(1))

        # load weights
        model.load_weights(latest_checkpoint)

        result = model.evaluate(test_ds)
        conf_mat_res = conf_mat.result()

        # log the result
        res_file = open(args.output, "a+")
        result_writer = csv.writer(res_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow([str(datetime.now()), args.model, args.loss, result[0], result[1], result[2]])
        res_file.close()

        # normalize the confusion matrix
        conf_matrix_norm = np.around(
            conf_mat_res.numpy().astype('float') / conf_mat_res.numpy().sum(axis=1)[:, np.newaxis], decimals=2)

        if is_3_class_model:
            con_mat_df = pd.DataFrame(conf_matrix_norm,
                                      index=["duplicates", "similar", "different"],
                                      columns=["duplicates", "similar", "different"])
        else:
            con_mat_df = pd.DataFrame(conf_matrix_norm,
                                      index=["different", "duplicates"],
                                      columns=["different", "duplicates"])

        # plot and save the confusion matrix
        plt.figure(figsize=(8, 9))
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, vmin=0, vmax=1)

        plt.title(args.model + "-" + args.loss)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(args.model + "_" + args.loss + "_conf_mat.png", bbox_inches='tight')

    # latest checkpoint is not available
    else:
        print("No checkpoint found!")
