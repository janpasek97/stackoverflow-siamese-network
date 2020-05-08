"""
Training on SNLI dataset
"""
import matplotlib

matplotlib.use('Agg')  # necessary in order to be able to produce plots on grid

import os
import argparse
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from datetime import datetime

# suppress useless warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# import other important procedures and utilities
from network.utils.snli_generator import SNLIGenerator
from elasticsearch_dsl.connections import connections
from data.documents import ES_HOSTS, ES_LOGIN
from network.utils import constants

from network.metrics.f_scores import F1Score
from network.utils.train_utils import load_idx2word, load_word2idx, load_word2vec_embeddings, plot_graphs


# hyperparameters values
hparams = {
    constants.HP_FIRST_DROPOUT: 0.5,
    constants.HP_SECOND_DROPOUT: 0.35,
    constants.HP_BATCH_SIZE: 256,
    constants.HP_SEQUENCE_LENGTH: 150,
    constants.HP_L2_REGULARIZATION: 0.05,
}

BUFFER_SIZE = 1000
EPOCHS = 15
PREFETCH_TRAIN = 2
PREFETCH_VAL = 2

if __name__ == '__main__':
    """
    Train the model selected by command line parameters on the SNLI dataset
    """
    connections.create_connection(hosts=ES_HOSTS, http_auth=ES_LOGIN)

    # parse cmdline args
    parser = argparse.ArgumentParser("Evaluate model performance on test data and creates confusion matrix.")
    parser.add_argument("--model", required=True, choices=constants.ALL_MODELS_CONFIG.keys(),
                        help="Model to be used.")
    parser.add_argument("--loss", required=True, choices=["sparse_categorical_crossentropy"],
                        help="Loss function to be used")
    args = parser.parse_args()

    # load idx to word dictionary and it's inversion
    text_word2idx = load_word2idx(constants.TEXT_EMBEDDINGS["snli"][1])
    text_idx2word = load_idx2word(constants.TEXT_EMBEDDINGS["snli"][2])

    # load pretrained embeddings
    text_embeddings = load_word2vec_embeddings(constants.TEXT_EMBEDDINGS["snli"][0])

    # configure rest of the hyperparameters
    hparams[constants.HP_MODEL] = args.model
    hparams[constants.HP_LOSS] = args.loss
    hparams[constants.HP_EMBEDDING] = "snli"

    # define log and checkpoint paths
    now = datetime.now()
    CHECKPOINT_PATH = "network/checkpoints/snli/cp-{epoch:04d}.ckpt"
    LOG_DIR = "network/logs/fit/snli/" + args.model + "_" + args.loss + "_" + now.strftime("%d_%m_%Y_%H%M%S")
    CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

    # define hparams statistics
    with tf.summary.create_file_writer(LOG_DIR).as_default():
        hp.hparams_config(
            hparams=[constants.HP_FIRST_DROPOUT, constants.HP_SECOND_DROPOUT, constants.HP_BATCH_SIZE,
                     constants.HP_SEQUENCE_LENGTH, constants.HP_MODEL, constants.HP_LOSS,
                     constants.HP_EMBEDDING, constants.HP_L2_REGULARIZATION],
            metrics=[hp.Metric('epoch_accuracy', display_name='Accuracy', group="validation"),
                     hp.Metric('epoch_f1_score', display_name='F1Score', group="validation")],
        )

    # SELECTED MODEL MUST BE A 3 CLASS MODEL AND MUST NOT HANDLE THE CODE
    # is 2 or 3 class model chosen
    is_3_class_model = True if constants.ALL_MODELS_CONFIG[args.model][1] == 3 else False
    assert is_3_class_model

    # is selected model using code embedding
    is_code_embedding = constants.ALL_MODELS_CONFIG[args.model][2]
    assert not is_code_embedding

    # -----------------------------TRAIN DATA-------------------------------------------------------------
    train_generator = SNLIGenerator(text_word2idx, text_idx2word, split="train",
                                    max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH])
    train_ds = tf.data.Dataset.from_generator(train_generator.generate,
                                              args=[],
                                              output_types=((tf.int32, tf.int32), tf.int32),
                                              output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                              (hparams[constants.HP_SEQUENCE_LENGTH],)), ()))

    train_ds = train_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    train_ds = train_ds.batch(hparams[constants.HP_BATCH_SIZE], drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    # --------------------------------------------------------------------------------------------------
    # -----------------------------DEV DATA-------------------------------------------------------------
    val_generator = SNLIGenerator(text_word2idx, text_idx2word, split="validation",
                                  max_sentence_length=hparams[constants.HP_SEQUENCE_LENGTH])
    val_ds = tf.data.Dataset.from_generator(val_generator.generate,
                                            args=[],
                                            output_types=((tf.int32, tf.int32), tf.int32),
                                            output_shapes=(((hparams[constants.HP_SEQUENCE_LENGTH],),
                                                            (hparams[constants.HP_SEQUENCE_LENGTH],)), ()))

    val_ds = val_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    val_ds = val_ds.batch(hparams[constants.HP_BATCH_SIZE], drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    # --------------------------------------------------------------------------------------------------

    # create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        verbose=1,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        save_weights_only=True,
        save_freq='epoch'
    )

    # create tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    # create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode="max", verbose=1)

    # create chosen model
    model = constants.ALL_MODELS_CONFIG[args.model][0](text_embeddings,
                                                       hparams,
                                                       constants.TEXT_EMBEDDINGS["snli"][3])

    # create HParams callback
    hparams_callback = hp.KerasCallback(LOG_DIR, hparams)

    # instantiate chosen loss
    if args.loss == "f1_loss":
        class_nr = 3 if is_3_class_model else 2
        loss = constants.LOSSES[args.loss](class_nr)
    else:
        loss = constants.LOSSES[args.loss]()

    # Compile the model with Adam optimizer, crossentropy and macro F1 score + accuracy metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss,
                  metrics=[F1Score(num_classes=constants.ALL_MODELS_CONFIG[args.model][1], average="macro"),
                           tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    # train the model
    history = model.fit(train_ds, epochs=EPOCHS,
                        callbacks=[tensorboard_callback, cp_callback, early_stopping, hparams_callback],
                        validation_data=val_ds)

    # plot training graphs
    plot_graphs(history, "accuracy", args.model + " - " + args.loss, LOG_DIR + "/accuracy.png")
    plot_graphs(history, "f1_score", args.model + " - " + args.loss, LOG_DIR + "/f1_score.png")
    plot_graphs(history, "loss", args.model + " - " + args.loss, LOG_DIR + "/loss.png")
