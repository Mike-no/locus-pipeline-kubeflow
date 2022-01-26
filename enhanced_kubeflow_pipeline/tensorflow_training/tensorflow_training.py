import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import sys
import os
import pickle
from typing import List, Any, Tuple
from pathlib import Path

class SeqGenerator(tf.keras.utils.Sequence):
    """
    A Sequence Generator Class. 
    It generates batches of tracklets for a Seq2Seq encoder, with two inputs and one output.
    """
    def __init__(self, data, tree, nodes, embeds, lookback, batch_size, shuffle=True, drop_remainder=True):
        self.data = data
        self.tree = tree
        self.nodes = nodes
        self.embeds = embeds
        self.lookback = lookback
        self.batch_size = batch_size
        self.indices = np.arange(data.shape[0]*np.ceil((data.shape[1]-lookback+1)).astype(int))
        if shuffle:
            np.random.shuffle(self.indices)
        num_batchs = self.indices.shape[0]//batch_size
        remainder = self.indices[-(len(self.indices) - num_batchs*batch_size):]
        self.indices = self.indices[:num_batchs*batch_size].reshape((num_batchs, batch_size))
        if not drop_remainder:
            self.indices = [elem for elem in self.indices]
            self.indices += [remainder]
        self.window = data.shape[1]-lookback+1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        indices = self.indices[i]
        ind0 = np.ceil(indices//self.window).astype(int).reshape((-1,1))
        ind1 = np.array([np.arange(i%self.window, i%self.window+self.lookback) for i in indices])
        enc = self.data[ind0, ind1, :]
        dec = np.concatenate([np.zeros((len(indices), 1, self.data.shape[-1])), enc[:, 1:, :]], axis=1)
        context = np.array([create_context(track, self.tree, self.nodes, self.embeds) for track in enc])
        return (enc, context, dec), enc

def seq2seq(units: int, n_layers: int, input_shape: Tuple[int, int], context_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> Tuple[Any, Any]:
    """
    Create a simple GRU based seq2seq autoencoder.
    It learns to reconstruct the input.

    Args:
        units (int): number of GRU units per layer
        n_layers (int): nuber of GRU layers
        shape (Tuple[int, int]): Input shape

    Returns:
        (Tuple[Any, Any]): End-to-End Autoencder and a sperate instance of the Encoder
    """

    encoder_inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_inputs')
    context_inputs = tf.keras.layers.Input(shape=context_shape, name='context_inputs')
    encoder_outputs = encoder_inputs
    for i in range(n_layers):
        encoder_outputs, encoder_states = tf.keras.layers.GRU(
            units,
            return_sequences=False if i==n_layers-1 else True,
            return_state=True, 
            name=f'GRU_encoder_{i}')(encoder_outputs)
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = tf.keras.layers.Input(shape=input_shape, name='decoder_inputs')
    decoder_outputs = tf.keras.layers.Concatenate()([decoder_inputs, context_inputs])
    for i in range(n_layers):
        decoder_outputs = tf.keras.layers.GRU(
            units, 
            return_sequences=True,
            return_state=False, 
            name=f'GRU_decoder_{i}')(
                decoder_outputs,
                initial_state=encoder_states if i==0 else None)
    decoder_outputs = tf.keras.layers.Dense(output_shape[-1])(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, context_inputs, decoder_inputs], decoder_outputs)
    return model, encoder_model

def fit(train_gen, val_gen, epochs) -> Any:
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './checkpoint.hdf5',
        onitor='val_ade', 
        save_best_only=True)
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=epochs, 
        callbacks=[earlystop, checkpoint])
    return history

def embed(encoder: Any, data_gen: Any) -> np.ndarray:
    embeddings = encoder.predict(data_gen)
    return embeddings

def create_context(track, tree, nodes, embeds):
    neigh = tree.query(track, 1, return_distance=False)
    context_ids = nodes[neigh, 0].astype('uint64').reshape((-1,))
    context = embeds.loc[context_ids].iloc[:, 1:].to_numpy()
    return context

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Input', type = str, help = "Path of the pre-processed openpflow trajectories file for the training.")
    parser.add_argument('--Input2', type = str, help = "Path of the nodes.npz file for the training.")
    parser.add_argument('--Input3', type = str, help = "Path of the tokyo_osmpbf.ft file for the training.")
    parser.add_argument('--Input4', type = str, help = "Path of the tree.pkl file for the training.")
    parser.add_argument('--Output', type = str, help = "Path of the local file where the encoder model should be written.")
    parser.add_argument('--Output2', type = str, help = "Path of the local file where the embeddings should be written.")
    args = parser.parse_args()

    if len(sys.argv) != 13:
        parser.print_help(sys.stderr)
        sys.exit(1)

    Path(args.Output).parent.mkdir(parents = True, exist_ok = True)
    Path(args.Output2).parent.mkdir(parents = True, exist_ok = True)

    lookback = 60
    batch_size = 32

    data = np.load(args.Input)
    nodes = np.load(args.Input2)
    embeds = pd.read_pickle(args.Input3)
    with open(args.Input4, 'rb') as f:
        tree = pickle.load(f)

    train_gen = SeqGenerator(data[:int(.6*len(data))], tree, nodes, embeds, lookback, batch_size)
    val_gen = SeqGenerator(data[int(.6*len(data)):int(.8*len(data))], tree, nodes, embeds, lookback, batch_size)
    test_gen = SeqGenerator(data[int(.8*len(data)):], tree, nodes, embeds, lookback, batch_size)

    model, encoder = seq2seq(8, 4, (lookback, 2), (lookback, 300), (lookback, 2))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

    history = fit(train_gen, val_gen, 1)
    score = model.evaluate(test_gen)

    encoder.save(args.Output)

    embeddings = embed(encoder, test_gen)
    np.save(args.Output2, embeddings)
    os.rename(args.Output2 + '.npy', args.Output2)
