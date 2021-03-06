from math import sqrt

import tensorflow as tf
from tensorflow.contrib.seq2seq import (AttentionWrapper, BahdanauAttention,
                                        BasicDecoder, GreedyEmbeddingHelper,
                                        TrainingHelper, dynamic_decode,
                                        sequence_loss)
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.array_ops import concat
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell


class Seq2SeqBasicModel:
    def __init__(self, config, vecs, phase):
        self.config = config
        self.vecs = vecs
        self.build_base_model(phase)

    def build_base_model(self, phase):
        self.init_placeholders(phase)
        self.build_encoder()
        self.build_decoder(phase)

        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self, phase):
        # encoder_inputs: (batch_size, max_seq_len)
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=(None, None),
                                             name='encoder_inputs')

        # encoder_inputs_length: (batch_size)
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32,
                                                    shape=(None,),
                                                    name='encoder_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        # load pretrained embeddings
        self.embedding = tf.get_variable('embedding', shape=self.vecs.shape,
                                         initializer=tf.constant_initializer(self.vecs), trainable=False)

        if phase == 'train':
            # Learning rate
            self.learning_rate = tf.placeholder(tf.float32)

            # decoder_inputs: (batch_size, max_seq_len)
            self.decoder_inputs = tf.placeholder(dtype=tf.int32,
                                                 shape=(None, None),
                                                 name='decoder_inputs')

            # decoder_inputs_length: (batch_size)
            self.decoder_inputs_length = tf.placeholder(dtype=tf.int32,
                                                        shape=(None,),
                                                        name='decoder_inputs_length')

            # Add _GO and _EOS
            # TODO: maybe better add to data preprocess?
            decoder_start_token = tf.ones(shape=(self.batch_size, 1), dtype=tf.int32) * self.config._GO
            decoder_end_token = tf.ones(shape=(self.batch_size, 1), dtype=tf.int32) * self.config._EOS

            self.decoder_inputs_train = tf.concat([decoder_start_token, self.decoder_inputs], axis=1)
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # decoder_targets_train: (batch_size, max_time_steps + 1)
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets_train = tf.concat([self.decoder_inputs, decoder_end_token], axis=1)

    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder'):
            # Building encoder_cell
            self.encoder_cell = self.build_encoder_cell()

            # Embedded_inputs: (batch_size, time_step, embedding_size)
            self.encoder_inputs_embedded = embedding_lookup(params=self.embedding,
                                                            ids=self.encoder_inputs)

            # Input projection layer to feed embedded inputs to the cell
            # Fully connected layer
            input_layer = Dense(self.config.hidden_units, dtype=tf.float32, name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs, self.encoder_last_state = dynamic_rnn(
                cell=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=tf.float32,
                time_major=False    # time_major=True => the first dimension is time
            )

    def build_decoder(self, phase):
        print("building decoder and attention..")
        with tf.variable_scope('decoder'):
            # Building decoder_cell and decoder_initial_state
            decoder_cells, decoder_initial_state = self.build_decoder_cell()

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.config.hidden_units, dtype=tf.float32, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.config.decoder_symbols_num, name='output_projection')

            if phase == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                decoder_inputs_embedded = embedding_lookup(
                    params=self.embedding,
                    ids=self.decoder_inputs_train
                )

                # Embedded inputs having gone through input projection layer
                decoder_inputs_embedded = input_layer(decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = TrainingHelper(inputs=decoder_inputs_embedded,
                                                 sequence_length=self.decoder_inputs_length_train,
                                                 time_major=False,
                                                 name='training_helper')

                training_decoder = BasicDecoder(cell=decoder_cells,
                                                helper=training_helper,
                                                initial_state=decoder_initial_state,
                                                output_layer=output_layer)

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                self.decoder_outputs_train, self.decoder_last_state_train, \
                self.decoder_outputs_length_train = dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length)

                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: (batch_size, max_time_step + 1, num_decoder_symbols)
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)

                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')

                # masks: masking for valid and padded time steps, (batch_size, max_time_step + 1)
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length,
                                         dtype=tf.float32,
                                         name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                self.loss = sequence_loss(logits=self.decoder_logits_train,
                                          targets=self.decoder_targets_train,
                                          weights=masks,
                                          average_across_timesteps=True,
                                          average_across_batch=True)

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

                # Contruct graphs for minimizing loss
                self.build_optimizer()

            elif phase == 'decode':

                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones((self.batch_size,), tf.int32) * self.config._GO
                end_token = self.config._EOS

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.embedding, inputs))

                # Helper to feed inputs for greedy decoding: uses the argmax of the output
                decoding_helper = GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                        end_token=end_token,
                                                        embedding=embed_and_input_proj)

                # Basic decoder performs greedy decoding at each time step
                inference_decoder = BasicDecoder(cell=decoder_cells,
                                                 helper=decoding_helper,
                                                 initial_state=decoder_initial_state,
                                                 output_layer=output_layer)

                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #                         namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
                #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
                #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True

                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #                         namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
                #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
                # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
                #                                                    namedtuple(scores, predicted_ids, parent_ids)

                self.decoder_outputs_decode, self.decoder_last_state_decode, \
                self.decoder_outputs_length_decode = dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    # impute_finished=True,	# error occurs??
                    maximum_iterations=self.config.max_decode_step)

                # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                # Or use argmax to find decoder symbols to emit:
                # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                #                                      axis=-1, name='decoder_pred_decode')

                # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                # self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)
                self.decoder_pred_decode = self.decoder_outputs_decode.sample_id

    def build_encoder_cell(self):
        # Currently, dropout and residual component is ignored
        return MultiRNNCell([LSTMCell(self.config.hidden_units)] * self.config.encoder_depth)

    def build_decoder_cell(self):
        # No beam search currently

        # Attention
        # TODO: other attention mechanism?
        attention_mechanism = BahdanauAttention(num_units=self.config.hidden_units,
                                                memory=self.encoder_outputs,
                                                memory_sequence_length=self.encoder_inputs_length)

        decoder_cells = [LSTMCell(self.config.hidden_units)] * self.config.decoder_depth
        decoder_initial_state = list(self.encoder_last_state)

        def attn_decoder_input_fn(inputs, attention):
            if not self.config.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(self.config.hidden_units,
                                 dtype=tf.float32,
                                 name='attn_input_feeding')
            return _input_layer(concat([inputs, attention], -1))

        decoder_cells[-1] = AttentionWrapper(
            cell=decoder_cells[-1],
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.config.hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=decoder_initial_state[-1],
            alignment_history=False,
            name='Attention_Wrapper'
        )

        decoder_initial_state[-1] = decoder_cells[-1].zero_state(batch_size=self.batch_size,
                                                                 dtype=tf.float32)
        decoder_initial_state = tuple(decoder_initial_state)

        return MultiRNNCell(decoder_cells), decoder_initial_state

    def build_optimizer(self):
        trainable_params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients = tf.gradients(self.loss, trainable_params)

        # TODO: use gradient clipping here?
        self.updates = opt.apply_gradients(zip(gradients, trainable_params))

    def build_feed_dict(self, **kargs):
        ret = {getattr(self, name): value for name, value in kargs.items()}
        return ret

    def build_fetch_dict(self, fetch_list):
        ret = {name: getattr(self, name) for name in fetch_list}
        return ret
