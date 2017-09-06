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
    def __init__(self, config, phase):
        self.config = config

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

        if phase == 'train':
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

            # Initialize encoder_embeddings to have variance=1.
            initializer = tf.random_uniform_initializer(-sqrt(3), sqrt(3), dtype=tf.float32)    # TODO: other init func?
            self.encoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=(self.config.encoder_symbols_num, self.config.embedding_size),
                                                      initializer=initializer,
                                                      dtype=tf.float32)

            # Embedded_inputs: (batch_size, time_step, embedding_size)
            self.encoder_inputs_embedded = embedding_lookup(params=self.encoder_embeddings,
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

            # Initialize decoder embeddings to have variance=1.
            initializer = tf.random_uniform_initializer(-sqrt(3), sqrt(3), dtype=tf.float32)

            self.decoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=(self.config.decoder_symbols_num, self.config.embedding_size),
                                                      initializer=initializer,
                                                      dtype=tf.float32)

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.config.hidden_units, dtype=tf.float32, name='input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.config.decoder_symbols_num, name='output_projection')

            if phase == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                decoder_inputs_embedded = embedding_lookup(
                    params=self.decoder_embeddings,
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
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))

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
                self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

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

        #Add an attentionWrapper in the lastest layer of decoder
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
        opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gradients = tf.gradients(self.loss, trainable_params)

        # TODO: use gradient clipping here?
        self.updates = opt.apply_gradients(zip(gradients, trainable_params))

    def build_feed_dict(self, **kargs):
        ret = {getattr(self, name): value for name, value in kargs.items()}
        return ret

    def build_fetch_dict(self, fetch_list):
        ret = {name: getattr(self, name) for name in fetch_list}
        return ret

class GANBasicModel(Seq2SeqBasicModel):
    def __init__(self, config,phase):
        self.config = config
        with tf.variable_scope('generator') as scope:
            Seq2SeqBasicModel.__init__(self,config,phase='decode')
        self.discriminator()

    def seq2logit(self,seq_raw, keep_prob, max_time_step, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            emb_matrix = self.encoder_embeddings
            emb_ans = tf.reduce_mean(tf.multiply(
                tf.reshape(seq_raw, [max_time_step, self.config.minibatch_size, self.config.decoder_symbols_num, 1]), emb_matrix), axis=2)

            _, state = tf.nn.dynamic_rnn(self.cell, emb_ans, sequence_length=max_time_step, initial_state=None, dtype=tf.float32,
                                         time_major=False)
            tmp_state = tf.convert_to_tensor(state[-1])  # 2*batch_size*emb_size
            h_state = tf.slice(tmp_state, [1, 0, 0], [1, self.config.minibatch_size, self.config.embedding_size])
            state = tf.reshape(h_state, [self.config.minibatch_size, -1])
            h1_size = 32
            w1 = tf.get_variable("w1", [self.config.embedding_size, h1_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b1 = tf.get_variable("b1", h1_size, initializer=tf.constant_initializer(0.0))
            h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(state, w1) + b1), keep_prob)
            w3 = tf.get_variable("w3", [h1_size, 1], initializer=tf.truncated_normal_initializer())
            b3 = tf.get_variable("b3", [1], initializer=tf.constant_initializer(0.0))
            h3 = tf.matmul(h1, w3) + b3
            return h3

    def discriminator(self):
        self.true_ans = self.decoder_inputs_train
        self.fake_ans = self.encoder_outputs
        self.true_score = self.seq2logit(self.true_ans,self.config.keep_prob,max_time_step=self.config.max_decode_step)
        self.fake_score = self.seq2logit(self.true_ans, self.config.keep_prob, max_time_step=self.config.max_decode_step,reuse=True)

        self.d_loss_real = tf.reduce_mean(self.true_score)
        self.d_loss_fake = tf.reduce_mean(self.fake_score)
        self.d_loss = self.d_loss_fake - self.d_loss_real
        self.g_loss =  tf.reduce_mean(-self.fake_score)

        self.optimizer_dis = tf.train.RMSPropOptimizer(self.config.dis_learning_rate, name='RMSProp_dis')
        self.optimizer_gen = tf.train.RMSPropOptimizer(self.config.gen_learning_rate, name='RMSProp_gen')

        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

        self.d_trainer = self.optimizer_dis.minimize(self.d_loss, var_list=self.d_params)
        self.g_trainer = self.optimizer_gen.minimize(self.g_loss, var_list=self.g_params)

        # clip discrim weights
        self.d_clip = [tf.assign(v, tf.clip_by_value(v, self.config.clip_min, self.config.clip_max)) for v in self.d_params]
