class Config:

    # Const
    UNK_ID = 0
    PAD_ID = 1
    START_ID = 2
    STOP_ID = 3
    UNALIGNED_ID = 4

    UNK = '<UNK>'
    PAD = '<PAD>'
    START = '<START>'
    STOP = '<EOS>'
    UNALIGNED = '<UNALIGNED>'
    source_side = 'source'
    target_side = 'target'

    # Data
    paths = {'train': 'raw_data/train/',
             'valid': 'raw_data/valid/',
             'test': 'raw_data/test/'}
    
    # Vocabulary
    vocabulary_options = {'source-vocab-min-frequency': 2,
                          'target-vocab-min-frequency': 2,
                          'keep-rare-words-with-embeddings': True,
                          'add-embeddings-vocab': False,
                          'source-embeddings': 'file',
                          'target-embeddings': 'file'}
    


    # Model
    model_name = 'Estimator' #'BilstmPredictor'
    pre_model_name = 'TransformerPredictor'
    # Save Model path
    checkpoint_path = 'checkpoints/'+model_name+'/'
    # Load Model path
    model_path = 'checkpoints/'+model_name+'/'+model_name+'_tran.pth'
    # Prediction path
    pred_path = 'prediction/'+model_name+'/'
    


    ###################### Predictor model setting ######################
    # LSTM Settings (Both SRC and TGT)
    hidden_pred = 400
    rnn_layers_pred = 2
    # If set, takes precedence over other embedding params
    embedding_sizes = 200    
    # Source, Target, and Target Softmax Embedding
    source_embeddings_size = 200
    target_embeddings_size = 200
    out_embeddings_size = 400
    share_embeddings = True
    # Dropout
    dropout_pred = 0.5
    # Set to true to predict from target to source
    # (To create a source predictor for source tag prediction)
    predict_inverse = False

    # Transformer perdictor
    nhead = 8
    tdropout_pred = 0.1
    transformer_layers_pred = 6
    ###################### Predictor model setting ######################

    ###################### Estimator model setting ######################
    # LSTM Settings
    hidden_est = 125
    rnn_layers_est = 1
    dropout_est = 0.0
    # Use linear layer to reduce dimension prior to LSTM
    mlp_est = True

    # Multitask Learning Settings #

    # Continue training the predictor on the postedited text.
    # If set, will do an additional forward pass through the predictor
    # Using the SRC, PE pair and add the `Predictor` loss for the tokens in the
    # postedited text PE. Recommended if you have access to PE
    # Requires setting train-pe, valid-pe
    token_level = False
    # Predict Sentence Level Scores
    # Requires setting train-sentence-scores, valid-sentence-scores
    sentence_level = True
    # Use probabilistic Loss for sentence scores instead of squared error.
    # If set, the model will output mean and variance of a truncated Gaussian
    # distribution over the interval [0, 1], and use log-likelihood loss instead
    # of mean squared error.
    # Seems to improve performance
    sentence_ll = False
    # Predict Binary Label for each sentence, indicating hter == 0.0
    # Requires setting train-sentence-scores, valid-sentence-scores
    binary_level = False
    
    load_pred_source = 'checkpoints/'+pre_model_name+'/'+pre_model_name+'.pth'
    load_pred_target = 'checkpoints/'+pre_model_name+'/'+pre_model_name+'.pth'

    # Include start and stop embedding
    start_stop = True
    ###################### Estimator model setting ######################


    ### TRAIN OPTS ###
    epochs = 4
    # Eval and checkpoint every n samples
    # Disable by setting to zero (default)
    checkpoint_validation_steps = 100
    # Save Model Every n epochs
    save_checkpoint_interval = 2
    # Keep Only the n best models according to the main metric (Perplexity by default)
    # Ueful to avoid filling the harddrive during a long run
    checkpoint_keep_only_best = 1
    # If greater than zero, Early Stop after n evaluation cycles without improvement
    checkpoint_early_stop_patience = 0

    optimizer = 'adam'
    # Print Train Stats Every n batches
    log_interval = 20
    # Learning Rate
    # 1e_3 * (batch_size / 32) seems to work well
    lr =  2e-3
    learning_rate_decay = 0.6
    learning_rate_decay_start = 2
    train_batch_size = 64
    valid_batch_size = 64

    ### Prediction OPTS ###
    seed = 42  # random
    test_batch_size = 64

    ###########################


    # Hyperparameter
    lengths = {'source_min_length': 1,
               'source_max_length': 50,
               'target_min_length': 1,
               'target_max_length': 50}
    
    






opt = Config()
