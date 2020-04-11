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
             'test': 'raw_data/test'}
    
    # Vocabulary
    vocabulary_options = {'source-vocab-size': 100000,
                          'target-vocab-size': 100000,
                          'source-vocab-min-frequency': 2,
                          'target-vocab-min-frequency': 2,
                          'keep-rare-words-with-embeddings': True,
                          'add-embeddings-vocab': False,
                          'source-embeddings': 'file',
                          'target-embeddings': 'file'}
    
    # Model
    model_name = 'BilstmPredictor'
    model_path = None
    # LSTM Settings (Both SRC and TGT)
    hidden_pred = 400
    rnn_layers_pred = 2
    # If set, takes precedence over other embedding params
    embedding_sizes = 200    
    # Source, Target, and Target Softmax Embedding
    source_embeddings_size = 200
    target_embeddings_size = 200
    out_embeddings_size = 200
    share_embeddings = True
    # Dropout
    dropout_pred = 0.5
    # Set to true to predict from target to source
    # (To create a source predictor for source tag prediction)
    predict_inverse = False

    ### TRAIN OPTS ###
    epochs = 6
    # Eval and checkpoint every n samples
    # Disable by setting to zero (default)
    checkpoint_validation_steps = 5000
    # If False, never save the Models
    checkpoint_save = True
    # Keep Only the n best models according to the main metric (Perplexity by default)
    # Ueful to avoid filling the harddrive during a long run
    checkpoint_keep_only_best = 1
    # If greater than zero, Early Stop after n evaluation cycles without improvement
    checkpoint_early_stop_patience = 0

    optimizer = 'adam'
    # Print Train Stats Every n batches
    log_interval = 100
    # Learning Rate
    # 1e_3 * (batch_size / 32) seems to work well
    lr = 2e-3
    learning_rate_decay = 0.6
    learning_rate_decay_start = 2
    train_batch_size = 64
    valid_batch_size = 64


    # Hyperparameter
    lengths = {'source_min_length': 1,
               'source_max_length': 50,
               'target_min_length': 1,
               'target_max_length': 50}
    
    






opt = Config()