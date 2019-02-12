from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def getCallbacks(params):
    verbose = params['verbose']
    val_to_monitor = params['val_to_monitor']
    #Checkpoint callback
    cp_params = params['checkpoint']
    Checkpoint = ModelCheckpoint(
         cp_params['name'],
         monitor = val_to_monitor,
         verbose = verbose,
         save_best_only = cp_params['save_best_only'],
         save_weights_only = cp_params['save_weights_only'],
         period = cp_params['period'])

    #Earlystopping callback
    es_params = params['early_stopping']
    EarlyStop = EarlyStopping(
         monitor=val_to_monitor,
         verbose=verbose,
         min_delta = es_params['min_delta'],
         patience = es_params['patience'],
         restore_best_weights = es_params['restore_best_weights'])

    #Reduce LR callback
    lr_params = params['reduce_lr']
    ReduceLR = ReduceLROnPlateau(
         monitor = val_to_monitor,
         verbose = verbose,
         factor = lr_params['factor'],
         patience = lr_params['patience'],
         min_lr = lr_params['min_lr'])

    #CSV logger callback
    log_params = params['csv_logger']
    Logger = CSVLogger(
         log_params['name'],
         separator = log_params['separator'],
         append = log_params['append'])

    return Checkpoint, EarlyStop, ReduceLR, Logger
