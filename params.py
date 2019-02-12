def getParams(exp_name):
    # General Training
    epochs = 100
    batch_size = 32
    monitor = 'val_loss'
    verbose = 1
    train_augmantation = True

    # File names
    cp_name = './weights/%s_weights.h5'%exp_name
    log_name = './logs/%s_log.csv'%exp_name
    model_name = './json_models/%s'%exp_name

    #Checkpoint
    save_best_only = True
    save_weights_only = False
    period = 1

    #Earyl Stopping
    es_patience = 15
    min_delta = 0
    restore_best_weights = True

    #Reduce LR
    factor = 0.1
    lr_patience = 3
    min_lr = 0.000001

    #Logger
    separator = ','
    append = False

    params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': verbose,
            'val_to_monitor': monitor,
            'train_augmantation': train_augmantation,
            'model_name': model_name,
            'checkpoint': {
                'name': cp_name,
                'save_best_only': save_best_only,
                'save_weights_only': save_weights_only,
                'period': period
                },
            'early_stopping': {
                'patience': es_patience,
                'min_delta': min_delta,
                'restore_best_weights': restore_best_weights
                },
            'reduce_lr': {
                'factor': factor,
                'patience': lr_patience,
                'min_lr': min_lr,
                },
            'csv_logger': {
                'name': log_name,
                'separator': separator,
                'append': append
                }
            }

    return params
