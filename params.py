import os

#helped function to set all params of the training, modify here and it will be used in all
#other files
def getParams(exp_name, unet_type='unet', is_lstm = False):
    # General Training
    epochs = 25
    batch_size = 16
    monitor = 'val_dice_coef'
    verbose = 1
    train_augmantation = True

    # File names
    # cp_name = './weights/%s_weights.h5'%exp_name
    # log_name = './logs/%s_log.csv'%exp_name
    if is_lstm:
        log_dir_name = './logs/%s_LSTM/kfold_%s_LSTM/%s'%(unet_type,unet_type, exp_name)
    else:
        log_dir_name = './logs/%s/kfold_%s/%s'%(unet_type,unet_type, exp_name)

    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)

    cp_name = os.path.join(log_dir_name, '%s_weights.h5'%exp_name)
    log_name = os.path.join(log_dir_name, '%s_log.csv'%exp_name)
    json_name = os.path.join(log_dir_name, '%s_model.json'%exp_name)

    #Checkpoint
    save_best_only = True
    save_weights_only = False
    period = 1

    #Earyl Stopping
    es_patience = 10
    min_delta = 0
    restore_best_weights = True

    #Reduce LR
    factor = 0.1
    lr_patience = 10
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
            'model_name': json_name,
            'log_dir_name': log_dir_name,
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
