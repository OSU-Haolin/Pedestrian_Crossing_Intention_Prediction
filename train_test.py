"""Summary
"""
import os
import yaml
import getopt
import sys

import numpy as np
from tensorflow.keras import backend as K

from action_predict import action_prediction
from action_predict import ActionPredict
#from new_model import NewModel, HybridModel, MultiRNN3D, MultiRNN3D_MATT

from jaad_data import JAAD
# if use PIE data:
# from pie_data import PIE
import tensorflow as tf
# tf.test.is_gpu_available()

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )


# config = tf.compat.v1.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction=0.8
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)



# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)



# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# tf.keras.backend.set_session(tf.Session(config=config));

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def write_to_yaml(yaml_path=None, data=None):
    """
    Write model to yaml results file
    
    Args:
        model_path (None, optional): Description
        data (None, optional): results from the run
    
    Deleted Parameters:
        exp_type (str, optional): experiment type
        overwrite (bool, optional): whether to overwrite the results if the model exists
    """
    with open(yaml_path, 'w') as yamlfile:
        yaml.dump(data, yamlfile)


def run(config_file=None):
    """
    Run train and test on the dataset with parameters specified in configuration file.
    
    Args:
        config_file: path to configuration file in yaml format
        dataset: dataset to train and test the model on (pie, jaad_beh or jaad_all)
    """
    print(config_file)
    # Read default Config file
    configs_default ='config_files/configs_default.yaml'
    with open(configs_default, 'r') as f:
        configs = yaml.safe_load(f)

    with open(config_file, 'r') as f:
        model_configs = yaml.safe_load(f)

    # Update configs based on the model configs
    for k in ['model_opts', 'net_opts']:
        if k in model_configs:
            configs[k].update(model_configs[k])

    # Calculate min track size
    tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
        configs['model_opts']['time_to_event'][1]
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte

    # update model and training options from the config file
    for dataset_idx, dataset in enumerate(model_configs['exp_opts']['datasets']):
        configs['data_opts']['sample_type'] = 'beh' if 'beh' in dataset else 'all'
        configs['model_opts']['overlap'] = 0.6 if 'pie' in dataset else 0.8
        configs['model_opts']['dataset'] = dataset.split('_')[0]
        configs['train_opts']['batch_size'] = model_configs['exp_opts']['batch_size'][dataset_idx]
        configs['train_opts']['lr'] = model_configs['exp_opts']['lr'][dataset_idx]
        configs['train_opts']['epochs'] = model_configs['exp_opts']['epochs'][dataset_idx]

        model_name = configs['model_opts']['model']
        # Remove speed in case the dataset is jaad
        if 'RNN' in model_name and 'jaad' in dataset:
            configs['model_opts']['obs_input_type'] = configs['model_opts']['obs_input_type']

        for k, v in configs.items():
            print(k,v)

        # set batch size
        if model_name in ['ConvLSTM']:
            configs['train_opts']['batch_size'] = 2
        if model_name in ['C3D', 'I3D']:
            configs['train_opts']['batch_size'] = 4
        if model_name in ['PCPA']:
            configs['train_opts']['batch_size'] = 1
        if 'MultiRNN' in model_name:
            configs['train_opts']['batch_size'] = 8
        if model_name in ['TwoStream']:
            configs['train_opts']['batch_size'] = 16

        if configs['model_opts']['dataset'] == 'pie':
            pass
            # imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])
        elif configs['model_opts']['dataset'] == 'jaad':
            # if use docker:
            # imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])

            # if use local path
            imdb = JAAD(data_path='/home/haolin/CITR/PedestrianActionBenchmark/JAAD/')

        # get sequences
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
        beh_seq_val = None
        # Uncomment the line below to use validation set
        # beh_seq_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts']) ## load_dataset

        # get the model
        method_class = action_prediction(configs['model_opts']['model'])(**configs['net_opts'])

        # train and save the model
        saved_files_path = method_class.train(beh_seq_train, beh_seq_val, **configs['train_opts'],
                                              model_opts=configs['model_opts'])
        # test and evaluate the model
        acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)

        # save the results
        data = {}
        data['results'] = {}
        data['results']['acc'] = float(acc)
        data['results']['auc'] = float(auc)
        data['results']['f1'] = float(f1)
        data['results']['precision'] = float(precision)
        data['results']['recall'] = float(recall)
        write_to_yaml(yaml_path=os.path.join(saved_files_path, 'results.yaml'), data=data)

        data = configs
        write_to_yaml(yaml_path=os.path.join(saved_files_path, 'configs.yaml'), data=data)

        print('Model saved to {}'.format(saved_files_path))

def usage():
    """
    Prints help
    """
    print('Benchmark for evaluating pedestrian action prediction.')
    print('Script for training and testing models.')
    print('Usage: python train_test.py [options]')
    print('Options:')
    print('-h, --help\t\t', 'Displays this help')
    print('-c, --config_file\t', 'Path to config file')
    print()

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'config_file'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)    

    config_file = None
    model_name = None
    dataset = None

    for o, a in opts:
        if o in ["-h", "--help"]:
            usage()
            sys.exit(2)
        elif o in ['-c', '--config_file']:
            config_file = a

    # if neither the config file or model name are provided
    if not config_file:
        print('\x1b[1;37;41m' + 'ERROR: Provide path to config file!' + '\x1b[0m')
        usage()
        sys.exit(2)

    run(config_file=config_file)
