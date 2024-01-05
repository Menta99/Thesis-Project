import copy
import sys

import pandas as pd
import tensorflow as tf

relative_path = ''
sys.path.append(relative_path)

from Utils.trainer_utils import inference_latency
from LatentCA.model import AutoEncoderDown2, NAFCA
from NAFNet.model import NAFNet
from Restormer.model import Restormer
from ViTCA.model import ViTCA

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    img_shape = tuple([int(arg) for arg in sys.argv[1:5]])
    print('Resolution: {}'.format(img_shape))
    runs = int(sys.argv[5])
    max_depth = 16
    num_down_sampling = 2
    cell_in_channels = cell_out_channels = max_depth
    cell_hidden_channels = 32
    pool_shape = (img_shape[0], img_shape[1] // (2 ** num_down_sampling), img_shape[1] // (2 ** num_down_sampling),
                  cell_in_channels + cell_out_channels + cell_hidden_channels)
    latent_shape = (img_shape[1] // (2 ** num_down_sampling), img_shape[1] // (2 ** num_down_sampling), max_depth)
    latent_params = {
        'input_shape': img_shape,
        'pool_shape': pool_shape,
        'latent_shape': latent_shape,
        'CAParams': None,
        'Conv2DParams1': {
            'filters': max_depth // 2,
            'kernel_size': (3, 3),
            'strides': (2, 2),
            'padding': 'same',
            'activation': 'swish',
            'batch_normalization': True,
            'dropout_rate': 0.0
        },
        'Conv2DParams2': {
            'filters': max_depth,
            'kernel_size': (3, 3),
            'strides': (2, 2),
            'padding': 'same',
            'activation': 'swish',
            'batch_normalization': True,
            'dropout_rate': 0.0
        },
        'PassThroughParams1': {
            'filters': max_depth // 2,
            'kernel_size': (3, 3),
            'strides': (2, 2),
            'padding': 'same',
            'activation': 'swish',
            'batch_normalization': True,
            'dropout_rate': 0.0
        },
        'Conv2DTransposeParams2': {
            'filters': max_depth // 2,
            'kernel_size': (3, 3),
            'strides': (2, 2),
            'padding': 'same',
            'activation': 'swish',
            'batch_normalization': True,
            'dropout_rate': 0.0
        },
        'MixParams': {
            'filters': max_depth // 2,
            'kernel_size': (3, 3),
            'strides': (1, 1),
            'padding': 'same',
            'activation': 'swish',
            'batch_normalization': True,
            'dropout_rate': 0.0
        },
        'Conv2DTransposeParams1': {
            'filters': 3,
            'kernel_size': (3, 3),
            'strides': (2, 2),
            'padding': 'same',
            'activation': 'sigmoid',
            'batch_normalization': True,
            'dropout_rate': 0.0
        }
    }
    latent_nafca_params = {
        'class': NAFCA,
        'localized_attention_neighbourhood': (3, 3),
        'ffn_expand': 4,
        'dropout': 0.1,
        'embed_dim': 128,
        'cell_in_channels': cell_in_channels,
        'cell_out_channels': cell_out_channels,
        'cell_hidden_channels': cell_hidden_channels
    }
    latent_vitca_params = {
        'class': ViTCA,
        'localized_attention_neighbourhood': (3, 3),
        'patch_size': 1,
        'pos_encoding_max_freq': 5,
        'depth': 1,
        'heads': 4,
        'mlp_dim': 64,
        'dropout': 0.0,
        'embed_cells': True,
        'embed_dim': 128,
        'embed_dropout': 0.0,
        'pool_shape': pool_shape,
        'cell_in_channels': cell_in_channels,
        'cell_out_channels': cell_out_channels,
        'cell_hidden_channels': cell_hidden_channels
    }
    nafnet_params = {
        'input_shape': img_shape,
        'width': 64,
        'enc_block_nums': [2, 2, 4, 8],
        'middle_block_num': 12,
        'dec_block_nums': [2, 2, 2, 2]
    }
    restormer_params = {
        'input_shape': img_shape,
        'out_channels': 3,
        'dim': 48,
        'num_blocks': [4, 6, 6, 8],
        'num_refinement_blocks': 4,
        'heads': [1, 2, 4, 8],
        'ffn_expansion_factor': 2.66,
        'bias': False
    }
    vitca_params = {
        'input_shape': img_shape,
        'localized_attention_neighbourhood': [3, 3],
        'patch_size': 1,
        'overlapping_patches': False,
        'pos_encoding_method': 'vit_handcrafted',
        'pos_encoding_basis': 'raw_xy',
        'pos_encoding_max_freq': 5,
        'depth': 1,
        'heads': 4,
        'mlp_dim': 64,
        'dropout': 0.0,
        'cell_init': 'constant',
        'cell_in_channels': img_shape[-1],
        'cell_out_channels': img_shape[-1],
        'cell_hidden_channels': cell_hidden_channels,
        'embed_cells': True,
        'embed_dim': 128,
        'embed_dropout': 0.0
    }
    latent_nafca_params_complete = copy.deepcopy(latent_params)
    latent_nafca_params_complete['CAParams'] = latent_nafca_params
    latent_vitca_params_complete = copy.deepcopy(latent_params)
    latent_vitca_params_complete['CAParams'] = latent_vitca_params

    # Initialize models

    nafnet_model = NAFNet(nafnet_params)
    restormer_model = Restormer(restormer_params)
    vitca_model = ViTCA(vitca_params)
    latent_vitca_model = AutoEncoderDown2(latent_vitca_params_complete)
    latent_nafca_model = AutoEncoderDown2(latent_nafca_params_complete)

    # Inference

    results_nafnet = inference_latency(model=nafnet_model, batch_shape=img_shape, num_runs=runs, model_type='classic')
    results_restormer = inference_latency(model=restormer_model, batch_shape=img_shape, num_runs=runs,
                                          model_type='classic')
    results_vitca = inference_latency(model=vitca_model, batch_shape=img_shape, num_runs=runs, model_type='nca')
    results_latent_vitca = inference_latency(model=latent_vitca_model, batch_shape=img_shape, num_runs=runs,
                                             model_type='latent_nca')
    results_latent_nafca = inference_latency(model=latent_nafca_model, batch_shape=img_shape, num_runs=runs,
                                             model_type='latent_nca')

    # Create dataframe of results in ms

    df = pd.DataFrame(data={'NAFNet': results_nafnet, 'Restormer': results_restormer, 'ViTCA': results_vitca,
                            'LatentViTCA': results_latent_vitca, 'LatentNAFCA': results_latent_nafca},
                      columns=['NAFNet', 'Restormer', 'ViTCA', 'LatentViTCA', 'LatentNAFCA'])
    df /= 10 ** 6
    print('Elapsed time:\n{}'.format(df))
    print('Mean elapsed time:\n{}'.format(df.mean()))
