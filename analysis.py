import os
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams import plugin_data_pb2
import glob
import pandas as pd
from tqdm import tqdm


def read_and_save_tb():
    paths = glob.glob('./runs/**/*')

    for path in tqdm(paths):
        ea = event_accumulator.EventAccumulator(
            path,
            size_guidance={  # see below regarding this argument
                event_accumulator.IMAGES: 1,
                event_accumulator.SCALARS: 0,}
        )

        ea.Reload()
        # ea.Tags()
        d = ea.summary_metadata['_hparams_/session_start_info']
        content = d.plugin_data.content
        plugin_data = plugin_data_pb2.HParamsPluginData.FromString(content)
        W = plugin_data.session_start_info.hparams['W'].number_value
        df = pd.DataFrame(ea.Scalars('Error/test'))
        df.to_pickle(f'./errs/W{W}')


def read_and_parse_pandas():
    paths = glob.glob('./errs/*')

    df = pd.DataFrame()
    for path in tqdm(paths):
        _df = pd.read_pickle(path)
        col_name = os.path.basename(path)
        df[col_name] = _df.value

    df['step'] = _df.step
    df.set_index('step')

    return df


def organize_matrix(_df):
    cols = _df.columns
    weights = sorted([int(float(col.strip('W'))) for col in cols if col.startswith('W')])

    df = pd.DataFrame()
    for weight in weights:
        df[weight] = _df[f'W{weight}.0']

    return df
