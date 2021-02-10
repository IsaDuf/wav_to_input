import os
import sys
import librosa
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Process, Lock


from config import *


def print_dataset_details(df):
    """
    Print a few information about the dataset.

    Implementation is limited to immediate needs, but should be extended.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    None

    """
    hour = list(df['hour'])
    print("hour data:")
    for i in range(0, 24):
        print('{}: {}'.format(i, hour.count(i)))

    split = list(df['split'])

    print('train: ', split.count('train'))
    print('validate: ', split.count('validate'))
    print('test: ', split.count('test'))


def create_df():
    """
    Create a pandas DataFrame from annotations.csv.

    Assumes the annotations.csv file is in data/config.dataset/ directory.

    Returns
    -------
    df: pandas.DataFrame

    """
    csv_dir = '/'.join((config.dir_prefix, config.dataset, 'annotations.csv'))
    df = pd.read_csv(csv_dir)
    df = df.drop_duplicates(subset=['audio_filename'])

    # print_dataset_details(df)

    return df


def save_to_h5(new_rep, data, i):
    """
    Populate a `track` group of h5 file created earlier with one dataset for each individual input representation.

    Create a group named after the data index value of one sample (one file). The group is then populated with:
    'X_{}' : a dataset containing the numpy array of the new data representation (mel_log, mel_pcen etc)
            where {} is the chunk index. If 'config.duration = None', there should be only one chunk ('X_0')
    'X_sensor' : sensor id
    'y_hour' : hour label
    'y_day' : day label
    'y_week' : week label

    attribute: 'filename': contain original 'audio_filename'


    Parameters
    ----------
    new_rep : numpy.ndarray
        new representation of the data to be used as input

    data : pd.DataFrame
        a pandas data frame containing only the row associated to the track

    i : int
        the index of the chunk to be inserted in datasets. E.g. a 10 seconds track could have 10 chunks of 1 second.
        Each chunk is saved using the 'X_{i}' convention.

    Returns
    -------
    None

    Notes
    -----
    Implementation is limited to immediate needs, but should be extended to accept flexible annotations
    """
    dataset = '/'.join([config.dataset] * 2)
    this_file = 'track/' + str(data.index.values[0])

    with h5py.File('data/' + dataset + '.h5', 'a') as f:
        if this_file in f:
            this_group = f[this_file]
            this_group.create_dataset("X_{}".format(i), data=new_rep)
        else:
            this_group = f.create_group(this_file)
            this_group.create_dataset("X_{}".format(i), data=new_rep)
            this_group.create_dataset('X_sensor', data=data.iloc[0]['sensor_id'])
            this_group.create_dataset('y_hour', data=data.iloc[0]['hour'])
            this_group.create_dataset('y_day', data=data.iloc[0]['day'])
            this_group.create_dataset('y_week', data=data.iloc[0]['week'])
            this_group.attrs['filename'] = data.iloc[0]['audio_filename']


def transform_to_chunks(sublist, df, lock):
    """
    Iterates through list of audio files and transform to specified representation (e.g. pcen_mel, log_mel).

    Parameters
    ----------
    sublist : list
        list containing all the audio files to process

    df : pandas.DataFrame
        dataframe containing the annotations for the dataset being processed

    lock : LockType
        lock used when writing to disk

    Returns
    -------
    None

    """
    files = sublist

    for file in files:
        y, sr = librosa.load(file,
                             sr=config.sr,
                             mono=config.mono,
                             duration=config.duration)

        tot_duration = librosa.get_duration(y=y, sr=sr)
        n_chunk = 1 if config.chunk_duration is None else int(tot_duration/config.chunk_duration)

        for i, audio_chunk in enumerate(np.array_split(y, n_chunk)):
            if config.to == 'log_mel':
                mel_audio = librosa.feature.melspectrogram(y=audio_chunk,
                                                           sr=sr,
                                                           n_fft=config.n_fft,
                                                           hop_length=config.hop_length,
                                                           win_length=config.win_length,
                                                           window=config.window,
                                                           n_mels=config.n_mels,
                                                           center=True,
                                                           power=2,
                                                           htk=True)
                new_rep = librosa.amplitude_to_db(mel_audio, ref=np.max)

            elif config.to == 'pcen_mel':
                mel_audio = librosa.feature.melspectrogram(y=audio_chunk,
                                                           sr=sr,
                                                           n_fft=config.n_fft,
                                                           hop_length=config.hop_length,
                                                           win_length=config.win_length,
                                                           window=config.window,
                                                           n_mels=config.n_mels,
                                                           center=True,
                                                           power=1,
                                                           htk=True)
                new_rep = librosa.pcen(mel_audio,
                                       sr=sr,
                                       hop_length=config.hop_length,
                                       eps=1e-10,
                                       gain=0.7,
                                       bias=0,
                                       power=0.125,
                                       time_constant=0.25)

            else:
                raise NotImplementedError("This representation is currently not available")

            filename = file.split('/')[-1]
            data = df.loc[df['audio_filename'] == filename]
            with lock:
                save_to_h5(new_rep, data, i)


def main():
    """
    Create a h5 file containing a new data representation of all the audio files, along with the labels and
    diverse attributes of a given audio dataset and annotations.csv.

    Can be distributed over multiple cores by specifying the number of cores available through the --num_cpu argument.
    Returns
    -------
    None
    """
    n_sample = 'all' if config.load_limit is None else str(config.load_limit)
    print("Processing {} samples in {}".format(n_sample, config.dataset))

    # create a file of all audio files contained in directory
    files = librosa.util.find_files(config.data_dir,
                                    ext=config.ext,
                                    limit=config.load_limit)

    # create pandas dataframe from annotations.csv
    df = create_df()
    dataset = '/'.join([config.dataset] * 2)
    test_df = df.loc[df['split'] == 'test']
    train_df = df.loc[df['split'] == 'train']
    val_df = df.loc[df['split'] == 'val']

    # create h5 file and populates splits group, and hyper-parameters as info attributes
    with h5py.File('data/' + dataset + '.h5', 'w') as f:
        print("data/{}/{}.h5 file created".format(config.dataset, config.dataset))
        splits = f.create_group("splits")
        splits.create_dataset("train", data=train_df.index.values)
        splits.create_dataset("test", data=test_df.index.values)
        splits.create_dataset("val", data=val_df.index.values)

        f.create_group("info")
        f["info"].attrs.create("n_fft", config.n_fft)
        f["info"].attrs.create("win_length", config.win_length)
        f["info"].attrs.create("hop_length", config.hop_length)
        f["info"].attrs.create("n_mels", config.n_mels)
        f["info"].attrs.create("type", config.to)
        f["info"].attrs.create("duration", 'NoneType' if config.duration is None else config.duration)

    # setup multiprocessing stuff
    slice_end = -(-len(files)//config.num_cpu)
    process_list = []
    start = 0
    lock = Lock()

    # distribute and execute jobs
    for i in range(config.num_cpu):
        sub_list = files[start:start + slice_end]
        process = Process(target=transform_to_chunks,
                          args=(sub_list, df, lock))
        process_list.append(process)
        process.start()
        start = start + slice_end

    for process in process_list:
        process.join()

    # add shape attribute to h5 file
    with h5py.File('data/' + dataset + '.h5', "a") as f:
        f["info"].attrs.create("data_shape", f["track"]["0"]["X_0"].shape)


if __name__ == "__main__":

    config, not_parsed = get_config()

    if len(sys.argv) == 1:
        print_no_arg_error()

    if len(not_parsed) > 0:
        print_usage()
        print_help()
        exit(1)

    if config.win_length is None:
        config.win_length = config.n_fft

    if config.hop_length is None:
        config.hop_length = config.win_length//16

    config.data_dir = '/'.join((config.dir_prefix, config.dataset, config.dir_suffix))
    config.load_limit = 1 if config.mode == "sample" else config.load_limit

    print(config)
    main()

