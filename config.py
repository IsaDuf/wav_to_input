import argparse

# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()

# add_arg_to_parser()


# ----------------------------------------
# Some nice macros to be used for argparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    args, not_parsed = parser.parse_known_args()

    return args, not_parsed


def print_usage():
    parser.print_usage()


def print_help():
    parser.print_help()


def print_no_arg_error():
    print_help()
    print('\n' + '-' * 20)
    print("Usage example:")
    print("python main.py tricycle")

    exit(1)


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("main")

main_arg.add_argument("dataset", type=str,
                      default="tricycle",
                      help="sound dataset to convert to h5")

main_arg.add_argument("--mode", type=str,
                      default="all",
                      choices=["all", "sample"],
                      help="Run code on the whole dataset (all) or only on one sample")

main_arg.add_argument("--sr", type=int,
                      default=None,
                      help="Sample rate. Default use the native sample rate of the files")

main_arg.add_argument("--to", type=str,
                      default="log_mel",
                      choices=["pcen_mel", "log_mel"],
                      help="sound data representation you would like to save the audio dataset as h5. "
                           "Default: mel")

main_arg.add_argument("--dir_prefix", type=str,
                      default="data",
                      help="directory containing <dataset> directory")

main_arg.add_argument("--dir_suffix", type=str,
                      default="audio",
                      help="directory containing audio files. Should be a subdirectory of <dir_prefix>/<dataset>/")

main_arg.add_argument("--data_dir", type=str,
                      default="",
                      choices=[""],
                      help="place holder. Will hold <dir_prefix>/<dataset>/<dir_suffix>/")

main_arg.add_argument("--ext", nargs='*',
                      default=None,
                      help="limit type of file to process in audio directory e.g. 'mp3' 'wav'. "
                           "Default: None, correspond to ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']")

main_arg.add_argument("--load_limit", type=int,
                      default=None,
                      help="limit number of file to process in audio directory. "
                           "Default: None, process the entire dataset. "
                           "Will be overwritten to 1 if --mode sample")

main_arg.add_argument("--num_cpu", type=int,
                      default=1,
                      help="number of core available to distribute jobs")

# ----------------------------------------
# Arguments for the specific to loading audio
load_arg = add_argument_group("load")

load_arg.add_argument("--mono", type=str2bool,
                      default=True,
                      help="force an audio signal down to mono. Default: True")

load_arg.add_argument("--offset", type=float,
                      default=-0.0,
                      help="start reading at this time in audio. Default: start at beginning")

load_arg.add_argument("--duration", type=float,
                      default=None,
                      help="duration of audio to load. Default: load entire sample")

load_arg.add_argument("--chunk_duration", type=float,
                      default=None,
                      help="duration of the audio loaded to compute transformation on. "
                           "e.g. if a 10.0 sec. audio is loaded and chunk_duration is 1.0, ten non-overlapping one "
                           "second transformations will be computed. "
                           "Default: None")


# ----------------------------------------
# Arguments for the specific to loading audio
spectral_arg = add_argument_group("spectral")

spectral_arg.add_argument("--n_fft", type=int,
                          default=4096,
                          help="length of the windowed signal after padding with zeros. "
                               "Default: 2048; recommended value to be a power of 2")

spectral_arg.add_argument("--hop_length", type=int,
                          default=None,
                          help="number of audio samples between adjacent STFT columns. "
                               "Default: win_length//16")

spectral_arg.add_argument("--win_length", type=int,
                          default=None,
                          help="Each frame of audio is windowed by window of length win_length and then padded with "
                               "zeros to match n_fft. "
                               "Default win_length = n_fft")

spectral_arg.add_argument("--window", type=str,
                          default='hann',
                          help="window specification. See librosa.filter.get_window for details. "
                               "Default: hann")

spectral_arg.add_argument("--n_mels", type=int,
                          default=256,
                          help="number of Mel bands to generate. "
                               "Default: 256")