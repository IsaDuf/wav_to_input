wav_to_input
=======================================
Preprocessing library to create preprocessed datasets from audio datasets

## Description

wav_to_input is a pre-processing pipeline to transform raw audio datasets to h5 datasets containing the desired representation for input (e.g. spectograms, mel-spectogragms, cqt, chroma). This can be used to save computational time when processing audio on the fly for a given articheture is not required.

Most of the dsp is built on the librosa library.

## Getting Started

### Dependencies

* [librosa](https://librosa.org/doc/latest/index.html)
* [numpy](https://numpy.org/) 
* [pandas](https://pandas.pydata.org/)
* [h5py](https://www.h5py.org/)

If using ```toy.py```:

* [matplotlib](https://matplotlib.org/stable/users/installing.html)

### Installing
Clone this repo and navigate to the directory.

Install dependencies can be done different ways. You can find detailed instructions on how to install each following 
the links above. To install them on a virtual environment via pip from PyPI:

create and activate virtual environment:
```
virtualenv venv
source venv/bin/activate
```
install dependencies:
```
pip install librosa
pip install pandas
pip install h5py
```
additionally, if using toy.py:
```
pip install matplotlib
```
numpy will be installed by librosa dependency.

### Specifying the data directory:
The directory containing the audio files to be processed and the annotation.csv file can be specified as 
```<dir_prefix>/<dataset_name>/<dir_suffix>```.

The default configuration is ```data/tricycle/audio```, but any of these 3 directory names can be modified 
through the ```<dataset_name> --dir_prefix --dir_suffix``` args.

### Executing program

To extract 256 bands mel-spectrograms using pcen normalization on non-overlapping 1 second extracts of the 
[SONYC-UST dataset](https://zenodo.org/record/3693077#.YCq-JBNKhTY) renamed tricycle here.
```
python main.py tricycle --chunk_duration 1.0 --n_fft 2048 --to pcen_mel --n_mels 256
```

To extract 32 bands mel-spectrograms using pcen normalization on the entire extract (10 seconds)
```
python main.py tricycle --n_fft 2048 --to pcen_mel --n_mels 32
```

To distribute the processing of the audio over 4 cores or cpu:
```
python main.py tricycle --chunk_duration 1.0 --n_fft 2048 --to pcen_mel --n_mels 256 --num_cpu 4
```

To get log mel-spectorgrams (instead of pcen):
```
python main.py tricycle --chunk_duration 1.0 --n_fft 2048 --to log_mel --n_mels 256 --num_cpu 4
```

To process a dataset name circMood whose audio files are in ```audio_tracks``` sub-directory:
```
python main.py circMood --dir_suffix audio_tracks --chunk_duration 1.0 --n_fft 2048 --to pcen_mel 
--n_mels 256 --num_cpu 4
```

An addition ```toy.py``` script can be used to experiment and visualize different dsp parameters. 
Parameters can be specified by modifying the source code:
```
toy.py
```

## Help

Check usage and list all parameters that can be specified through args, and their default values using:
```
main.py -h
```

## Authors

Isabelle Dufour  

## Version History
    
* 0.0
    * Initial Implementation

## License

This project is licensed under the [TBA] License - see the LICENSE.md file for details

## Acknowledgments
This repo is just meant to be quick pipelines built on [librosa](https://librosa.org/doc/latest/index.html) 
for preparing audio datasets for experiments.
