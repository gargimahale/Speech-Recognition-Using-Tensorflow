import os
import re
import sys
import urllib
from pathlib import Path
import pickle
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from python_speech_features import mfcc


def report_hook(count, block_size, total_size):
    """
    Shows the downloading progress.
    """
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()


def download_file(download_path, save_path, file_name):
    """
    Downloads data and created dirs if necessary.
    Args:
        download_path: String. path to download data from.
        save_path: String. path to save data to.
        file_name: String. name to save tdata with.

    """
    if os.path.exists(save_path):
        if file_name in os.listdir(save_path):
            return
        else:
            file_path = os.path.join(save_path, file_name)
            urllib.request.urlretrieve(download_path, file_path, report_hook)
    else:
        os.makedirs(save_path)
        file_path = os.path.join(save_path, file_name)
        urllib.request.urlretrieve(download_path, file_path, report_hook)


def plot_wave(path):
    """
    Args:
        path: Path to the audio file we want to plot
    """
    samples, sample_rate = librosa.load(path, mono=True, sr=None)
    plt.figure(figsize=[15, 5])
    librosa.display.waveplot(samples, sr=sample_rate)
    plt.show()


def plot_melspectogram(path, n_mels=128):
    """
    Args:
        path: The path to to the audiofile we want to plot.
    """
    samples, sample_rate = librosa.load(path, mono=True, sr=None)
    plt.figure(figsize=[20, 5])
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S)
    plt.show()



def audioToInputVector(audio_filename, numcep, numcontext):
    """
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.

    Borrowed from Mozilla's Deep Speech and slightly modified.
    https://github.com/mozilla/DeepSpeech
    """

    audio, fs = librosa.load(audio_filename)

    # # Get mfcc coefficients
    features = mfcc(audio, samplerate=fs, numcep=numcep, nfft=551)
    # features = librosa.feature.mfcc(y=audio,
    #                                 sr=fs,
    #                                 n_fft=551,
    #                                 n_mfcc=numcep).T

    # We only keep every second feature (BiRNN stride = 2)
    features = features[::2]

    # One stride per time step in the input
    num_strides = len(features)

    # Add empty initial and final contexts
    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)

    features = np.concatenate((empty_context, features, empty_context))

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2 * numcontext + 1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, numcep),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    # Flatten the second and third dimensions
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    # Whiten inputs (TODO: Should we whiten?)
    # Copy the strided array so that we can write to it safely
    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    # Return results
    return train_inputs


def load_data(dir_path, how_many=0):
    """

    Args:
        dir_path: path to the directory with txt and audio files.
        how_many: Integer. Number of directories we want to iterate,
                  that contain the audio files and transcriptions.

    Returns:
        txts: The spoken texts extracted from the .txt files,
              which correspond to the .flac files in audios.
              Text version.
        audios: The .flac file paths corresponding to the
                sentences in txts. Spoken version.

    """
    dir_path = Path(dir_path)
    txt_list = [f for f in dir_path.glob('**/*.txt') if f.is_file()]
    audio_list = [f for f in dir_path.glob('**/*.flac') if f.is_file()]

    print('Number of audio txt paths:', len(txt_list))
    print('Number of audio file paths:', len(audio_list))

    txts = []
    audios = []
    audio_paths = []

    # for development we want to reduce the numbers of files we read in.
    if how_many == 0:
        how_many = len(txt_list)

    for i, txt in enumerate(txt_list[:how_many]):
        print('Text#:', i+1)
        with open(txt) as f:
            for line in f.readlines():
                for audio in audio_list:
                    if audio.stem in line:
                        line = re.sub(r'[^A-Za-z]', ' ', line)
                        line = line.strip()
                        txts.append(line)
                        audios.append(audioToInputVector(audio, 26, 9))
                        audio_paths.append(audio)
                        break
    return txts, audios, audio_paths


def split_txts(txts):
    """
    Args:
        txts: The texts that will be split
              into single characters

    Returns:
        The splitted texts and array of all unique characters
        in those texts.

    """
    txts_splitted = []
    unique_chars = set()

    for txt in txts:
        splitted = list(txt)
        splitted = [ch if ch != ' ' else '<SPACE>' for ch in splitted]
        txts_splitted.append(splitted)
        unique_chars.update(splitted)
    return txts_splitted, sorted(unique_chars)


def create_lookup_dicts(unique_chars, specials=None):
    """

    Args:
        unique_chars: Set of unique chars appearning in texts.
        specials: Special characters we want to add to the dict,
                  such as <PAD>, <SOS> or <EOS>

    Returns:
        char2ind: look updict from character to index
        ind2char: lookup dict from index to character

    """
    char2ind = {}
    ind2char = {}
    i = 0

    if specials is not None:
        for sp in specials:
            char2ind[sp] = i
            ind2char[i] = sp
            i += 1
    for ch in unique_chars:
        char2ind[ch] = i
        ind2char[i] = ch
        i += 1
    return char2ind, ind2char


def convert_txt_to_inds(txt, char2ind, eos=False, sos=False):
    """

    Args:
        txt: Array of chars to convert to inds.
        char2ind: Lookup dict from chars to inds.

    Returns: The converted chars, i.e. array of ints.

    """
    txt_to_inds = [char2ind[ch] for ch in txt]
    if eos:
        txt_to_inds.append(char2ind['<EOS>'])
    if sos:
        txt_to_inds.insert(0, char2ind['<SOS>'])
    return txt_to_inds


def convert_inds_to_txt(inds, ind2char):
    """

    Args:
        inds: Array of ints to convert to chars
        ind2char: Lookup dict from ind to chars

    Returns: The converted inds, i.e. array of chars.

    """
    inds_to_txt = [ind2char[ind] for ind in inds]
    return inds_to_txt


def process_txts(txts, specials):
    """
    Processes the texts. Calls the functions split_txts,
    create_lookup_dicts and uses convert_txt_to_inds.

    Args:
        txts: Array of strings. Input texts.
        specials: Specials tokens we want to include in the
                  lookup dicts

    Returns:
        txts_splitted: Array of the input texts splitted up into
                       characters
        unique_chars: Set of Unique chars appearing in input texts.
        char2ind: Lookup dict from character to index.
        ind2char: Lookup dict from index to character.
        txts_converted: txts splitted converted to indices of
                        word2ind. i.e. array of arrays of ints.

    """
    txts_splitted, unique_chars = split_txts(txts)
    char2ind, ind2char = create_lookup_dicts(unique_chars, specials)
    txts_converted = [convert_txt_to_inds(txt, char2ind, eos=True, sos=True)
                      for txt in txts_splitted]

    return txts_splitted, unique_chars, char2ind, ind2char, txts_converted


def sort_by_length(audios,
                   txts,
                   audio_paths,
                   txts_splitted,
                   txts_converted,
                   by_text_length=True):
    """
    Sort texts by text length from shortest to longest.
    To keep everything in order we also sort the rest of the data.

    Args:
        by_text_length: Boolean. Sort either by text lengths or
                        by length of audios.

    Returns:

    """

    # check if that works. if not audios isn't a  numpy array.
    # in that case we could convert beforehand.
    if by_text_length:
        indices = [txt[0] for txt in sorted(enumerate(txts_converted), key=lambda x: len(x[1]))]
    else:
        indices = [a[0] for a in sorted(enumerate(audios), key=lambda x: x[1].shape[0])]
    txts_sorted = np.array(txts)[indices]
    audios_sorted = np.array(audios)[indices]
    audio_paths_sorted = np.array(audio_paths)[indices]
    txts_splitted_sorted = np.array(txts_splitted)[indices]
    txts_converted_sorted = np.array(txts_converted)[indices]

    return txts_sorted, audios_sorted, audio_paths_sorted, txts_splitted_sorted, txts_converted_sorted


def preds2txt(preds,
              ind2char,
              beam=False):
    """
    Converts the predictions to text and removes
    <SOS> and <EOS> tokens
    Args:
        preds: the predictions. output of either
               greedy or beam search decoding.
        ind2char: Lookup dict from index to character.
        beam: Boolean. Wheter preds is the output
              of greedy or beam search decoding.

    Returns:
        p2t: The converted predictions,
             i.e. the predicted text.

    """
    if beam:
        p2t = []
        for batch in preds:
            for sentence in batch:
                converted_sentence = []
                for p in sentence:
                    converted_ch = ind2char[p[0]]
                    if converted_ch != '<EOS>' and converted_ch != '<SOS>':
                        converted_sentence.append(converted_ch)
                p2t.append(converted_sentence)
                converted_sentence = []
    else:
        p2t = []
        for batch in preds:
            for sentence in batch:
                converted_sentence = convert_inds_to_txt(sentence, ind2char)
                converted_sentence = [ch for ch in converted_sentence
                                      if ch != '<EOS>' and ch != '<SOS>']
                p2t.append(converted_sentence)

    return p2t


def print_samples(preds, targets):
    """
    Print predicted text and actual text side by side
    and measures the accuracy of the predicted
    characters. If we produce shorter texts than
    the actual ones we penalize the acc score.

    Args:
        preds: Array of converted sentences.
        targets: The actual sentences.
    """
    accs = []
    for p, t in zip(preds, targets):
        if len(p) >= len(t):
            acc_score = accuracy_score(p[:len(t)], t)
        else:
            acc_score = accuracy_score(p, t[:len(p)]) - 0.3

        accs.append(acc_score)
        print('Created:', p)
        print('Actual:', t)
        print('Accuracy score:', acc_score, '\n\n')

    print('Mean acc score:', np.mean(accs))


def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    """
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def write_pkl(path, data):
    """
    Writes the given data to .pkl file.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    """
    Loads data from given path to .pkl file.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data