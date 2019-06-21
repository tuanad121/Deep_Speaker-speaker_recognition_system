from pathlib import Path
from timeit import default_timer as timer

from python_speech_features import fbank
import numpy as np
from scipy.io.wavfile import read
import librosa

import constants as c
from utils import get_last_checkpoint_if_any
from models import convolutional_model
import silence_detector
import argparse


def VAD(audio):
    chunk_size = int(c.SAMPLE_RATE*0.05)  # 50ms
    index = 0
    sil_detector = silence_detector.SilenceDetector(15)
    nonsil_audio = []
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)


def read_audio(filename, sample_rate=c.SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = VAD(audio.flatten())
    start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    return audio


def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]


def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=c.SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)  #  filter_bank (num_frames , 64),energies (num_frames ,)
    #delta_1 = delta(filter_banks, N=1)
    #delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    #delta_1 = normalize_frames(delta_1)
    #delta_2 = normalize_frames(delta_2)

    #frames_features = np.hstack([filter_banks, delta_1, delta_2])    # (num_frames , 192)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)
    return np.reshape(np.array(frames_features),(num_frames, 64, 1))   #(num_frames,64, 1)


def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames + 20:
        bias = np.random.randint(20, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    elif x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x


def test_one_file():
    wav_path = 'audio/LibriSpeechSamples/train-clean-100/19/198/19-198-0000.wav'
    s = read_audio(filename=wav_path, sample_rate=c.SAMPLE_RATE)
    feature = extract_features(s, target_sample_rate=c.SAMPLE_RATE)
    feature = clipped_audio(feature)
    print(feature.shape)
    assert feature.shape[0] == 160


def test_predict_one_file():
    model = convolutional_model()
    wav_path = 'audio/LibriSpeechSamples/train-clean-100/19/198/19-198-0000.wav'
    _, s = read(wav_path)
    s = s / (2**15)
    feature = extract_features(s, target_sample_rate=c.SAMPLE_RATE)
    feature = clipped_audio(feature)
    last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        print(f"Found checkpoint {last_checkpoint}. Resume from here...")
        model.load_weights(last_checkpoint)
    feature = feature[np.newaxis, ...]
    print(feature.shape)
    emb_vector = model.predict(feature)
    print(emb_vector.shape)
    assert emb_vector.shape[1] == 512


def collect_data_by_speakers(data):
    data_by_speaker = {}
    for row in data:
        if row[3] not in data_by_speaker.keys():
            data_by_speaker[row[3]] = [row[0]]
        else:
            data_by_speaker[row[3]].append(row[0])
    for k in data_by_speaker.keys():
        assert len(data_by_speaker[k]) > 0
    return data_by_speaker


def embed_speakers_and_save(model, data_dict, output_path, number_samples=10):
    if not output_path.is_dir():
        output_path.mkdir()
    for spk in data_dict.keys():
        st = timer()
        choices = np.random.choice(data_dict[spk], size=number_samples, replace=True)
        print(f'Processing {spk}')
        emb_vectors = []
        for wav_path in choices:
            # preprocess wav_path
            if 1:  # modify paths when running on local Windows machine
                wav_path = wav_path.replace('/dat', '/Volumes')
            try:
                s = read_audio(filename=wav_path, sample_rate=c.SAMPLE_RATE)
            except Exception:
                print(f"no {wav_path}")
            feature = extract_features(s, target_sample_rate=c.SAMPLE_RATE)
            feature = clipped_audio(feature)
            if feature.shape[0] < 160:
                continue
            feature = feature[np.newaxis, ...]
            emb_vector = model.predict(feature)
            assert emb_vector.shape[1] == 512
            emb_vectors.append(emb_vector)
        emb_vectors = np.array(emb_vectors)
        spk_emb_vector = np.mean(emb_vector, axis=0)
        assert len(spk_emb_vector) == 512
        np.save(Path.joinpath(output_path, f'{spk}.npy'), spk_emb_vector)
        en = timer()
        print(f'finish after {en - st} s')


if __name__ == '__main__':
    # load pre-trained model
    parser = argparse.ArgumentParser(description='speaker embedding with pre-train model')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--csv_dir', type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    model = convolutional_model()
    last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        print(f'Found checkpoint [{last_checkpoint}]. Resume from here...')
        model.load_weights(last_checkpoint)

    csv_dir = Path(args.csv_dir)
    csv_list = list(csv_dir.glob('*.csv'))
    print(f'number of csv = {len(csv_list)}')
    for csv_path in csv_list:
        print(f'Process {csv_path}')
        data = np.genfromtxt(csv_path, delimiter='|', dtype=str)
        print(data.shape)
    
        data_dict = collect_data_by_speakers(data)
        embed_speakers_and_save(model, data_dict, output_path, number_samples=20)
        
    npy_list = list(output_path.glob('*.npy'))
    print(f'number of speakers = {len(npy_list)}')
