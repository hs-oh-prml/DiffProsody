import essentia
import essentia.standard as ess
import matplotlib.pyplot as plt
from dtw import dtw
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm
import librosa
import glob
import os
import pyworld

# https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_mfcc_the_htk_way.py
def extractor(audio, numberBands=26):  # mel capstral 추출
    # fs = 22050
    # audio = ess.MonoLoader(filename=filename,
    #                        sampleRate=fs)()
    # dynamic range expansion as done in HTK implementation
    audio = audio * 2 ** 15
    frameSize = 1024  # corresponds to htk default WINDOWSIZE = 250000.0
    hopSize = 256  # corresponds to htk default TARGETRATE = 100000.0
    fftSize = 1024
    spectrumSize = fftSize // 2 + 1
    zeroPadding = fftSize - frameSize

    w = ess.Windowing(type='hamming',  # corresponds to htk default  USEHAMMING = T
                      size=frameSize,
                      zeroPadding=zeroPadding,
                      normalized=False,
                      zeroPhase=False)

    spectrum = ess.Spectrum(size=fftSize)

    mfcc_htk = ess.MFCC(inputSize=spectrumSize,
                        type='magnitude',  # htk uses mel filterbank magniude
                        warpingFormula='htkMel',  # htk's mel warping formula
                        weighting='linear',  # computation of filter weights done in Hz domain
                        highFrequencyBound=8000,  # corresponds to htk default
                        lowFrequencyBound=0,  # corresponds to htk default
                        numberBands=numberBands,  # corresponds to htk default  NUMCHANS = 26
                        numberCoefficients=13,
                        normalize='unit_sum',  # htk filter normaliation to have constant height = 1
                        dctType=2,  # htk uses DCT type III
                        logType='log',
                        liftering=0)  # corresponds to htk default CEPLIFTER = 22

    mfccs = []
    # startFromZero = True, validFrameThresholdRatio = 1 : the way htk computes windows
    for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize, startFromZero=True,
                                    validFrameThresholdRatio=1):
        spect = spectrum(w(frame))
        mel_bands, mfcc_coeffs = mfcc_htk(spect)
        mfccs.append(mfcc_coeffs)

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    # mfccs = essentia.array(pool['MFCC']).T
    mfccs = essentia.array(mfccs).T[1:]

    # plt.imshow(mfccs[1:,:], aspect = 'auto', interpolation='none') # ignore enery
    # plt.xlabel('Frame', fontsize=14)
    # plt.ylabel('MCC', fontsize=14)
    # plt.imshow(mfccs, aspect = 'auto', interpolation='none')
    # plt.show() # unnecessary if you started "ipython --pylab"
    return (mfccs)

def get_pitchContour(wav, hop_size, sample_rate):

    frame_period = (hop_size / (0.001 * sample_rate))
    f0, timeaxis = pyworld.harvest(wav, sample_rate, frame_period=frame_period)

    return f0

def MCD(audio_one, audio_two, numberBands=13):  # distortion 계산
    # https://github.com/danijel3/PyHTK/blob/master/python-notebooks/HTKFeaturesExplained.ipynb
    # normalization
    mfcc_one = extractor(audio_one, numberBands) * np.sqrt(2 / numberBands)
    mfcc_two = extractor(audio_two, numberBands) * np.sqrt(2 / numberBands)

    if np.isnan(mfcc_one[0][-1]) or np.isinf(mfcc_one[0][-1]):
        mfcc_one = mfcc_one[:, :-1]
    if np.isnan(mfcc_two[0][-1]) or np.isinf(mfcc_two[0][-1]):
        mfcc_two = mfcc_two[:, :-1]

    dist, cost, acc_cost, path = dtw(mfcc_one.T, mfcc_two.T, dist=lambda x, y: norm(x - y, ord=1))

    dtw_one = mfcc_one.T[path[0]]
    dtw_two = mfcc_two.T[path[1]]

    mcd = 10 / np.log(10) * np.sqrt(2 * np.sum(((dtw_one - dtw_two) ** 2), axis=1))
    mcd = np.sum(mcd) / len(mcd)

    return mcd

def F0_RMSE(audio_one, audio_two, numberBands=26):  # distortion 계산
    hop_size = 256
    sample_rate = 22050
    
    p1 = get_pitchContour(audio_one.astype(np.float64), hop_size, sample_rate)
    p2 = get_pitchContour(audio_two.astype(np.float64), hop_size, sample_rate)

    p1 = np.nan_to_num(p1)
    p2 = np.nan_to_num(p2)

    # numpy.linalg.norm option: https://leebaro.tistory.com/entry/numpylinalgnorm
    p1 = np.expand_dims(p1, axis=0)
    p2 = np.expand_dims(p2, axis=0)
    dist, cost, acc_cost, path = dtw(p1.T, p2.T, dist=lambda x, y: norm(x - y, ord=1)) # 1->0, ord가 norm의 옵션임.

    dtw_one = p1.T[path[0]]
    dtw_two = p2.T[path[1]]

    # rmse 공식 확인! sqrt가 마지막이다!
    f0_rmse = np.sqrt(np.sum((dtw_one - dtw_two) ** 2) / dtw_one.shape[0])

    return f0_rmse

def DDUR(audio_one, audio_two, sample_rate=22050, rescaling_max=0.999):

    audio_one = audio_one / np.abs(audio_one).max() * rescaling_max
    audio_two = audio_two / np.abs(audio_two).max() * rescaling_max

    def cal_DDUR(audio):
        # computed the average absolute differences between the durations of the converted and target utterances
        intervals = librosa.effects.split(audio, top_db=20, frame_length=1024, hop_length=256) # [(s1, e1),..., (sn, en)]
        DDUR = np.sum([(e-s) for s, e in intervals], axis=0) / sample_rate # (단위: sample개수 -> s)
        return DDUR

    return np.mean(np.abs(cal_DDUR(audio_one) - cal_DDUR(audio_two)))
