"""Compute mfcc base on pycuda"""

from __future__ import division
import decimal

import numpy as np
import math
import logging
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cu_math
import pycuda.compiler as compiler
import skcuda.fft as cu_fft
import skcuda.cublas as cublas
from kernels import *

from scipy.fftpack import dct

class cuMfccFeaturizer(object):

    def __init__(self,
                 nfilt=26,
                 nfft=512,
                 samplerate=16000,
                 lowfreq=0,
                 highfreq=None,
                 batch=500):
        mod = compiler.SourceModule(matrix_square_kernel)
        self._matrix_square = mod.get_function("MatrixSquareKernel")
        # get filter banks
        self._nfilt = nfilt
        self._nfft = nfft
        self._samplerate = samplerate
        self._cublas_handle = cublas.cublasCreate()
        self.get_filterbanks(nfilt, nfft, samplerate,lowfreq, highfreq)
        # cufft plan
        self.plan(nfft, batch)
        # warm

    def plan(self, nfft, batch):
        self._nfft = nfft
        self._batch = batch
        self._mag_spec_gpu = gpuarray.empty((batch, nfft//2+1), np.complex64)
        self._power_spec_gpu = gpuarray.empty((batch, nfft//2+1), np.float32)
        self._plan = cu_fft.Plan(nfft, np.float32, np.complex64, batch)

    def mfcc(self,signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
             nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,
             appendEnergy=True,
             winfunc=lambda x:np.ones((x,))):
        """Compute MFCC features from an audio signal.

        :param signal: the audio signal from which to compute features.
                       Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds.
                      Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds.
                     Default is 0.01s (10 milliseconds)
        :param numcep: the number of cepstrum to return, default 13
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters.
                 In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient.
                0 is no filter. Default is 0.97.
        :param ceplifter: apply a lifter to final cepstral coefficients.
                0 is no lifter. Default is 22.
        :param appendEnergy: if this is true, the zeroth cepstral coefficient
               is replaced with the log of the total frame energy.
        :param winfunc: the analysis window to apply to each frame.
               By default no window is applied.
               You can use np window functions here e.g. winfunc=np.hamming
        :returns: A np array of size (NUMFRAMES by numcep) containing features.
              Each row holds 1 feature vector.
        """
        feat,energy = self.fbank(signal,samplerate,winlen,winstep,nfilt,nfft,
                          lowfreq,highfreq,preemph,winfunc)
        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
        feat = self.lifter(feat,ceplifter)
        if appendEnergy: feat[:,0] = np.log(energy)
            # replace first cepstral coefficient with log of frame energy
        return feat

    def fbank(self,signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
              winfunc=lambda x:np.ones((x,))):
        """Compute Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features.
                Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds.
               Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds.
               Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters.
              In Hz, default is 0.
        :param highfreq: highest band edge of mel filters.
               In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient.
            0 is no filter. Default is 0.97.
        :param winfunc: the analysis window to apply to each frame.
             By default no window is applied. You can use np window functions here e.g. winfunc=np.hamming
        :returns: 2 values. The first is a np array of size
               (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
            second return value is the energy in each frame (total energy, unwindowed)
        """
        highfreq= highfreq or samplerate/2
        # if the samplerate of some sample is changed,
        # recalculate the filterbanks
        if self._samplerate != samplerate:
            self.get_filterbanks(nfilt, nfft, samplerate,lowfreq, highfreq)
        signal = self.preemphasis(signal,preemph)
        frames = self.framesig(signal, winlen*samplerate,
                           winstep*samplerate,
                        winfunc)
        pspec_gpu = self.powspec(frames,nfft)
        pspec = pspec_gpu.get()
        energy = np.sum(pspec,1) # this stores the total energy in each frame
        energy = np.where(energy == 0,np.finfo(float).eps,energy)
        # if energy is zero, we get problems with log
        #fb = self.get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
        #feat = np.dot(pspec,fb.T) # compute the filterbank energies
        #pspec_gpu = gpuarray.to_gpu(pspec)
        m, k = np.shape(pspec)
        #print ("pspec = ", pspec)
        n = self._nfilt
        feat_gpu = gpuarray.empty((m,n), np.float32)
        cublas.cublasSgemm(self._cublas_handle, 't', 'n', n, m,
                          k, np.float32(1.0), self._fb_gpu.gpudata,
                          k, pspec_gpu.gpudata, k,
                          np.float32(0.0), feat_gpu.gpudata, n)

        feat = feat_gpu.get()
        feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log

        return feat,energy

    def logfbank(self,signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97):
        """Compute log Mel-filterbank energy features from an audio signal.

        :param signal: the audio signal from which to compute features. Should be an N*1 array
        :param samplerate: the samplerate of the signal we are working with.
        :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
        :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
        :param nfilt: the number of filters in the filterbank, default 26.
        :param nfft: the FFT size. Default is 512.
        :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
        :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
        :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
        :returns: A np array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
        """
        feat,energy = self.fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph)
        return np.log(feat)

    def hz2mel(self,hz):
        """Convert a value in Hertz to Mels

        :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1+hz/700.)

    def mel2hz(self,mel):
        """Convert a value in Mels to Hertz

        :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700*(10**(mel/2595.0)-1)

    def get_filterbanks(self,nfilt=20,nfft=512,samplerate=16000,lowfreq=0,highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A np array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq= highfreq or samplerate/2
        assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = self.hz2mel(lowfreq)
        highmel = self.hz2mel(highfreq)
        melpoints = np.linspace(lowmel,highmel,nfilt+2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft+1)*self.mel2hz(melpoints)/samplerate)

        fbank = np.zeros([nfilt,nfft//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        # neccesary datatype conversion to np.float32 and write to GPU
        self._fb_gpu = gpuarray.to_gpu(np.asarray(fbank, np.float32))


    def lifter(self,cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
        magnitude of the high frequency DCT coeffs.

        :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
        """
        if L > 0:
            nframes,ncoeff = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L/2.)*np.sin(np.pi*n/L)
            return lift*cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra

    def delta(self,feat, N):
        """Compute delta features from a feature vector sequence.

        :param feat: A np array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        :param N: For each frame, calculate delta features based on preceding and following N frames
        :returns: A np array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
        """
        if N < 1:
            raise ValueError('N must be an integer >= 1')
        NUMFRAMES = len(feat)
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        delta_feat = np.empty_like(feat)
        padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
        for t in range(NUMFRAMES):
            delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        return delta_feat


    def round_half_up(self,number):
        return int(decimal.Decimal(number).quantize(decimal.Decimal('1'),
                  rounding=decimal.ROUND_HALF_UP))

    def framesig(self,sig,frame_len,frame_step,winfunc=lambda x:np.ones((x,))):
        """Frame a signal into overlapping frames.

        :param sig: the audio signal to frame.
        :param frame_len: length of each frame measured in samples.
        :param frame_step: number of samples after the start of the previous
                          frame that the next frame should begin.
        :param winfunc: the analysis window to apply to each frame.
                         By default no window is applied.
        :returns: an array of frames. Size is NUMFRAMES by frame_len.
        """
        slen = len(sig)
        frame_len = int(self.round_half_up(frame_len))
        frame_step = int(self.round_half_up(frame_step))
        if slen <= frame_len:
            numframes = 1
        else:
            numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))

        padlen = int((numframes-1)*frame_step + frame_len)

        zeros = np.zeros((padlen - slen,))
        padsignal = np.concatenate((sig,zeros))

        indices = np.tile(np.arange(0,frame_len),(numframes,1)) \
                  + np.tile(np.arange(0,numframes*frame_step,
                      frame_step),(frame_len,1)).T
        indices = np.array(indices,dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len),(numframes,1))
        return frames*win


    def deframesig(self,frames,siglen,frame_len,frame_step,
                winfunc=lambda x:np.ones((x,))):
        """Does overlap-add procedure to undo the action of framesig.

        :param frames: the array of frames.
        :param siglen: the length of the desired signal, use 0 if unknown.
                       Output will be truncated to siglen samples.
        :param frame_len: length of each frame measured in samples.
        :param frame_step: number of samples after the start of the previous
                       frame that the next frame should begin.
        :param winfunc: the analysis window to apply to each frame.
                       By default no window is applied.
        :returns: a 1-D signal.
        """
        frame_len = round_half_up(frame_len)
        frame_step = round_half_up(frame_step)
        numframes = np.shape(frames)[0]
        assert np.shape(frames)[1] == frame_len, \
              '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

        indices = np.tile(np.arange(0,frame_len),(numframes,1)) \
                 + np.tile(np.arange(0,numframes*frame_step,frame_step)
                         ,(frame_len,1)).T
        indices = np.array(indices,dtype=np.int32)
        padlen = (numframes-1)*frame_step + frame_len

        if siglen <= 0: siglen = padlen

        rec_signal = np.zeros((padlen,))
        window_correction = np.zeros((padlen,))
        win = winfunc(frame_len)

        for i in range(0,numframes):
            window_correction[indices[i,:]] = window_correction[indices[i,:]] \
                        + win + 1e-15 #add a little bit so it is never zero
            rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]

        rec_signal = rec_signal/window_correction
        return rec_signal[0:siglen]

    def powspec(self,frames,NFFT):
        """Compute the power spectrum of each frame in frames.
          If frames is an NxD matrix, output will be Nx(NFFT/2+1).

        :param frames: the array of frames. Each row is a frame.
        :param NFFT: the FFT length to use. If NFFT > frame_len,
                 the frames are zero-padded.
        :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1).
               Each row will be the power spectrum of the corresponding frame.
        """
        frames = frames.astype(np.float32)
        num_frames, len_frame = np.shape(frames)
        #print  num_frames, NFFT
        if len_frame > NFFT:
            logging.warn('frame length (%d) is greater than FFT size (%d),'
                'frame will be truncated. Increase NFFT to avoid.', len_frame, NFFT)
            frames = frames[0: -1, 0: NFFT]

        if len_frame < NFFT:
            # pad zero after each row
            npad = ((0, 0), (0, NFFT-len_frame))
            frames =  np.pad(frames, pad_width=npad, mode='constant', constant_values=0)

        if num_frames > self._batch:
            self.plan(NFFT, num_frames)


        """
        if num_frames < self._batch:
            padding = np.asarray(np.random.random((self._batch-num_frames, NFFT)), np.float32)
            frames = np.vstack((frames, padding))
        """
        frames_gpu = gpuarray.to_gpu(frames)
        cu_fft.fft(frames_gpu, self._mag_spec_gpu, self._plan)
        mag_spec_gpu = self._mag_spec_gpu[0: num_frames]
        # compute the power spectrum
        power_spec_gpu = self._power_spec_gpu[0: num_frames]
        self._matrix_square(mag_spec_gpu,
                            power_spec_gpu,
                            np.float32(1.0/NFFT),
                            np.int32(num_frames),
                            np.int32(NFFT//2+1),
                            block = (32, 16, 1),
                            grid = ((num_frames-1)//32 +1, (NFFT//2)//16 + 1))
        return power_spec_gpu

    def preemphasis(self, signal,coeff=0.95):
        """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient.
                   0 is no filter, default is 0.95.
        :returns: the filtered signal.
        """
        return np.append(signal[0],signal[1:]-coeff*signal[:-1])



