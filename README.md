Accelerate the mfcc computation by pyCUDA

## Dependency

- pycuda: https://pypi.python.org/pypi/pycuda
- scikit-cuda: ```pip install scikit-cuda```

## Example

```python
import pycuda.autoinit
import scipy.io.wavfile as wav
from pycuda_mfcc_featurizer import cuMfccFeaturizer

mfcc_featurizer = cuMfccFeaturizer()

(rate,sig) = wav.read("english.wav")
mfcc_feat = mfcc_featurizer.mfcc(sig,rate)
d_mfcc_feat = mfcc_featurizer.delta(mfcc_feat, 2)
dd_mfcc_feat = mfcc_featurizer.delta(d_mfcc_feat, 2)
fbank_feat = mfcc_featurizer.logfbank(sig,rate)

```
