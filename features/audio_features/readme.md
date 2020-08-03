### Audio
looking into actively
* [surfboard](https://github.com/novoic/surfboard) - GPL
* [sonopy](https://github.com/MycroftAI/sonopy) - MFCCs fastest featurizer (Mycroft)
* [Pysptk](https://github.com/r9y9/pysptk) - Tokoyo based lab
* [pysepm](https://github.com/schmiph2/pysepm) - speech quality measures
* [pystoi](https://github.com/mpariente/pystoi) - speech intelligibility measure 
* [pb_bss](https://github.com/fgnt/pb_bss/blob/master/examples/mixture_model_example.ipynb) - blind source separation (training models)
* [Kaldi](https://pykaldi.github.io/api/kaldi.feat.html#)
* [speechpy](https://github.com/astorfi/speechpy)
* [sigfeat](https://github.com/SiggiGue/sigfeat)
* [kapre-kears](https://github.com/keunwoochoi/kapre)
* [torchaudio-contrib](https://github.com/keunwoochoi/torchaudio-contrib)
* [spafe](https://github.com/SuperKogito/spafe) - many features
* [Essentia](https://github.com/kushagrasurana/Essentia-feature-extraction)
* [Wav2Letter](https://github.com/facebookresearch/wav2letter/wiki/Python-bindings) 
* [Formant extraction](https://github.com/danilobellini/audiolazy/blob/master/examples/formants.py)
* [speaker diarization speakers]() - assumes long-form audio files

tried but not user friendly
* [Gammatone](https://github.com/detly/gammatone) - spectrograms of fixed lengths
* [Shennong](https://github.com/bootphon/shennong) - using kaldi / post-processing
* Ludwig audio features - add them in.
* [pyspk](https://nbviewer.jupyter.org/github/r9y9/pysptk/blob/master/examples/pysptk%20introduction.ipynb) - Fundamental frequency estimation using pyspk (bottom)
* [auDeep](https://github.com/auDeep/auDeep)
* fix myprosody_features.py feature script (in helpers for now). This is buggy and may change into the future as the library is more supported by more senior developers.
* allow Ludwig model type to dictate featurization - 'audio_ludwig_features'
* [Wavelet transforms](http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/) - could be useful for dataset augmentation techniques.
* [fft python](https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files)
* [kaldi features](https://github.com/pykaldi/pykaldi)  - GMM and other such features. https://pykaldi.github.io/api/kaldi.feat.html#module-kaldi.feat.fbank
* [CountNet](https://github.com/faroit/CountNet) - number of speakers in a mixture (5 second interval). Combine with WebRTC VAD (https://github.com/wiseman/py-webrtcvad) to get featurization per segment like average lengths, etc. 
* [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
* [resin](https://github.com/kylerbrown/resin) - copyleft license
