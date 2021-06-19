import torch, sys
from transformers import HubertModel, HubertConfig
from transformers import Wav2Vec2Processor, HubertForCTC
import soundfile as sf
import numpy as np

def featurize_hubert(file, model, size):
    audio_input, _ = sf.read(file)
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    features=list(np.array(predicted_ids).flatten())
    labels=list()
    for i in range(len(features)):
        labels.append('hubert_%s'%(str(i)))
    
    
    if len(features) == size:
        pass
    elif len(features) > size:
        features=features[0:size]
    elif len(features) < size:
        # zero out values that were not there
        difference=len(features)-size
        for i in range(len(difference)):
            features.append(0)
            labels.append('hubert_%s'%(str(len(features)+i+1)))
    return features, labels

# features, labels= featurize_hubert(sys.argv[1], model, 100)
# print(dict(zip(labels, features)))
