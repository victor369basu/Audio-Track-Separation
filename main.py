from Training import train
from Test import test_model
from config import get_parameters

import librosa
import librosa.display
import matplotlib.pyplot as plt

import numpy as np


def main(config):
    if config.train:
        '''
           Train the AudioTrackSeparation Model
        '''
        model = train(config)
        # saving model weights
        model.save_weights(config.model_save_path+'model_weights')

    else:
        '''
           Validate the AudioTrackSeparation Model
        '''
        y, sr = librosa.load(config.test, duration=6.80345, sr=config.sampling_rate)
        input_audio = np.expand_dims(y, axis=1)
        input_audio = np.expand_dims(input_audio, axis=0)
        

        pred = test_model(config, input_audio)

        title = ["Predicted Vocals", "Predicted Drums"]
        plt.figure(figsize=(15,9))
        for i in range(2):
            plt.subplot(2, 2, i + 1)
            plt.title(title[i])
            librosa.display.waveplot(pred[i][0].squeeze(), sr=config.sampling_rate)

        plt.savefig('foo.png')

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)