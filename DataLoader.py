import tensorflow as tf
import numpy as np
import librosa

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, sampling_rate, dim, batch_size=6, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = data.items
        self.sampling_rate = sampling_rate
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, idx):
        """Load input and target audio."""
        item = self.data[idx]

        input = librosa.resample(item['mix'].audio_data[0], 
                                 item['mix'].sample_rate, 
                                 self.sampling_rate)
        input = np.expand_dims(input, axis=1)

        vocals = librosa.resample(item['sources']['vocals'].audio_data[0], 
                                  item['mix'].sample_rate, 
                                  self.sampling_rate)
        vocals = np.expand_dims(vocals, axis=1)

        drums = librosa.resample(item['sources']['drums'].audio_data[0], 
                                 item['mix'].sample_rate, 
                                 self.sampling_rate)
        drums = np.expand_dims(drums, axis=1)
        

        return input, vocals, drums

    def data_generation(self, batch):

        x = np.empty((self.batch_size, self.dim, 1))
        y0 = np.empty((self.batch_size, self.dim, 1))
        y1 = np.empty((self.batch_size, self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y0[i,], y1[i,] = self.load(
                batch_id
            )

        return x, [y0, y1]