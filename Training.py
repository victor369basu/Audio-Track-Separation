import tensorflow as tf

from Dataset import get_dataset
from DataLoader import DataGenerator
from Model.UNetModel import AudioTrackSeparation

def train(model_config):
    '''
        Args:
           model_config: Config variable.
        Returns: Trained model.
    '''
    musdb_train, musdb_validation = get_dataset()
    train_loader = DataGenerator(
        data=musdb_train,
        sampling_rate = model_config.sampling_rate,
        dim = model_config.dim,
        batch_size = model_config.batch,
    )

    validation_loader = DataGenerator(
        data=musdb_validation,
        sampling_rate=model_config.sampling_rate,
        dim = model_config.dim,
        batch_size = model_config.batch
        )

    vocals_optimizer = tf.keras.optimizers.Adam(
        learning_rate = model_config.learning_rate,
        amsgrad=False,
    )

    drums_optimizer = tf.keras.optimizers.Adam(
        learning_rate =  model_config.learning_rate,
        amsgrad=False,
    )
    model = AudioTrackSeparation()

    # Compile the model
    model.compile(vocals_optimizer, drums_optimizer)

    history = model.fit(
        train_loader,
        epochs =  model_config.epoch,
        validation_data=validation_loader,
    )
    return model