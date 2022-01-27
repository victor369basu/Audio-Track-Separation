from Model.UNetModel import AudioTrackSeparation

def test_model(config, test_input):
    '''
        Args:
           config: configuration variable to get model weight path.
           test_input: Processed array of provided audio data.
        Returns: Model prediction for processed input array.
        
    '''
    model = AudioTrackSeparation()
    model.load_weights(config.model_save_path+'model_weights')

    prediction = model.predict(test_input)

    return prediction
    