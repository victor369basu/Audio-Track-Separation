from Model.UNetModel import AudioTrackSeparation

def test_model(config, test_input):
    model = AudioTrackSeparation()
    model.load_weights(config.model_save_path+'model_weights')

    prediction = model.predict(test_input)

    return prediction
    