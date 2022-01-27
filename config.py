import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # re-sampling audio
    parser.add_argument('--sampling_rate', dest='sampling_rate', default=11025, type=int,
                        help="sampling rate of the audio for Re-Sampling.")
    parser.add_argument('--dim', dest='dim', default=75008, type=int,
                        help="Length of Audio array after re-sampling.")

    # Misc
    parser.add_argument('-t','--train', type=str2bool, default=False,help="True when train the model, else used for testing.")
    parser.add_argument('-v','--test', default=False, type=str, help="Path to the audio file for testing through model prediction.")

    # Model hyperparameter
    parser.add_argument('-b', '--batch', dest='batch',type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-e', '--epoch', type=int, default=50)

    # Base Directory
    parser.add_argument('-m', '--model_save_path', type=str, default='./models/',
                        help="Path to Saved model directory.")
    
    return parser.parse_args()