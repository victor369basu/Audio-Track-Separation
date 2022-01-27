# Audio-Track Separator

<img align="center" alt="architecture" src="./images/intro.png" style="background-color:white"/><br>

In this Repository, We developed an audio track separator in tensorflow that successfully separates Vocals and Drums from an input audio song track.

We trained a **U-Net** model with two output layers. One output layer predicts the **Vocals** and the other predicts the **Drums**. The number of Output layers could be increased based on the number of elements one needs to separate from input Audio Track.

# Technologies used:

1. The entire architecture is built with tensorflow. 
2. Matplotlib has been used for visualization. 
3. Numpy has been used for mathematical operations. 
4. Librosa have used for the processing of Audio files.
5. nussl for Dataset.

# Exploratory Data Analysis

<img align="center" alt="eda" src="./images/ex3.png" style="background-color:white"/><br>

<img align="center" alt="resample" src="./images/resample.png" style="background-color:white"/><br>

# Unet Architecture

``` python
model = AudioTrackSeparation()
model.build(input_shape=(None, DIM, 1))
model.build_graph().summary()
```
<img align="center" alt="summary" src="./images/summary1.png" style="background-color:white"/>
<img align="center" alt="summary" src="./images/summary2.png" style="background-color:white"/><br>

# Implementation
## Training
``` python
!python main.py --sampling_rate 11025 --train True --epoch 50 --batch 16 --model_save_path ./models/
```
Trains the u-net model on MUSDB18 Dataset and saves the trained model to the provided directory ( *--model_save_path* ).

## Testing
``` python
!python main.py --sampling_rate 11025 --test /content/pop.00000.wav --model_save_path ./models/
```
Loads the model from *model_save_path*, reads the audio file from the provided path( --test ) with librosa, process it and use the model to predict the output. In the end, the predictions are visualized by a wave plot and saved to the root directory.

<img align="center" alt="example1" src="./images/ex10.png" style="background-color:white"/><br>

<img align="center" alt="example2" src="./images/ex20.png" style="background-color:white"/>

# Model Performance

<img align="center" alt="vocal loss" src="./images/Vocal loss.png" style="background-color:white"/>

<img align="center" alt="drum loss" src="./images/Drum loss.png" style="background-color:white"/><br>

## Predictions

<img align="center" alt="Drums" src="./images/ex40.png" style="background-color:white"/><br>

<img align="center" alt="Drums" src="./images/ex30.png" style="background-color:white"/><br>

# References

1. <a href='https://arxiv.org/pdf/1806.03185v1.pdf'>Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation</a>

2. <a href='https://arxiv.org/pdf/1706.09588v1.pdf'>Multi-scale Multi-band DenseNets for Audio Source Separation</a>

3. <a href='https://arxiv.org/pdf/1811.11307v1.pdf'>Improved Speech Enhancement with the Wave-U-Net</a>