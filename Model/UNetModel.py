from Model.UNetBlocks import DownscaleBlock, UpscaleBlock, BottleNeckBlock
import tensorflow as tf
from tensorflow.keras import layers

from config import get_parameters

class AudioTrackSeparation(tf.keras.Model):
    def __init__(self):
        super().__init__()
        f = [8, 16, 32, 64, 128, 256]
        
        self.downscale_blocks = [
            DownscaleBlock(f[0]),
            DownscaleBlock(f[1]),
            DownscaleBlock(f[2]),
            DownscaleBlock(f[3]),
            DownscaleBlock(f[4]),
        ]

        self.bottle_neck_block = BottleNeckBlock(f[5])

        self.upscale_blocks = [
            UpscaleBlock(f[4]),
            UpscaleBlock(f[3]),
            UpscaleBlock(f[2]),
            UpscaleBlock(f[1]),
            UpscaleBlock(f[0]),
        ]

        self.output_layer_0 = layers.Conv1D(1, 1, 1, activation='tanh', 
                                            padding='valid', name='output_0')
        self.output_layer_1 = layers.Conv1D(1, 1, 1, activation='tanh', 
                                            padding='valid', name='output_1')

    @property
    def metrics(self):
        return [self.loss_metric_vocals, self.loss_metric_drums]
    
    def compile(self, 
                vocals_optimizer, 
                drums_optimizer,
                **kwargs):
        super().compile(**kwargs)
        self.vocals_optimizer = vocals_optimizer
        self.drums_optimizer = drums_optimizer

        self.loss_metric_vocals = tf.keras.metrics.Mean(name="vocal_loss")
        self.loss_metric_drums = tf.keras.metrics.Mean(name="drums_loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    def calculate_loss(self, target, pred):
        l1_loss = tf.reduce_mean(tf.abs(target - pred))
        mse_loss = tf.reduce_mean(tf.reduce_sum(self.mse(target, pred)))

        return l1_loss + mse_loss
        
    def train_step(self, batch_data):
        input, target = batch_data
        target0, target1 = target

        with tf.GradientTape() as tape_0, tf.GradientTape() as tape_1:
            pred0, pred1 = self(input, training=True)

            loss0 = self.calculate_loss(target0, pred0)
            loss1 = self.calculate_loss(target1, pred1)

        gradients_0 = tape_0.gradient(loss0, self.trainable_variables)
        self.vocals_optimizer.apply_gradients(zip(gradients_0, self.trainable_variables))
        self.loss_metric_vocals.update_state(loss0)

        gradients_1 = tape_1.gradient(loss1, self.trainable_variables)
        self.drums_optimizer.apply_gradients(zip(gradients_1, self.trainable_variables))
        self.loss_metric_drums.update_state(loss1)

        return {
            "vocal_loss": self.loss_metric_vocals.result(),
            "drums_loss": self.loss_metric_drums.result()
        }
    
    def test_step(self, batch_data):
        input, target = batch_data
        target0, target1 = target

        pred0, pred1 = self(input, training=False)

        loss0 = self.calculate_loss(target0, pred0)
        loss1 = self.calculate_loss(target1, pred1)
        
        self.loss_metric_vocals.update_state(loss0)
        self.loss_metric_drums.update_state(loss1)

        return {
            "vocal_loss": self.loss_metric_vocals.result(),
            "drums_loss": self.loss_metric_drums.result()
        }

    def build_graph(self):
        config = get_parameters()
        x = layers.Input(shape=(config.dim, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
        
    def call(self, x):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)
        c5, p5 = self.downscale_blocks[4](p4)

        bn = self.bottle_neck_block(p5)

        u1 = self.upscale_blocks[0](bn, c5)
        u2 = self.upscale_blocks[1](u1, c4)
        u3 = self.upscale_blocks[2](u2, c3)
        u4 = self.upscale_blocks[3](u3, c2)
        u5 = self.upscale_blocks[4](u4, c1)

        return self.output_layer_0(u5), self.output_layer_1(u5)