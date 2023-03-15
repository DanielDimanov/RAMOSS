# Taken and adapted from https://gist.github.com/Booritas/403c85de8eab95fa160044990765af31#file-image_history_callback-py

from tensorflow import keras
from PIL import Image
import io
import numpy as np
import tensorflow as t
class ImageHistory(keras.callbacks.Callback):
    ...
    def on_batch_end(self, batch, logs={}):
        if batch % self.draw_interval == 0:
            images = []
            labels = []
            for item in self.data:
                image_data = item[0]
                label_data = item[1]
                y_pred = self.model.predict(image_data)
                images.append(y_pred)
                labels.append(label_data)
            image_data = np.concatenate(images,axis=2)
            label_data = np.concatenate(labels,axis=2)
            data = np.concatenate((image_data,label_data), axis=1)
            self.last_step += 1
            self.saveToTensorBoard(data, 'batch',
               self.last_step*self.draw_interval)
        return

    def make_image(self, npyfile):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from 
        https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = npyfile.shape
        image = Image.frombytes('L',(width,height),
                               npyfile.tobytes())
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.compat.v1.Summary.Image(height=height,
                             width=width, colorspace=channel,
                             encoded_image_string=image_string)

    def saveToTensorBoard(self, npyfile, tag, epoch):
        data = npyfile[0,:,:,:]
        image = (((data - data.min()) * 255) / 
             (data.max() - data.min())).astype(np.uint8)
        image = self.make_image(image)
        summary = tf.compat.v1.Summary(
             value=[tf.compat.v1.Summary.Value(tag=tag,
                 image=image)])
        writer = tf.compat.v1.summary.FileWriter(
                 self.tensor_board_dir)
        writer.add_summary(summary, epoch)
        writer.close()