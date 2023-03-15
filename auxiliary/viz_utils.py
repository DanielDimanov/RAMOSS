from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf


def display_segmentation(display_list, save_path=False):
    # save_path is the name of the file the results should be saved at
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    if(save_path):
        plt.savefig('{}.png'.format(save_path))
    else:
        plt.show()
        

def create_mask(pred_mask):
    return pred_mask[0]


def show_predictions_segmentation(model=None,dataset=None, num=10, save_path=False):
    # save is the name of the file the results should be saved at
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        if(save_path):
            display_segmentation([image[0], create_mask(mask) , create_mask(pred_mask)], save_path)
        else:
            display_segmentation([image[0], create_mask(mask) , create_mask(pred_mask)])
        

def display_classification(display_list,label):
    plt.figure(figsize=(15, 15))

    title = ['Input Image']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.text(10, 10, label[i], bbox=dict(fill=False, edgecolor='red', linewidth=2))
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()