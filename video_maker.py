import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from LLE_Solver import *

def make_imgs(directory):
    """
    This function takes a directory that has a series of .pkl files and loads
    them, making a series of images that are saved in a subdirectory of to be
    used to make a video

    Parameters
    ----------
    directory : str
        Relative directory of the .pkl file locations.

    Returns
    -------
    None. (But saves images in directory)

    """
    #####################################
    # Make image directory (if required)
    outdir = directory + '/imgs/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    #####################################
    # Initialise figure/axes
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    #####################################
    # Run through files to make images
    no_files = len(os.listdir(directory))
    for idx, file in enumerate(os.listdir(directory)):
        print('\rMaking images: {:.2f}%'.format(100*idx/no_files),
              end="")
        if file.endswith('.pkl'):
            # Remove previous lines
            ax1.lines = []
            ax2.lines = []
            # Load LLE Soln and plot it in axes
            f = load_previous(directory+'/'+file)
            f.plot_self([ax1,ax2])
            #  Save figure
            fig.savefig(outdir + file.split('.')[0] + '.png')
    #####################################
    
def video_maker(directory, name=None):
    if name is None:
        name = 'animation.mp4'
    directory = directory + '/imgs/'
    no_imgs = len(os.listdir(directory))
    images = [img for img in os.listdir(directory)
              if img.endswith(".png")]

    frame = cv2.imread(directory+images[0])
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(directory+name, 0, 20, (width,height))
    
    for idx,image in enumerate(images):
        print('\rMaking images: {:.2f}%'.format(100*idx/len(images)),
              end="")
        video.write(cv2.imread(directory+image))
    cv2.destroyAllWindows()
    video.release()
        
    
if __name__ == '__main__':
    directory = '20210803_1641'
    make_imgs(directory='data/lle/' + directory)
    video_maker(directory = 'data/lle/' + directory)