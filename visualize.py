import numpy as np
from PIL import Image
import os
import cv2 as cv
import argparse


def combine(path_orig, path_sal, output_path):
    # create the output directory if it doesn't exist
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # iterate over the original images
    for filename in os.listdir(os.path.join(path_orig, 'images')):
        # read the original image
        img_orig = cv.imread(os.path.join(path_orig, 'images', filename))

        # read the corresponding saliency map
        sal_path = os.path.join(path_sal, filename)
        if not os.path.exists(sal_path):
            continue
        img_sal = cv.imread(sal_path)

        # remove the blue and green channels from the saliency map
        img_sal[:, :, 0] = 0
        img_sal[:, :, 1] = 0

        # combine the original image with the saliency map
        out = cv.addWeighted(img_orig, 1.0, img_sal, 1.0, 0.0)

        # save the output image
        output_filename = filename.replace('.png', '_combined.png')
        output_pathname = os.path.join(output_path, output_filename)
        cv.imwrite(output_pathname, out)
    
def combine_orig(path_orig, path_sal, output_path):
    # create the output directory if it doesn't exist
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # iterate over the original images
    for filename in os.listdir(os.path.join(path_orig, 'images')):
        # read the original image
        img_orig = cv.imread(os.path.join(path_orig, 'images', filename))

        # read the corresponding saliency map
        sal_path = os.path.join(path_sal, filename)
        if not os.path.exists(sal_path):
            continue
        img_sal = cv.imread(sal_path)

        # remove the blue and green channels from the saliency map
        img_sal[:, :, 0] = 0
        img_sal[:, :, 1] = 0

        # combine the original image with the saliency map
        out = cv.addWeighted(img_orig, 1.0, img_sal, 1.0, 0.0)

        # save the output image
        output_filename = filename.replace('.png', '_combined.png')
        output_pathname = os.path.join(output_path, output_filename)
        cv.imwrite(output_pathname, out)

def create_animation(input_folder, output_filename, frame_duration=100):
    # Get a list of the image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Sort the image files alphabetically
    image_files.sort()
    
    # Open the first image to get the image size
    with Image.open(os.path.join(input_folder, image_files[0])) as img:
        width, height = img.size
    
    # Create a new list of Image objects from the image files
    images = [Image.open(os.path.join(input_folder, f)).resize((width, height)) for f in image_files]
    
    # Save the GIF animation
    images[0].save(output_filename, format='GIF', append_images=images[1:], save_all=True, duration=frame_duration, loop=0)

parser = argparse.ArgumentParser()
parser.add_argument('--pred_input', type=str)
parser.add_argument('--orig_input', type=str)
parser.add_argument('--output', type=str)

def main():
    args = parser.parse_args()
    output_orig = args.output + '/orig'
    output_pred = args.output + '/pred'
    output_orig_gif = args.output + '/orig.gif'
    output_pred_gif = args.output + '/pred.gif'
    pred_input = args.pred_input
    ground_truth_input = args.orig_input + '/maps'
    combine_orig(args.orig_input, ground_truth_input, output_orig)
    create_animation(output_orig, output_orig_gif)
    combine(args.orig_input, pred_input, output_pred)
    create_animation(output_pred, output_pred_gif)

if __name__ == '__main__':
    main()