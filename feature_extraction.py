import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import os

def hsv_features(image: Image):
    """
    Compute some hsv features: saturation mean, warm/cool color pixel fraction
    Args:
        image (PIL.Image): Image to process and extract features from
    Returns:
        tuple:
            - float: Fraction of warm colored pixels
            - float: Fraction of cool colored pixels
            - float: Saturation average
    """
    hsv_image = np.array(image.convert('HSV')) / 255
    warm_mask = (hsv_image[:, :, 0] >= 0) & (hsv_image[:, :, 0] <= 1/6) | \
                     (hsv_image[:, :, 0] >= 1/6) & (hsv_image[:, :, 0] <= 1/3)
        
    cool_mask = (hsv_image[:, :, 0] >= 0.5) & (hsv_image[:, :, 0] <= 0.67)
    
    warm_pixels = np.sum(warm_mask)
    cool_pixels = np.sum(cool_mask)
    
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]

    warm_fraction = warm_pixels / total_pixels
    cool_fraction = cool_pixels / total_pixels
    sat_mean = np.mean(hsv_image[:, :, 1])
    return warm_fraction, cool_fraction, sat_mean

def dominant_colors(image_np: np.array, n_colors: int = 10):
    """
    Extract dominant colors by clustering (kmeans)
    Args:
        image_np: numpy image array
    Returns:
        tuple:
            - np.array: cluster center colors sorted by cluster size
            - np.array: cluster pixel frequencies by cluster size
    """
    clustering = KMeans(n_clusters=n_colors * 2)
    pixels_table = image_np.reshape((-1,3))
    clustering.fit(pixels_table)
    clusters, counts = np.unique(clustering.labels_, return_counts = True)
    percentages = counts / pixels_table.shape[0]
    sorted_idx = np.argsort(percentages)
    sorted_colors = clustering.cluster_centers_[sorted_idx] / 255
    sorted_percentages = percentages[sorted_idx]
    return sorted_colors[:n_colors], sorted_percentages[:n_colors]

def convert_image(image: Image):
    """
    Convert pillow image to RGB. Returns image and boolean indicating if
    the image has color or is B/W - gray
    """
    color = 1
    if image.mode == 'L':
       image = image.convert('RGB')
       color = 0
    elif image.mode == 'LA':
        lum, alp = image.split()
        rgb_image = Image.merge("RGB", (lum, lum, lum))
        background = Image.new("RGB", rgb_image.size, (250, 250, 250))
        image = Image.composite(rgb_image, background, alp)
        color = 0
    elif image.mode in ['P', 'RGBA', 'CMYK']:
       image = image.convert('RGB')
    return image, color
    
def get_features(image: Image):
    """
    Produce classic features of an image
    """
    width, height = image.size
    aspect_ratio = width / height

    image, color = convert_image(image)

    ####
    image_np = np.array(image)
    red_mean, green_mean, blue_mean = image_np.mean(axis=(0,1)) / 255
    red_std, green_std, blue_std = image_np.std(axis=(0,1))
    #####
    image_gray = np.array(image.convert('L'))
    brightness = image_gray.mean() / 255
    contrast = image_gray.std()
    entropy = image.entropy()
    ###
    warm_fraction, cool_fraction, sat_mean = hsv_features(image)
    sorted_colors, sorted_percentages = dominant_colors(image_np)

    features = {
        'aspect_ratio': aspect_ratio,
         'color': color,
        'read_mean': red_mean,
        'green_mean': green_mean,
        'blue_mean': blue_mean,
        'red_std': red_std,
        'green_std': green_std,
        'blue_std': blue_std,
        'brightness': brightness,
        'contrast': contrast,
        'entropy': entropy,
        'warm_fraction': warm_fraction,
        'cool_fraction': cool_fraction,
        'saturation_mean': sat_mean,
    }
    for ic in range(len(sorted_colors)):
        for ib, b in enumerate(['red', 'green', 'blue']): 
            features[f'{b}_dom_color{ic}'] = sorted_colors[ic][ib]
        features[f'weight_dom_color{ic}'] = sorted_percentages[ic]
    return features

def process_images(images_path: str, features_file_path: str) -> pd.DataFrame:
    """
    Process images from a folder and save features.
    This assumes a local paths. It could be replaced by any cloud storage (or database for the features)
    with proper interface connector.
    It also assumes we are processing all the images in the folder path. For individual processing just few
    adjustments would be needed.
    Arguments:
        images_path (str): Folder path were the images are contained
        features_file_path (str): Full file path of the csv file where to save the features
    """
    images_list = [f for f in os.listdir(images_path)]
    map_images_paths = {int(image_name.split('.')[0]): images_path +'/' + image_name for image_name in images_list}
    list_features = []
    
    for image_id, image_path in map_images_paths.items():
        print(f'features image {image_id}')
        image = Image.open(image_path)
        list_features.append(get_features(image))
    
    creative_features = pd.DataFrame(index=map_images_paths.keys(), data=list_features)
    creative_features.index.name = 'creative_id'
    creative_features.to_csv(features_file_path)

