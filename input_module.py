#input
from PIL import Image
import cv2
def load_images(front_card_path, back_card_path):

    try:
        #load both image
        front_card_img = cv2.imread(front_card_path)
        back_card_img = cv2.imread(back_card_path)
        return front_card_img, back_card_img
    except Exception as e:
        front_card_img = None
        back_card_img = None
        print(e)
        print()
        print("Image Loading Failed")
        return None, None

import os
import requests
from urllib.parse import urlsplit

def download_image(url, folder_path):
    try:
        # Extract the image name and extension from the URL
        filename = os.path.basename(urlsplit(url).path)
        filename, ext = os.path.splitext(filename)
        # Create the full file path
        file_path = folder_path+ext

        # Download the image
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded: {file_path}")
            return file_path
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def delete_image(filepath):
    # Delete the file if it exists
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Image deleted: {filepath}")
    else:
        print(f"File {filepath} does not exist.")
