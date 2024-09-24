# this is OUR own model module
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch

from comments_utils import auth_comment
from utils import Load_model, main_menu 
from input_module import front_card_path,back_card_path
from auth_utils import crop_center, pred_block, resizeImage
#load both image
front_card_img = Image.open(front_card_path)
back_card_img = Image.open(back_card_path)

#loading authentication models
eng_model, jap_model, japv_model, front_grade_model, back_grade_model = Load_model()

if __name__ == "__main__":
    
    # Authentication

    back_image = crop_center(back_card_img)
    back_image = resizeImage(back_image)
    K_test = (back_image - back_image.mean()) / back_image.std()
    K_test = np.reshape(K_test, (1, 256, 256, 3))
    predictions, selection = pred_block(K_test, model_choice)
    confi = predictions[0][0]
    if selection == "Japanese Modern" or selection == "Japanese Vintage" :
        card = selection
    else:
        card = 'English'

    if confi <= 0.5:
        confi = 1- confi
    confi = round(confi*100, 2)
    predictions = list(map(lambda x: 0 if x<0.5 else 1, predictions)) # get binary values predictions with 0.5 as thresold
    print(predictions)
    comments , names = auth_comment(preds=predictions,choice=selection)

    proper_comments = f"{names[0][1]}\n{names[0][0]} : {comments[0]}\n\n{names[1][1]}\n{names[1][0]} : {comments[1]}"
