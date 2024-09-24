#utils
import numpy as np
import torch
from PIL import Image
from tensorflow.keras.models import load_model

from input_module import front_card_path,back_card_path
from auth_utils import crop_center, pred_block, resizeImage
from grading_utils import initialize_model,device
from comments_utils import auth_comment
#load both image
front_card_img = Image.open(front_card_path)
back_card_img = Image.open(back_card_path)

def Load_model():
    #loading authentication models
    eng_model = load_model("/content/drive/MyDrive/pkm-models/eng_detect.h5")
    jap_model = load_model("/content/drive/MyDrive/pkm-models/jap_model.h5")
    japv_model = load_model("/content/drive/MyDrive/pkm-models/japv_model.h5")

    #loading grading models
    Load_dict = True
    front_grade_model , _ = initialize_model(model_name='resnet',num_classes=10)
    back_grade_model , _ = initialize_model(model_name='resnet',num_classes=10)
    if Load_dict:
        front_model_path = 'front_grade_model.pth'
        back_model_path = 'back_grade_model.pth'
        front_grade_model.load_state_dict(torch.load(front_model_path))
        back_grade_model.load_state_dict(torch.load(back_model_path))

    front_grade_model.eval()
    back_grade_model.eval()

    front_grade_model = front_grade_model.to(device)
    back_grade_model = back_grade_model.to(device)

    return eng_model, jap_model, japv_model, front_grade_model, back_grade_model

def sub_menu():
    print("\nSelect a category:")
    print("1. English Cards")
    print("2. Japanese Cards")
    print("3. Japanese Vintage Cards")
    category = input("Enter your choice (1-3): ")
    if category == "1":
        print("Selected English Cards")
        model_choice = "English"
    elif category == "2":
        print("Selected Japanese Cards")
        model_choice = "Japanese Modern"
    elif category == "3":
        print("Selected Japanese Vintage Cards")
        model_choice == "Japanese Vintage"
    else:
        print("Invalid category. Please choose again.")
        sub_menu()
    return model_choice

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1. Authenticate Card and Select Category")
        print("2. Grade Cards")
        print("3. Authenticate Card and Get Review")
        print("4. Ximilar Card Grading")
        print("5. Ximilar Card Description")
        print("6. Ximilar Card Score for Surface, Corners, and Edges")
        print("0. Do All")
        print("q. Quit")

        choice = input("Enter your choice: ")

        if choice == "1":
            model_choice = sub_menu()
            authenticate_card(back_card_img=back_card_img, model_choice=model_choice)
        elif choice == "2":
            grade_cards()
        elif choice == "3":
            model_choice = sub_menu()
            authenticate_card_with_review(back_card_img=back_card_img, model_choice=model_choice)
        elif choice == "4":
            ximilar_grading()
        elif choice == "5":
            ximilar_description()
        elif choice == "6":
            ximilar_score()
        elif choice == "0":
            do_all()
        elif choice.lower() == "q":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select again.")

def authenticate_card(back_card_img,model_choice):
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
    if selection == "English":
      if predictions == [1]:
          return f"Card is {card} and This card is likely genuine. confidence: {confi}%"
      else:
          return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%"

    else:
      if predictions == [1]:
        return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%"
      else:
        return f"Card is {card} and This card is likely genuine. confidence: {confi}%"

def authenticate_card_with_review(back_card_img,model_choice):
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
    comments , names = auth_comment(preds=predictions,choice=selection)
    proper_comments = f"{names[0][1]}\n{names[0][0]} : {comments[0]}\n\n{names[1][1]}\n{names[1][0]} : {comments[1]}"

    if selection == "English":
      if predictions == [1]:
          return f"Card is {card} and This card is likely genuine. confidence: {confi}%" , proper_comments
      else:
          return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%" , proper_comments

    else:
      if predictions == [1]:
        return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%" , proper_comments
      else:
        return f"Card is {card} and This card is likely genuine. confidence: {confi}%" , proper_comments

def grade_cards():
    jj=0