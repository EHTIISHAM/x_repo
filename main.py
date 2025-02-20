from utils import Load_model, main_menu 

#loading models
eng_model, jap_model, japv_model, front_grade_model, back_grade_model = Load_model()

if __name__ == "__main__":
    
    main_menu([eng_model,jap_model,japv_model,front_grade_model,back_grade_model])
