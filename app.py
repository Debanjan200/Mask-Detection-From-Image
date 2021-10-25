from flask import Flask,render_template,request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import numpy as np

model=load_model("my_mask_detector.model")
app=Flask(__name__)

def model_predict(img_path):
    my_model=load_model("my_mask_detector.model")
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=preprocess_input(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=my_model.predict(test_image)
    print(np.argmax(result))

    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/",methods=["POST"])
def predict():
    if request.method=="POST":
        target_img=os.path.join(os.getcwd(),'uploads')
        f=request.files["my_img"]

        f.save(os.path.join(target_img,f.filename))
        img_path=os.path.join(target_img,f.filename)

        pred=model_predict(img_path)
        print(pred)
        st=""

        if np.argmax(pred)==0:
            st="Thankfully you wear a mask.Stay home,stay safe"
        
        else:
            st="Please wear a mask.It is very important.Stay home,stay safe"

        return render_template("index.html",st=st,prediction=np.argmax(pred))


if __name__=="__main__":
    app.run(debug=True)