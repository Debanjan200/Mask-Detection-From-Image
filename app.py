from flask import Flask,render_template,request
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np

model=load_model("image_mask_detector_model.model")
app=Flask(__name__)

def model_predict(img_path):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result[0][0]


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
        st=""

        if pred==0:
            st="Thankfully you wear a mask.Stay home,stay safe"
        
        elif pred==1:
            st="Please wear a mask.It is very important.Stay home,stay safe"

        return render_template("index.html",st=st,prediction=pred)


if __name__=="__main__":
    app.run(debug=True)