import streamlit as st
import pickle
from PIL import Image

#create a function
def main():
    #to add title (st means streamlit)
    st.title(':red[SPECIES PREDICTION]')
    #to read image
    image=Image.open(r"C:\Users\user\Desktop\deployiris\iris image.jfif")
    st.image(image,width=600)

    #identify the features

    #input features

    SepalLengthCm=st.text_input('SepalLengthCm','Type Here')
    SepalWidthCm=st.text_input('SepalWidthCm','Type Here') 
    PetalLengthCm=st.text_input('PetalLengthCm','Type Here') 
    PetalWidthCm=st.text_input('PetalWidthCm','Type Here') 
    

    #store all features in a variable
    f=[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]
    #load the stored model and scalar
    model1 = pickle.load(open(r"C:\Users\user\Desktop\deployiris\model_knn1.sav1", "rb"))
    scaler1 = pickle.load(open(r"C:\Users\user\Desktop\deployiris\scaler_knn1.sav1", "rb"))


    #to predict we add a button

    pred=st.button('PREDICT')


    #enable button
    prediction = None  # default value

    if pred:
        prediction=model1.predict(scaler1.transform([f]))#single squre bracket bcz features already in list format
        if prediction==0:
        #to print use write
            st.write('Iris-setosa')
            st.balloons()
        else:
            st.write('Iris-virginica')
            st.balloons()
main()
           







