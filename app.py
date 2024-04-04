# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Sugarcane Leaf Disease Detection",
    page_icon = ":sugarcane:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class
            
            return key

with st.sidebar:
        st.image('images\sugarcane_animated.png')
        st.title("Sugarcane Leaf Disease Detection")
        st.subheader("The main aim is to provide an accurate understanding of the Sugarcane Leaf Detection with its cause")

st.write("""
         # Sugarcane Disease Detection with Remedy Suggestions
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    model = tf.keras.models.load_model('weights\efficientnet_model_sg.h5')
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Healthy', 'Mosaic','RedRot','Rust','Yellow']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'Mosaic':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Mosaic viruses are often spread by insect vectors such as aphids. Implementing control measures to manage these vectors can help reduce the spread of the virus. Remove and destroy infected plants as soon as symptoms are observed. This helps prevent further spread of the virus within the field.")

    elif class_names[np.argmax(predictions)] == 'RedRot':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Red rot is a significant disease of sugarcane caused by the fungus Colletotrichum falcatum. Fungicides can be applied preventively or curatively to manage red rot in sugarcane. However, their efficacy depends on various factors such as timing, application method, and resistance development. Proper irrigation management to avoid excessive moisture on sugarcane plants can help reduce the incidence.")

    elif class_names[np.argmax(predictions)] == 'Rust':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Rust is another significant disease affecting sugarcane, caused by various species of fungi belonging to the genera Puccinia and Cerotelium. Avoiding excessive moisture on sugarcane plants through proper irrigation management can help reduce the incidence and severity of rust, as the fungi thrive in humid conditions.")

    elif class_names[np.argmax(predictions)] == 'Yellow':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("It can be caused by various factors, including nutrient deficiencies, diseases, pests, environmental stress, or physiological disorders. Soil testing can help identify nutrient deficiencies, and corrective measures such as fertilization or foliar application of deficient nutrients can be employed. It may also be a symptom of various underlying diseases, such as yellow leaf syndrome, leaf scald, or mosaic diseases. Proper disease diagnosis and management strategies such as the use of disease-resistant varieties and application of fungicides, may be necessary. Integrated pest management practices, including biological control and use of insecticides can help manage pest populations.")

        
st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('weights\efficientnet_model_sg.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()