import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *

st.set_page_config(
    page_title="Smart Garbage Segregation",
    page_icon="♻",
    layout="wide"
)

labels = gen_labels()

# ------------------ CUSTOM CSS ------------------

st.markdown("""
<style>

.main-title{
font-size:45px;
font-weight:700;
color:white;
text-align:center;
}

.header{
background: linear-gradient(90deg,#00c9a7,#00695c);
padding:30px;
border-radius:12px;
margin-bottom:20px;
}

.result-box{
padding:25px;
border-radius:12px;
background:#e8f5e9;
border-left:8px solid #2e7d32;
font-size:22px;
font-weight:600;
}

.tip-box{
padding:20px;
border-radius:12px;
background:#fff3e0;
border-left:8px solid #fb8c00;
font-size:18px;
}

.sidebar-title{
font-size:24px;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------

st.markdown("""
<div class="header">
<p class="main-title">♻ Smart Garbage Segregation System</p>
</div>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Waste Classifier", "About Project"]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
**Waste Categories**

🧴 Plastic  
📄 Paper  
🥫 Metal  
🍾 Glass  
"""
)

# ------------------ HOME PAGE ------------------

if page == "Home":

    st.title("Welcome 👋")

    st.write(
    """
    This **AI powered system** helps to identify and classify waste materials.

    Upload an image and our deep learning model will automatically detect the waste type.
    """
    )

    st.image(
"img.jpg",
caption="Smart Waste Management",
width=800
)

# ------------------ CLASSIFIER PAGE ------------------

elif page == "Waste Classifier":

    st.header("Upload Waste Image")

    opt = st.selectbox(
        "Choose Upload Method",
        (
        'Upload image from device',
        'Upload image via link',
        'Take a live picture by camera'
        )
    )

    image = None

    # Device Upload
    if opt == 'Upload image from device':

        file = st.file_uploader('Upload Image', type=['jpg','png','jpeg'])

        if file is not None:
            image = Image.open(file)

    # URL Upload
    elif opt == 'Upload image via link':

        try:
            img = st.text_input('Enter Image URL')

            if img != "":
                image = Image.open(urllib.request.urlopen(img))

        except:
            st.error("Invalid image link")

    # Camera
    elif opt == 'Take a live picture by camera':

        camera_photo = st.camera_input("Take Photo")

        if camera_photo is not None:
            image = Image.open(camera_photo)

    # ------------------ PREDICTION ------------------

    try:

        if image is not None:

            st.image(image, width=300, caption="Uploaded Image")

            if st.button("Predict Waste Type"):

                with st.spinner("AI is analyzing the waste image..."):

                    time.sleep(2)

                    img = preprocess(image)

                    model = model_arc()
                    model.load_weights(
                        "C:/Users/Admin/Documents/ece final year project 2026 batch/Smart-Garbage-Segregation/weights/modelnew.h5"
                    )

                    prediction = model.predict(img[np.newaxis,...])

                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100

                    result = labels[predicted_class]

                    # ---------------- RESULT ----------------

                    st.markdown(
                    f"""
                    <div class="result-box">
                    ♻ Waste Type: <b>{result}</b><br>
                    📊 Confidence: {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

                    # ---------------- ICON ----------------

                    icons = {
                        "plastic":"🧴",
                        "paper":"📄",
                        
                        "metal":"🥫",
                        "glass":"🍾"
                    }

                    icon = icons.get(result.lower(),"♻")

                    st.subheader(f"{icon} Detected Waste Category")

                    # ---------------- RECYCLING TIPS ----------------

                    tips = {
                        "plastic":"Reuse plastic bottles and send them to recycling centers.",
                        "paper":"Paper can be recycled into new notebooks or packaging.",
                        "organic":"Organic waste can be composted to produce natural fertilizer.",
                        "metal":"Metal cans can be melted and reused in manufacturing.",
                        "glass":"Glass bottles should be cleaned and recycled separately."
                    }

                    tip = tips.get(result.lower(),"Dispose responsibly.")

                    st.markdown(
                    f"""
                    <div class="tip-box">
                    ♻ Recycling Tip:<br><br>
                    {tip}
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

    except Exception as e:
        st.error(e)

# ------------------ ABOUT PAGE ------------------

elif page == "About Project":

    st.header("About This Project")

    st.write(
    """
    **Smart Garbage Segregation System** uses deep learning to classify waste materials automatically.

    ### Technologies Used
    - Python
    - Streamlit
    - TensorFlow / Keras
    - Convolutional Neural Networks (CNN)

    ### Objective
    To improve waste management by automatically identifying waste categories and promoting recycling.
    """
    )

    #st.success("Developed as Final Year Engineering Project 🎓")