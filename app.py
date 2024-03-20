import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import detect as detect


def main():
    st.header("Object Detection with YOLOv8",)
    confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.35)

    img_file_buffer_detect = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=0)
    DEMO_IMAGE = "demo.jpg"

    if img_file_buffer_detect is not None:
        img = cv.imdecode(np.fromstring(img_file_buffer_detect.read(), np.uint8), 1)
        image = np.array(Image.open(img_file_buffer_detect))
    else:
        img = cv.imread(DEMO_IMAGE)
        image = np.array(Image.open(DEMO_IMAGE))
    st.sidebar.text("Original Image")
    st.sidebar.image(image)
    detect.predict(img, confidence, st)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
        

