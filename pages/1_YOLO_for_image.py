# ------------------------------------------------------------------------------
#  <copyright file="YOLO_for_image.py" company="DKGLabs Pvt Ltd">
#      Copyright (c) DKGLabs Pvt Ltd. All Rights Reserved.
#      Information Contained Herein is Proprietary and Confidential.
#  </copyright>
#  ------------------------------------------------------------------------------

import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
from utils import Utils
from utils import logo

st.set_page_config(
    page_title="YOLO Object Detection", layout="wide", page_icon="./images/object.png"
)

# st.header("Get Object Detection on Image using YOLOv5 Model: {modelName}".format(modelName = Utils.get_model_name()))
st.header("Get Object Detection on Image")
st.write("Please Upload an Image to get the detections")
logo()

with st.spinner("Please wait while your model is loading"):
    yolo = YOLO_Pred(onnx_model=Utils.get_model(), data_yaml=Utils.get_yaml())
    # st.balloons()


def upload_image():
    # Upload Image
    image_file = st.file_uploader(label="Upload Image")
    if image_file is not None:
        size_mb = image_file.size / (1024**2)
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": "{:,.2f} MB".format(size_mb),
        }
        # st.json(file_details)
        # validate file
        if file_details["filetype"] in ("image/png", "image/jpeg"):
            st.success("VALID IMAGE file type (png or jpeg")
            return {"file": image_file, "details": file_details}

        else:
            st.error("INVALID Image file type")
            st.error("Upload only png,jpg, jpeg")
            return None


def main():
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object["file"])

        col1, col2 = st.columns(2)

        with col1:
            st.info("Preview of Image")
            st.image(image_obj)

        button = st.button("Get Detection")
        if button:
            with st.spinner(
                """
                Geting Objets from image. please wait
                                """
            ):
                # below command will convert
                # obj to array
                image_array = np.array(image_obj)
                pred_img = yolo.predictions(image_array)
                pred_img_obj = Image.fromarray(pred_img)
                prediction = True
        with col2:
            if prediction:
                st.subheader("Predicted Image")
                st.caption("Object detection")
                st.image(pred_img_obj)

        st.subheader("Check below for file details")
        st.json(object["details"])


if __name__ == "__main__":
    main()
