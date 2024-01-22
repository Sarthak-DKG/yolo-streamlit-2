# ------------------------------------------------------------------------------
#  <copyright file="Home.py" company="DKGLabs Pvt Ltd">
#      Copyright (c) DKGLabs Pvt Ltd. All Rights Reserved.
#      Information Contained Herein is Proprietary and Confidential.
#  </copyright>
#  ------------------------------------------------------------------------------

import streamlit as st
import yaml
from utils import Utils
from utils import logo

print("PageName")
print(__name__)

st.set_page_config(page_title="Home", layout="wide", page_icon="./images/home.png")

st.title("YOLO Object Detection App")
st.caption("This web application demostrate Object Detection")
st.header("Which model would you like to use?")
logo()
folderList, fileDict = Utils.load_model_data("./models")
Utils.set_model("")
Utils.set_yaml("")

print("@".join(folderList))
model_selected = st.selectbox(
    "Which model would you like to use?",
    folderList,
    index=None,
    placeholder="Choose your model",
)

print(model_selected)
if model_selected != None:
    Utils.set_model(model_selected)
    Utils.set_yaml(model_selected)
    if model_selected[-2:] == "v5":
        st.markdown(
            """
            # - [Image Detection](/YOLO_for_image/)
            # - [Live Camera Detection](/YOLO_for_LiveCam/)   
                        """
        )
        st.write("This model helps in detecting using Yolo V5")
    else:
        st.markdown(
            '<a href="/YOLOv8_for_image" target="_self">Image Detection</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<a href="/YOLOv8_for_LiveCam" target="_self">LiveCam Detection</a>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<a href="/YOLOv8_for_Video" target="_self">Video Detection</a>',
            unsafe_allow_html=True,
        )

        # st.markdown(
        #     """
        #     # - <a href="/YOLOv8_for_image/" target="_self"> Image Detection </a>
        #     # - [Image Detection](/YOLOv8_for_image/)
        #     # - [Live Camera Detection](/YOLOv8_for_LiveCam/)
        #     # - [Video Detection](/YOLOv8_for_Video/)
        #                 """
        # )
        st.write("This model helps in detecting using Yolo V8")

    with open(Utils.get_yaml(), "r") as file1:
        yaml_file1_data = yaml.safe_load(file1)
        Utils.set_yaml_data_list(yaml_file1_data["names"])
        classList = st.multiselect(
            "Choose the objects for detection", options=yaml_file1_data["names"]
        )

    st.write("You selected : ")
    st.write(classList)
    Utils.set_object_list(classList)
