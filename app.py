import streamlit as st
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import sqlite3
from Patchcore import PatchCore
from GAN import GanInference
from AE import AEInference

DB_NAME = r'./database/app.db'
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

def load_image(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((96, 96))
        ])
        image = Image.open(image_path)
        # print(image.size)
        image = transform(image)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def call_model (model, module):
    if model == 'PatchCore':
        model = PatchCore(ckp_root=r'./pretrained_models', name_module=module, outdir='./output')
    if model == 'GAN':
        model = GanInference(ckp_root=r'./pretrained_models', name_module=module, outdir='./output')
        # pass
    if model == 'AutoEncoder':
        model = AEInference(ckp_root=r'./pretrained_models', name_module=module, outdir='./output')
    if model == 'Classification':
        model = AnomalyClassifierInference(ckp_root=r'./pretrained_models', name_module=module, outdir='./output')
        
    return model

def create_sliders(model):
    if model == 'GAN':
        st.slider("Threshold for Classification", 0.0, 1.0, 0.4, step=0.005, key='gan_thres_cls')
        st.slider("Threshold for Mask", 0.0, 1.0, 0.005,step=0.005, key='gan_thres_mask')
    if model == 'AutoEncoder':
        st.slider("Threshold for Reconstruction", 0.0, 1.0, 0.005,step=0.005, key='ae_thres_reconstruction')
        st.slider("Threshold for Mask", 0.0, 1.0, 0.01, step=0.005, key='ae_thres_mask')
    if model == 'Classification':
        st.slider("Threshold for Classification", 0.0, 1.0, 0.5, step=0.005, key='cls_thres')
        
def initialize_session():
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'inference_result' not in st.session_state:
        st.session_state.inference_result = None
    if 'gan_thres_cls' not in st.session_state:
        st.session_state.gan_thres_cls = 0.4
    if 'gan_thres_mask' not in st.session_state:
        st.session_state.gan_thres_mask = 0.005
    if 'ae_thres_reconstruction' not in st.session_state:
        st.session_state.ae_thres_reconstruction = 0.005
    if 'ae_thres_mask' not in st.session_state:
        st.session_state.ae_thres_mask = 0.01
    if 'cls_thres' not in st.session_state:
        st.session_state.cls_thres = 0.5
    if 'selected_module' not in st.session_state:
        st.session_state.selected_module = None
    
def main():
    st.title("Image Inference Application")
    initialize_session()

    # input_type = st.sidebar.radio(
    #     "Choose Image Input Method", 
    #     ["Upload Image", "Select from Dataset"]
    # )
    
    available_models = [
        # "Classification",
        "AutoEncoder",
        "GAN",
        "PatchCore",
    ]
    selected_model = st.sidebar.selectbox(
        "Select Inference Model", 
        available_models
    )
    st.session_state.selected_model = selected_model
   
    cursor.execute("SELECT DISTINCT data_module FROM images_test")
    query_res = cursor.fetchall()
    DATA_MODULES = [row[0] for row in query_res]
    selected_module = st.sidebar.selectbox(
        "Choose Dataset", 
        list(DATA_MODULES)
    )
    st.session_state.selected_module = selected_module
    cursor.execute("SELECT image_path, groundtruth_path, label FROM images_test WHERE data_module = ?", (selected_module,))
    test_images = cursor.fetchall()
    # print(test_images)
    test_paths = [row[0] for row in test_images]
    groundtruth_paths = [row[1] for row in test_images]
    labels = [row[2] for row in test_images]
    st.subheader(f"Images in {selected_module} Dataset")
    
    st.markdown("""
        <style>
        .stImage {
            
            overflow-y: auto;
        }
        .stImage > div > div {
            font-size: 10px !important;
        }
        .stButton > button > div > p {
            font-size: 12px !important; 
            padding: 5px 5px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    with st.container(height=500):
        cols = st.columns(5)
        
        for i, image_path in enumerate(test_paths):
            col = cols[i % 5]
            
            with col:
                image_name = (image_path.split('\\')[-2]).split('.')[0]+ '_' +image_path.split('\\')[-1]
                
                image = load_image(image_path)
                    
                if image:
                    st.image(image, caption=image_name, use_container_width=True)
                    
                    if st.button(f"Inference", key=f"inf_{image_name}", use_container_width = True):
                        image_info = {
                            "image_name": image_name,
                            "groundtruth_path": groundtruth_paths[i],
                            "label": labels[i],
                            "image_path": image_path
                        }
                        st.session_state.selected_image = image_info
                        if available_models:
                            print(selected_model, selected_module)
                            model = call_model(selected_model, selected_module)
                            if selected_model == 'GAN':
                                thres_cls = st.session_state.gan_thres_cls or 0.8
                                thres_mask = st.session_state.gan_thres_mask or 0.005
                                model.inference(image_info, thres_cls, thres_mask)
                            elif selected_model == 'AutoEncoder':
                                thres_reconstruction = st.session_state.ae_thres_reconstruction or 0.005
                                thres_mask = st.session_state.ae_thres_mask or 0.01
                                model.inference(image_info, thres_reconstruction, thres_mask)
                            elif selected_model == 'Classification':
                                thres_cls = st.session_state.cls_thres or 0.5
                                model.inference(image_info, thres_cls)
                            else:
                                model.inference(image_info)
                            if selected_model != 'Classification':
                                model.visualization(image_info['groundtruth_path'], model.result['prediction']['pred_masks'])
                            st.session_state.inference_result = model.result
    if selected_model:
        create_sliders(selected_model)
        if st.session_state.selected_image:
            if st.button("Re-run Inference with New Thresholds"):
                model = call_model(selected_model, st.session_state.selected_module)
                current_image_info = st.session_state.selected_image
                if selected_model == 'GAN':
                    thres_cls = st.session_state.get('gan_thres_cls', 0.5)
                    thres_mask = st.session_state.get('gan_thres_mask', 0.005)
                    model.inference(st.session_state.selected_image, thres_cls, thres_mask)
                    model.visualization(current_image_info['groundtruth_path'], model.result['prediction']['pred_masks'])
                elif selected_model == 'AutoEncoder':
                    thres_reconstruction = st.session_state.get('ae_thres_reconstruction', 0.005)
                    thres_mask = st.session_state.get('ae_thres_mask', 0.01)
                    model.inference(st.session_state.selected_image, thres_reconstruction, thres_mask)
                    model.visualization(current_image_info['groundtruth_path'], model.result['prediction']['pred_masks'])
                else:
                    model.inference(st.session_state.selected_image)
                    model.visualization(current_image_info['groundtruth_path'], model.result['prediction']['pred_masks'])
                
                st.session_state.inference_result = model.result
    if st.session_state.selected_image:
        st.subheader("Selected Image")
        st.markdown(st.session_state.selected_image['image_name'])
        if st.session_state.selected_image['label'] == 0:
            st.write("This is not an anomaly")
        else:
            st.write("This is an anomaly")
        
        st.image(st.session_state.selected_image['image_path'], width=300)
    st.subheader("Inference Result")
    if st.session_state.inference_result:
        # st.write(st.session_state.inference_result)
        if st.session_state.inference_result['prediction']['pred_labels'] == 1:
            st.write("This is an anomaly")
        elif st.session_state.inference_result['prediction']['pred_labels'] == 0:
            st.write("This is not an anomaly")
        if selected_model == 'GAN':
            st.write("Threshold for Classification: ", st.session_state.get('gan_thres_cls'))
            st.write("Threshold for Mask: ", st.session_state.get('gan_thres_mask'))
        if selected_model == 'AutoEncoder':
            st.write("Threshold for Reconstruction: ", st.session_state.get('ae_thres_reconstruction'))
            st.write("Threshold for Mask: ", st.session_state.get('ae_thres_mask'))
        if st.session_state.inference_result['fig_masks']:
            st.image(st.session_state.inference_result['fig_masks'])
        if st.session_state.inference_result['fig_roc_curve']:
            # with st.container(height=500):
            st.image(st.session_state.inference_result['fig_roc_curve'])
            
            
        
if __name__ == "__main__":
    main()