import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import base64
from shutil import copyfile
import shutil
import os


# load your model
model = tf.keras.models.load_model('models\cherry-picker-v1.h5')


def project_summary():
    st.title("Project Summary and Business Case")
    st.info(f"\nThis project was comissioned by the customer to help with the identification of sick leaves."
            f"\nThis project aims to automate the process for detecting if a cherry leaf is infected or not."
            f"\nBy completing this project, the customer will reduce cost associated with time and labot of checking the leaves and instead focus on maximizing yeilds")
    

    st.write(f"\nThe tangiable products to be delived by this product are:"
        f"\n - A dasboard with which to inspect and classify pictures going forward so that customers can continute to optimize their business\n"
        f"\n - Data analysis insights on how to visually tell if a sample is healthy or not by displaying the mean and varience of the customer provided samples\n"
        f"\n - A tool in which images can be uploaded and classified with an accuracy of 97% or higher\n")
    
    st.info(f"The objective of the business case have been met and are evidenced on the next pages of the dashboard.\n"
        f" - The customer data analysis shows the representations of how mean leaves (sick and healthy) look, in-addition to showing a sample of the leaves in a montage to aid with customer understanding and education\n"
        f" - The customer leaf tool uses a trained ML model to classify new leaf samples.The model was trained on 90% of the data and had a validation accurracy of 99%. The page explains this in further detail\n")


def data_analysis():
    st.title("Analysis of Customer Data")
    st.write(f"This page can be uses as an educaiton page by the customer for future employees to undertand the tool and build intuition on healthy leaves.\n")
    f"This page shows the presents the results of traditional data analysis conducted on the the customer leaves dataset.\n"
    f"This page shows the presents both a montage of images for both classes (infected and healthy) but also visualizations of mean and varience in the images also.\n"

    st.info(f"The data provided by the user was:.\n"
            f"- 2104 healthy images of 256x256 size\n"
            f"- 2104 unhealthy images of 256x256 size\n"
            f"The data was an even distribution of both classes, no image files provided were empty\n"
            f"\nAs the distribution between classes was even only preprocessing of images for analysis and splitting of data into train and validtion sets was completed\n"
            f"\nData used in the project can be found on kaggle: [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) \n")
    

    base_dir = './cherry-leaves/'
    train_dir = os.path.join(base_dir, 'training/')
    train_healthy_dir = os.path.join(train_dir, 'healthy')
    train_sick_dir = os.path.join(train_dir, 'sick')
    train_healthy_fnames = os.listdir( train_healthy_dir )
    train_sick_fnames = os.listdir( train_sick_dir )

    datagen = ImageDataGenerator(rescale=1./255)

    leaves_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=20,
        class_mode='binary')
    

    def plot_images():
        nrows = 4
        ncols = 4
        pic_index = 0

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))

        pic_index+=8

        next_healthy_pics = [os.path.join(train_healthy_dir, fname) 
                        for fname in train_healthy_fnames[ pic_index-8:pic_index] 
                    ]

        next_sick_pics = [os.path.join(train_sick_dir, fname) 
                        for fname in train_sick_fnames[ pic_index-8:pic_index]
                    ]

        for i, img_path in enumerate(next_healthy_pics+next_sick_pics):
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off')

            img = mpimg.imread(img_path)
            plt.imshow(img)

        st.pyplot(fig)


    if st.checkbox('Show healthy/un-healthy leaves'):
        plot_images()
        st.write('The figure shows the a montage of both healthy leaves (top two rows) and unhealthy leaves (top two rows)')


    st.info(f"When looking at the images there is a visable difference in the appearence of leaves.\n"
            f"from visual inspection on average it can be seen that:"
            f"\n- healthy leaves have a consistant shape\n" 
            f"\n- healthy leaves have less white spots\n" 
            f"\n- healthy leaves have a deeper green color\n" )
    

    def segregate_images(generator):
        sick_images = []
        healthy_images = []
    
        for images, labels in generator:
            for i in range(images.shape[0]):
                if labels[i] == 0:  # assuming '0' is the label for 'sick'
                    sick_images.append(images[i])
                else:  # assuming '1' is the label for 'healthy'
                    healthy_images.append(images[i])
                    
            # break the loop once all images are processed
            if generator.batch_index == 0:
                break

        return np.array(sick_images), np.array(healthy_images)

    def plot_images(avg_image, var_image, title):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(avg_image)
        ax[0].set_title(f"Average {title}")
        ax[0].axis("off")

        ax[1].imshow(var_image)
        ax[1].set_title(f"Variability {title}")
        ax[1].axis("off")

        return fig

    # Assuming leaves_gen is defined somewhere else in your code
    sick_images, healthy_images = segregate_images(leaves_gen)

    sick_avg = np.mean(sick_images, axis=0)
    sick_var = np.std(sick_images, axis=0)

    healthy_avg = np.mean(healthy_images, axis=0)
    healthy_var = np.std(healthy_images, axis=0)

    if st.checkbox('Show unnhealthy leaves mean & varience'):
        fig = plot_images(sick_avg, sick_var, "Sick Leaves")
        st.pyplot(fig)

    if st.checkbox('Show healthy leaves mean & varience'):
        fig = plot_images(healthy_avg, healthy_var, "Healthy Leaves")
        st.pyplot(fig)

    st.write(f"The figures above shows the average mean and variance of sick and healthy leaves\n"
            f"from this we can understand :\n" 
            f"\n- There is greater variation in sick leaves than healty\n"
            f"\n- on average healthy leaves are brighter than unhealty leaves\n " )

def ai_predict():
    st.title("Wellness Detector")

    st.info(f"This page fulfils business requirement of a prediction tool with 97% + accuracy for leaf images\n"
            f"on this page users:\n" 
            f"\n- Can upload one or more images of leaves\n"
            f"\n- Can get a prediction of classifcation of uploaded image\n")

    results = pd.DataFrame(columns=['File', 'Predicted Class'])
    uploaded_files = st.file_uploader("Choose images...", type=['png', 'jpg'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)

            img = img.resize((256, 256))

            # Convert the image to a numpy array
            img_array = img_to_array(img)

            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Use the model for prediction
            predictions = model.predict(img_array)
            predicted_class = "unhealthy" if  predictions > 0.6 else "healthy"
            new_row = pd.DataFrame({'File': [str(uploaded_file.name)], 'Predicted Class': [predicted_class]})
            results = pd.concat([results, new_row], ignore_index=True)

            # Display the prediction
            st.info(f'Predicted class: {predicted_class}')

    st.write("Results Table")
    st.table(results)
    st.info(f"Results from this session can be downloaded below\n" )  

    csv = results.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)


def model_explained():
    st.title("Technical Detail on The Solution Delivered")
    
    st.info(f"This page shows how the technical aspects of the delivered solution\n"
        f"on this page users:\n" 
        f"\n- Can view model architecture\n"
        f"\n- Can review model performance\n"
        f"\n- Can understand model training and testing methodology\n" )

    if st.checkbox('Show model architecture'):
        st.write(model.summary())

    if st.checkbox('Show evaluation'):
        st.write(model.summary())

pages = {
    "Project Summry": project_summary,
    "Customer Data Analysis": data_analysis,
    "Customer Leaf Tool": ai_predict,
    "Technical Breakdown":model_explained,
}


st.sidebar.title('Cherry Leaf Crop Detector')
selection = st.sidebar.radio("Go to", list(pages.keys()))

if selection == 'Upload Images':
    files, images = pages[selection]()
else:
    pages[selection]()

