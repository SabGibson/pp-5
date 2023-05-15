## Business Requirements
The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute.  The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.


* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.


## Hypothesis and how to validate?
* The unhealthy - powdery mildew leaves will be visually different from healthy leaves and themselves. This will be evidenced by creating varience plots of the leaf samples and comparing them to the healthy plot.
* The business challenge of prediction can be resolved by using a deep learning model. This can be validated by applying a deep learning model with low bias and varience and high accuracy > 97% on the validation set 

## The rationale to map the business requirements to the Data Visualisations and ML tasks
* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* * The customer need will need a mean and vairence plot of the different images to understand the ways in which the classes of leaves differ.
* * To achieve this we will need to get data, audit data provided to understand its structure and clean it so it may be used for analysis

* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
* * To complete this we will need to organize our files into trainig and validation sets, to ensure the model produced will be robust enough to function outside the test environemet.
* * To compile a deep learning model and polt the accuracy for both training and validation to ensure it meets business requirements, and loss to ensure there is no significant bias and the model is optimal for it sbusiness purpose.
* * Classes must also be balanced if not already

## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.
* The business will require the project to produce, a report of the difference between leaf classes including key metrics. namely mean and varience to illustrate to the customer the key differences and aid with teh development of intuition and future onboarding.
* The business wil require a robust model to predict the class of leaves presented to the model with high levels of confidence in the methodology. This means producing a model with 97% accuracy or greater so processes incoparating these models can be trusted and time cost saved by its implimetation in practice.

## Dashboard Design
* Project Summry:
* * This page summarises the project and key findings. It gives context to users so that they may understand what the tool is for and the contents of the applicaiton. it features no widgets and only highlighted information sections.
* Customer Data Analysis : This page will highlight learnings from the anlaysis. present users with the option to show a montage of healthy and unhealty leaves in and nxn grid where samples are randomly selected form training data. Also the ablility to see mean and variance plots. with options being aactionable with checkboxes 
* Customer Leaf Tool : This page will allow users to make predictions with the pretrained model with 97+% accuracy . upload one or more images form their local drive and preview classifcations in a table of results. Users will also be able to download the data in csv format from this page. 
* Technical Breakdown : This page will communicate to userws teh technical detail and methodology used to make the model. users will have the option vi check box to view train Vs validation accuracy and loss from model training and/or model architecture. 

## Unfixed Bugs
* Due to methodology used the some of the pages take a while to load. This is not a bug but a performance fix that could be resolved in future dates.

## Deployment
### Google Cloud Run

* The App live link is: [Launch App](https://cherry-ai-lite-hi34eli5da-ew.a.run.app) 

1. Log in to gcp and create a project
2. Using CLI build docker image 
3. Pushing built image to gcp cloud run
4. Naming service and deploying

* The model deployed is a ".tflite" optimized for mobile use to meet customer needs
* Heroku could not be used to deploy as model was still to large and some dependancies on some of the files were required.


## Main Data Analysis and Machine Learning Libraries
* Numpy - used for matrix calculations
* Pandas - used to manipulate dataframes and return csv 
* Tensorflow - used to develop ml model and preprocess and augment images for training 
* Matplotlib - was used as the main plotting software to display visualizations
* Keras - keras comes with tensorflow 2, but was used to abstract some of the more complex elements and implement tf functionality successfully
* OS - base lib in python but was used to collect data and organize file store to be compaitbale with other parts of mlpipline
* VS code - used vs code ipynb jupyter lab instance to develop ml files
* pipenv - used to manage virtual environment for dependancies 
## Credits 

* In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- [IBM convNets Video](https://www.youtube.com/watch?v=QzY57FaENXg) was used to understand convolutional neural networks and develop archictecture for model.

### Media

- The photos and figures are user generated
