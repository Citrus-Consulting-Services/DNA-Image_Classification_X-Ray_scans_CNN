#Importing the necessary libraries
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import keras
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
import joblib
import warnings 
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
%matplotlib inline


file_list=glob.glob("D:/Images/*.png") #Replace the path with the Image folder path
image_list=pd.DataFrame(columns=['path','file_name','label']) #creating a new dataframe to store the image attributes
for i in file_list:
    path=i
    file_name=i.split('\\')[1]
    label=file_name.split('_')[2].replace('.png',"")
    data=path,file_name,label
    image_list=image_list.append({'path':path,'file_name':file_name,'label':label},ignore_index=True)

#Checking the labels distribution
count = image_list['label'].value_counts(sort = True)
colors = ["grey","orange"] 
labels=['Normal','Tuberculosis']
#plotting pie chart
plt.pie(count,labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)
plt.title('Tuberculosis Percentage in data')
plt.show

#Data Split
independent=image_list.drop("label",axis=1)
dependent=image_list['label']
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=42)

train_label = keras.utils.to_categorical(y_train, 2)
test_labels = keras.utils.to_categorical(y_test, 2)


#creating the neural network from scratch

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(256, 256, 1), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X,
    train_label,
    batch_size=32,
    epochs=30,
    validation_data=(Test, test_labels),
    shuffle=True
)

model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure) # saving the model architecture
model.save_weights("D:/Citrus/X-Ray/model_weights.h5") # saving the model weights

#Real time predictions
def XRay(name):
    data1=[]
    img = image.load_img(name,target_size=(256,256))
    img = image.img_to_array(img)
    img = img/255
    X = np.array(img)
    X=X.astype('float32')
    Y=X.reshape(1,256,256,3)
    with graph.as_default():
        value = model.predict(Y)
    #print(value)
    if value[0][1]>0.5:
        probability=value[0][1]
        disease = "Patient has Tuberculosis"
        data=probability,disease
        data1.append(data)
    else:
        probability=value[0][1]
        disease = "Patient is Healthy"
        data=probability,disease
        data1.append(data)
    Image_details=pd.DataFrame(data1,columns=['Probability','Disease'])
    Image_details=Image_details.to_json(orient='records')
    return Image_details

#Predictions using Transfer Learning
images=[]
labels=[]
for index,row in X_train.iterrows():
    # Load the image from drive
    img = image.load_img(row[0],target_size=(256,256))
    # Convert the image to a numpy array
    image_array = image.img_to_array(img)
    # Add the image to the list of images
    images.append(image_array)
    #add corresponding image label
    label=y_train[index]
    labels.append(label)
x_train = np.array(images)
y_train = np.array(labels)
x_train = vgg16.preprocess_input(x_train) #to standardize the data

# Load a pre-trained neural network to use as a feature extractor
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3)) # top=False will remove the output dense layer
features_x = pretrained_nn.predict(x_train) # Extract features for each image (all in one pass)

# Create a model and add layers
model = Sequential()
model.add(Flatten(input_shape=features_x.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=15,
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")

# Load the json file that contains the model's structure
from keras.models import model_from_json
f = Path("model_structure.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("model_weights.h5")

##Bulk Predtions for new images using VGGNET
#create the array for input test images
test_images=[]
for index,row in X_test.iterrows():
    # Load the image from disk
    img = image.load_img(row[0],target_size=(256,256))

    # Convert the image to a numpy array
    image_array = image.img_to_array(img)

    # Add the image to the list of images
    test_images.append(image_array)

x_test_images = np.array(test_images)
x_test_1 = vgg16.preprocess_input(x_test_images)

# Extracting features from the input images
feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
features = feature_extraction_model.predict(x_test_1)

# Given the extracted features, make a final prediction using our own model
results = model.predict(features)


