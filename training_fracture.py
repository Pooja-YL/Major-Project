import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# load images to build and train the model
#                       ....                                     /    img1.jpg
#             test      Hand            patient0000   positive  --   img2.png
#           /                /                         \    .....
#   Dataset   -         Elbow  ------   patient0001
#           \ train               \         /                           img1.png
#                       Shoulder        patient0002     negative --      img2.jpg
#                       ....                   \
#

def load_path(path, part):
    """
    load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = path + '/' + str(folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                if body == part:
                    body_part = body
                    path_p = folder + '/' + str(body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = path_p + '/' + str(id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            path_l = path_id + '/' + str(lab)
                            for img in os.listdir(path_l):
                                img_path = path_l + '/' + str(img)
                                dataset.append(
                                    {
                                        'body_part': body_part,
                                        'patient_id': patient_id,
                                        'label': label,
                                        'image_path': img_path
                                    }
                                )
    return dataset


# this function get part and know what kind of part to train, save model and save plots
def trainPart(part):
    # categories = ['fractured', 'normal']
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = THIS_FOLDER + '/Dataset/'
    train_valid_data = load_path(image_dir + 'train_valid/', part)
    test_data = load_path(image_dir + 'test/', part)
    labels = []
    filepaths = []
    patient_ids = []

    # add labels for dataframe for each category 0-fractured, 1- normal
    for row in train_valid_data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])
        patient_ids.append(row['patient_id'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    patient_ids = pd.Series(patient_ids, name='Patient')

    images = pd.concat([filepaths, labels, patient_ids], axis=1)

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    train_indices, validation_indices = next(split.split(images, groups=images['Patient']))
    train_df = images.iloc[train_indices].drop(columns=['Patient'])
    validation_df = images.iloc[validation_indices].drop(columns=['Patient'])
    test_df = pd.DataFrame(
        {
            'Filepath': [row['image_path'] for row in test_data],
            'Label': [row['label'] for row in test_data],
        }
    )

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    )
    evaluation_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        classes=['fractured', 'normal'],
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=42,
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=validation_df,
        x_col='Filepath',
        y_col='Label',
        classes=['fractured', 'normal'],
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=64,
        shuffle=False,
        seed=42,
    )

    test_images = evaluation_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        classes=['fractured', 'normal'],
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # we use rgb 3 channels and 224x224 pixels images, use feature extracting , and average pooling
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    # for faster performance
    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)

    # outputs Dense '2' because of 2 classes, fratured and normal
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    # print(model.summary())
    print("-------Training " + part + "-------")

    # Adam optimizer with low learning rate for better accuracy
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # early stop when our model is over fit or vanishing gradient, with restore best values
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=25, callbacks=[callbacks])

    # save model to this path
    model.save(THIS_FOLDER + "/weights/ResNet50_" + part + "_frac.h5")
    results = model.evaluate(test_images, verbose=0)
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")

    # create plots for accuracy and save it
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, "./plots/FractureDetection/" + part + "/_Accuracy.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

    # create plots for loss and save it
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    figAcc = plt.gcf()
    my_file = os.path.join(THIS_FOLDER, "./plots/FractureDetection/" + part + "/_Loss.jpeg")
    figAcc.savefig(my_file)
    plt.clf()


# run the function and create model for each parts in the array
categories_parts = ["Elbow", "Hand", "Shoulder"]
for category in categories_parts:
    trainPart(category)
