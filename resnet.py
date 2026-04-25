from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pylab as plt
import numpy as np


img_height,img_width=(224,224)
batch_size=12
train_data_dir="D:/Fathima/Python/cervical cancer/dataset/train"
test_data_dir="D:/Fathima/Python/cervical cancer/dataset/test"
print("====================================================")

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.4)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                target_size=(img_height,img_width),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                subset='validation')



test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                target_size=(img_height,img_width),
                                                batch_size=1,
                                                class_mode='categorical',
                                                subset='validation')
x,y=test_generator.next()
x.shape

base_model=ResNet50(include_top=False,weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(train_generator.num_classes,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable=False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,
          epochs=12,
        validation_data=test_generator)

        

model.save(r"D:\Fathima\Python\cervical cancer\resnet.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='validation accuracy',color='green')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(r"D:\Fathima\Python\cervical cancer\resnet.png")
plt.show()



vgg_acc=history.history['val_accuracy'][-1]
print(vgg_acc)