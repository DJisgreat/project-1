from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2
import glob
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tqdm import tqdm


'''获取文件的个数'''
def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

IM_WIDTH, IM_HEIGHT = 227, 227  
train_dir = 'G:/冬季学期/ML/X光影像肺炎分类/Data/新建文件夹/train'  # 训练集数据
val_dir = 'G:/冬季学期/ML/X光影像肺炎分类/Data/新建文件夹/val'  # 验证集数据
test_dir = 'G:/冬季学期/ML/X光影像肺炎分类/Data/新建文件夹/test'
nb_epoch = 30
batch_size = 20
nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_classes = 2  # 分类数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
nb_test_samples = get_nb_files(test_dir)

def image_preprocess():
    #   图片生成器
    # 　训练集的图片生成器，通过参数的设置进行数据扩增
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True
    )
    #   验证集的图片生成器，不进行数据扩增，只进行数据预处理
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    # 训练数据与测试数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, 
        class_mode='categorical'
    )
    return train_generator, validation_generator, test_generator

if __name__ == "__main__":
    model = Sequential()  
    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(BatchNormalization())

    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(BatchNormalization())
   
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))   
    model.add(BatchNormalization())

    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))   
    model.add(BatchNormalization())
  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(Dropout(0.25))
    
    model.add(Flatten())  
    model.add(Dense(4096,activation='relu')) 
    model.add(BatchNormalization()) 
    model.add(Dropout(0.5))
  
    model.add(Dense(4096,activation='relu')) 
    model.add(BatchNormalization()) 
    model.add(Dropout(0.5))

    model.add(Dense(2,activation='softmax'))  
    model.compile(optimizer=SGD(lr=0.002, momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])  
    model.summary()  

    train_generator,validation_generator,test_generator=image_preprocess()
    history= model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples,
            epochs=nb_epoch,
            validation_data=validation_generator,
            validation_steps=nb_val_samples
            )
    print(history.history)
    model.save('duibi.h5_1')

    print(sum(history.history['accuracy'])/len(history.history['accuracy']))
    print(sum(history.history['loss'])/len(history.history['loss']))
    
    print(sum(history.history['val_accuracy'])/len(history.history['val_accuracy']))
    print(sum(history.history['val_loss'])/len(history.history['val_loss']))

    #acc
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('acc_duibi_1.jpg')
    plt.cla()

    #loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('loss_duibi_1.jpg')
    plt.cla() 
    
    # model=tf.keras.models.load_model('duibi_3.h5')
    t,labels=[],[]
        
    for subdir in ['NORMAL/', 'PNEUMONIA/']:
        for file in tqdm(os.listdir('test/'+subdir)):
            img=cv2.resize(cv2.imread('test/'+subdir+file), (227,227))/255
            x=image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pred = model.predict(x)
            t.append(pred[0][0]-pred[0][1])
            if subdir == 'NORMAL/':
                labels.append(0)
            else:
                labels.append(1)

    t_max=max(t)
    t_min=min(t)
    print(t_max)
    print(t_min)
    Sp,Sn,FPR,TPR=[],[],[],[]
    for i in range(1001):
        tt=t_min+(t_max-t_min)*i/1000
        TP,FP,TN,FN=0,0,0,0
        for j in range(len(t)):
            if t[j]>tt and labels[j]==0:
                TP+=1
            elif t[j]>tt and labels[j]==1:
                FP+=1
            elif t[j]<tt and labels[j]==0:
                FN+=1
            elif t[j]<tt and labels[j]==1:
                TN+=1
        FPR.append((FP)/(TN+FP+0.0001))
        TPR.append((TP)/(TP+FN+0.0001))
        Sp.append((TN)/(TN+FP+0.0001))
        Sn.append((TP)/(TP+FN+0.0001))
    # plt.plot(Sp, Sn, linewidth=2.0, c = 'blue',label='ROC')
    print('Sp_0-12:%f'%(sum(Sp)/len(Sp)))
    print('Sn_0-12:%f'%(sum(Sn)/len(Sn)))
    plt.plot(FPR, TPR, linewidth=2.0, c = 'blue',label='ROC')
    plt.plot(FPR,FPR,linewidth=1.0,c='red')
    plt.title('ROC')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig('ROC_duibi_1.jpg')
    plt.show()

    # -0.99978757
    # -0.9998009
    # Sp_0-12:0.555965
    # Sn_0-12:0.562945

    