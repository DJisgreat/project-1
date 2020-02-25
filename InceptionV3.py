import os
import sys
import numpy as np
from tqdm import tqdm
from keras import regularizers
from keras.preprocessing import image
import cv2
import glob
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D ,Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''获取文件的个数'''
def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

IM_WIDTH, IM_HEIGHT = 299, 299  # InceptionV3指定的图片尺寸
FC_SIZE = 1024  # 全连接层的节点个数  
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量
train_dir = 'G:/冬季学期/ML/X光影像肺炎分类/Data/新建文件夹/train'  # 训练集数据
val_dir = 'G:/冬季学期/ML/X光影像肺炎分类/Data/新建文件夹/val'  # 验证集数据
test_dir = 'G:/冬季学期/ML/X光影像肺炎分类/Data/新建文件夹/test'
nb_epoch = 30
batch_size = 25
nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_classes = 2  # 分类数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
nb_test_samples = get_nb_files(test_dir)

def image_preprocess():
    #   图片生成器
    # 　训练集的图片生成器，通过参数的设置进行数据扩增
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    #   验证集的图片生成器，不进行数据扩增，只进行数据预处理
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')

    # 训练数据与测试数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical')

    return train_generator, validation_generator,test_generator

def add_new_last_layer(base_model, nb_classes):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(FC_SIZE, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(input=base_model.input, output=predictions)
        return model

def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_finetune(model):
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0002, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


if __name__ == "__main__":
    train_generator,validation_generator,test_generator=image_preprocess()
    base_model=InceptionV3(weights='imagenet', include_top=False)   # 使用带有预训练权重的InceptionV3模型，但不包括顶层分类器
    model = add_new_last_layer(base_model, nb_classes)   #添加顶层分类器
    model.summary()
    #训练顶层分类器
    # setup_to_transfer_learn(model, base_model)      
    # history_tl = model.fit_generator(
    #     train_generator,
    #     epochs=nb_epoch,
    #     steps_per_epoch=nb_train_samples//batch_size,
    #     validation_data=test_generator,
    #     validation_steps=nb_test_samples//batch_size
    # )

    '''对顶层分类器进行Fine-tune'''
    # Fine-tune以一个预训练好的网络为基础，在新的数据集上重新训练一小部分权重。fine-tune应该在很低的学习率下进行，通常使用SGD优化
    setup_to_finetune(model)
    history= model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_val_samples
    )
    print(history.history)
    model.save('InceptionV3_2.h5')
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
    plt.savefig('acc_2.jpg')
    plt.cla()

    #loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig('loss_2.jpg')
    plt.cla()
    
    # model=tf.keras.models.load_model('InceptionV3_1.h5')
    #test ROC
    t,labels=[],[]
    
    for subdir in ['NORMAL/', 'PNEUMONIA/']:
        for file in tqdm(os.listdir('test/'+subdir)):
            img=cv2.resize(cv2.imread('test/'+subdir+file), (299,299))/255
            x=image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pred = model.predict(x)
            # yy.append(pred)
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
        FPR.append((FP)/(TN+FP))
        TPR.append((TP)/(TP+FN))
        Sp.append((TN)/(TN+FP))
        Sn.append((TP)/(TP+FN))
    # plt.plot(Sp, Sn, linewidth=2.0, c = 'pink',label='ROC')
    print('Sp:%f'%(sum(Sp)/len(Sp)))
    print('Sn:%f'%(sum(Sn)/len(Sn)))
    plt.plot(FPR, TPR, linewidth=2.0, c = 'blue',label='ROC')
    plt.plot(FPR,FPR,linewidth=1.0,c='red')
    plt.title('ROC')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig('ROC_3.jpg')
    plt.show()

    # -0.8569218
    # -0.98438257
    # Sp:0.713415
    # Sn:0.341129

    
    # -0.5086602
    # -0.8899321
    # Sp:0.723891
    # Sn:0.356434
    
    
    
    
        
    
    

    
    
    
