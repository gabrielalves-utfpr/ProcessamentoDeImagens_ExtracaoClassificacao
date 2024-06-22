import os
import cv2
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
import time
from skimage.feature import local_binary_pattern

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainFeaturePath = './features_labels/lbp/train/'  # Alterado para LBP
    testFeaturePath = './features_labels/lbp/test/'    # Alterado para LBP
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractLBPFeatures(trainImages)  # Alterado para LBP
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractLBPFeatures(testImages)  # Alterado para LBP
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            if len(filenames) > 0:  # Está dentro de uma pasta com arquivos
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Obtendo imagens e rótulos de {folder_name}', max=len(filenames),
                          suffix='%(index)d/%(max)d Duração:%(elapsed)ds')
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels, dtype=object)

def extractLBPFeatures(images):
    bar = Bar('[INFO] Extraindo características LBP...', max=len(images), suffix='%(index)d/%(max)d  Duração:%(elapsed)ds')
    featuresList = []
    for image in images:
        if len(image.shape) > 2:  # Imagem colorida
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp_image = local_binary_pattern(image, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 10))
        featuresList.append(hist)
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)

# Restante do código permanece igual

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels,dtype=object), encoder.classes_

def saveData(path,labels,features,encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    #the name of the arrays will be used as filenames
    #f'{labels=}' gets both variable name and its corresponding values.
    #split('=')[0] gets the variable name from f'{labels=}'
    label_filename = f'{labels=}'.split('=')[0]+'.csv'
    feature_filename = f'{features=}'.split('=')[0]+'.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0]+'.csv'
    np.savetxt(path+label_filename,labels, delimiter=',',fmt='%i')
    np.savetxt(path+feature_filename,features, delimiter=',') #float does not need format
    np.savetxt(path+encoder_filename,encoderClasses, delimiter=',',fmt='%s') 
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Saving done in {elapsedTime}s')


if __name__ == "__main__":
    main()
