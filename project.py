from neural2 import neuralNetwork
import skimage as ski
from skimage import data, io, filters, exposure
from skimage.filters import rank, threshold_minimum
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from numpy import array
import colorsys
from skimage import data
from matplotlib import pyplot as plt
import cv2
import sys
import math
from scipy import ndimage
from scipy import stats

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def resize(part):
        rows, cols = part.shape
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            part = cv2.resize(part, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            part = cv2.resize(part, (cols, rows))

        # part = cv2.resize(part, (20,20))
        # cols = 20
        # rows = 20
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        part = np.lib.pad(part,(rowsPadding,colsPadding),'constant')
        shiftx,shifty = getBestShift(part)
        shifted = shift(part,shiftx,shifty)
        gray = shifted
        return gray


def thresh(image, t):
    t = threshold_minimum(image)
    binary = (image > t) * 1.0
    return binary

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range (passnum):
            if alist[i][0]>alist[i+1][0]:
                temp = alist[i]
                alist[i]= alist[i+1]
                alist[i+1]=temp
    # return alist

def na_pinc(image_name):
    fig, ax = plt.subplots()

    im = cv2.imread('data_set/'+image_name+'.jpg')
    # print(im.shape)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imgray = 255 - imgray

    average = imgray.mean(axis=0).mean(axis=0)
    print(average)
    


    if (average > 130):
        imgray = adjust_gamma(imgray, 0.1)
    # else:
    #     if (average > 93):
    #         imgray = adjust_gamma(imgray,0.3)
    #         print("dupa")
    #     else:
    #         if (average > 90):
    #             imgray = adjust_gamma(imgray, 0.4)

    # ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            # cv2.THRESH_BINARY,7,-9)
    # imgray = cv2.medianBlur(imgray,3, 0)

    # thresh = imgray
    # print(kernel)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # thresh = cv2.dilate(thresh, kernel, iterations = 13)
    
    # ax.imshow(imgray, cmap='Greys_r')
    # plt.show()
    # ret, thresh = cv2.threshold(imgray, 50, 255, cv2.THRESH_BINARY)

    ret, thresh = cv2.threshold(imgray, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.imshow(thresh)
    plt.show()

    ## usuwanie soli
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)


    ## powiekszanie ksztaltow dla latwiejszego rozpoznania
    thresh = cv2.dilate(thresh, kernel, iterations = 13)

    plt.imshow(thresh)
    plt.show()


    imFinal = thresh.copy()

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_len = len(contours)

    ## potrzebne do usuniecie najwiekszego konturu jezeli jest przesadnie duzy
    maxx = 0
    maxContour = [1] 
    for cnt in contours:
        if maxx < cv2.contourArea(cnt):
            maxx = cv2.contourArea(cnt)
            maxContour = cnt
    sumOfContours=0
    for cnt in contours:
        sumOfContours += cv2.contourArea(cnt)
    meanOfContours = sumOfContours / cont_len
    print("meanOfContours1 = " + str(meanOfContours) + "cont_len1 = " + str(cont_len))
    if (maxx>4*meanOfContours):
        contours.remove(maxContour)
        imFinal = cv2.fillPoly(img=imFinal,pts=[maxContour],color = (0,0,0))
        cv2.drawContours(im, [maxContour], 0, (255,0,0), 7)
        cont_len = len(contours)

    ## robimy teraz to samo dla malych teraz kiedy przesadnie duzy jest wywalony
    maxx = 0
    maxContour = [1]
    for cnt in contours:
        if maxx < cv2.contourArea(cnt):
            maxx = cv2.contourArea(cnt)
    for cnt in contours:
        if cv2.contourArea(cnt) < maxx/8:
            imFinal = cv2.fillPoly(img=imFinal,pts=[cnt],color = (0,0,0))
            cv2.drawContours(im, [cnt], 0, (0,255,0), 7)

    ## tak, powtorka z rozrywki  
    im2, contours, hierarchy = cv2.findContours(imFinal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_len = len(contours)
    print('pls ' + str(cont_len))
    ## znowu wyliczamy srednia z konturow ktore zostaly
    sumOfContours = 0
    for cnt in contours:
        sumOfContours += cv2.contourArea(cnt)
    meanOfContours = sumOfContours / cont_len

    # sumOfContours=0
    # i = 0
    # for cnt in contours:
    #     if (cv2.contourArea(cnt) < meanOfContours * 4.0 and cv2.contourArea(cnt) > meanOfContours / 2.0 ):
    #         sumOfContours += cv2.contourArea(cnt)
    #         i+=1
    # meanOfContours = sumOfContours / i
    # print("meanOfContours2 = " + str(meanOfContours) + "cont_len2 = " + str(cont_len))


    ## sortowanie wedlug kolejnosci wystapienia
    secondaryTable = []
    meanH = 0
    i = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > meanOfContours/3 and cv2.contourArea(cnt) < meanOfContours*4:
            x,y,w,h = cv2.boundingRect(cnt)
            meanH += h
            i += 1
            secondaryTable.append((x,y,w,h))
    meanH = meanH/i

    bubbleSort(secondaryTable)
    
    # ## wykrywanie znakow rowna sie
    # tempArr =[]
    # DISTANCE_FOR_EQUALS = 100
    # for i in range(len(secondaryTable)-1):
    #     # tutaj sprawdzanie tylko z jednej strony bo sa juz posortowane
    #     if secondaryTable[i][0] + DISTANCE_FOR_EQUALS > secondaryTable[i+1][0]:
    #         x,y,w,h = secondaryTable[i]
            
    #         newX = x
    #         if (secondaryTable[i][1] < secondaryTable[i+1][1]):
    #             newY = y
    #             newH = secondaryTable[i+1][1] - secondaryTable[i][1] + secondaryTable[i+1][3]
    #         else:
    #             newY = secondaryTable[i+1][1]
    #             newH = secondaryTable[i][1] - secondaryTable[i+1][1] + secondaryTable[i][3]
    #         newW = secondaryTable[i+1][2] + secondaryTable[i+1][0] - secondaryTable[i][0]

    #         secondaryTable[i+1]=(newX,newY,newW,newH)
    #         tempArr.append(i)

    # print('dupadupadupa ' + str(len(tempArr)))
    # for i in range(len(tempArr)):
    #     secondaryTable.remove(secondaryTable[tempArr[i]])
    #     for j in range(i,len(tempArr)-1,1):
    #         tempArr[j+1]-=1



    numbers = []
    operators = []

    for i in range(len(secondaryTable)):
        x,y,w,h = secondaryTable[i]
        print(x ,y ,w, h)

        part = imFinal[y:y+h,x:x+w].copy()

        cv2.rectangle(imFinal,(x,y),(x+w,y+h),(255,255,255),10)

        res = resize(part)
        res = (res/255.0 *0.99)+0.01
        plt.imshow(res, cmap='Greys_r')

        # if(h > meanH):
        #     plt.savefig('number'+str(i)+'.jpg')
        #     numbers.append(res)
        # else:
        #     plt.savefig('operator'+str(i)+'.jpg')
        #     operators.append(res)
        numbers.append(res)
        plt.show()

    ## TODO: wymyslic lepszy sposob dzielenia na operatory i numery
    print('numbers len ' + str(len(numbers)))
    print('operators len ' + str(len(operators)))

    # for i in range(cont_len):
    #     # k = i * (255/cont_len)
    #     h = i / cont_len
    #     s = 1
    #     v = 1
    #     gowno = np.asarray(colorsys.hsv_to_rgb(h,s,v)) * 255
    #     # print(gowno)
    #     if cv2.contourArea(contours[i]) > meanOfContours/6 and cv2.contourArea(contours[i]) < meanOfContours*4:
    #         cv2.drawContours(im, contours, i, gowno, 7)
    #         x,y,w,h = cv2.boundingRect(contours[i])
    #         # x,y,w,h = secondaryTable[i]
    #         print(x ,y ,w, h)

    #         part = imFinal[y:y+h,x:x+w].copy()

    #         cv2.rectangle(imFinal,(x,y),(x+w,y+h),(255,255,255),10)

    #         # height, width = part.shape[:2]
    #         res = resize(part)
    #         # res = cv2.resize(part,(28, 28), interpolation = cv2.INTER_CUBIC)
    #         # res = (((255.0-res)/255.0)*0.99)+0.01
    #         # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    #         # res = cv2.dilate(res,kernel,iterations = 1)
    #         res = (res/255.0 *0.99)+0.01
    #         # res = res.flatten()/255.0
    #         plt.imshow(res, cmap='Greys_r')
    #         plt.savefig('dupa'+str(i)+'.jpg')
    #         # print((255-res)/255)
    #         numbers.append(res)
    #         plt.show()

    # plt.subplot(2,1,1)
    # plt.imshow(im)
    # plt.imshow(imgray, cmap='Greys_r')
    plt.subplot(2,1,1)
    plt.imshow(thresh, cmap='Greys_r')
    plt.subplot(2,1,2)
    plt.imshow(imFinal, cmap='Greys_r')
    fig.savefig(image_name+'.pdf')
    plt.show()
    return numbers

# def process(img):


def load(path):
    data_file = open(path, "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list

if __name__ == "__main__":
    images = na_pinc(sys.argv[1])
    images = np.asarray(images)

    input_nodes = 784
    hidden_nodes = 300
    hidden2_nodes = 300
    output_nodes = 10
    learning_rate = 0.2
    epochs = 1

    n = neuralNetwork(input_nodes,hidden_nodes,hidden2_nodes,output_nodes,learning_rate,epochs)
  
    n.deserialize()

    test_images = []
    for item in images:

        pom = item.reshape((784,))
        # pom = np.flip(pom,0)

        # plt.imshow(item, cmap='Greys_r')
        # plt.show()
        test_images.append(pom)
        # print(item.shape)

    results = []
    for item in test_images:
        # print(item)
        output = np.argmax(n.query(item))
        # output = n.query(item)
        results.append(output)

    print(results)


