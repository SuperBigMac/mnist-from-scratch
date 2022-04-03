import numpy as np
import idx2numpy
import matplotlib.pyplot as plt


"""def showImg(filename, index):
    #imagefile = '/Users/bigmac/Desktop/resources/train-images-idx3-ubyte'
    #labelfile = '/Users/bigmac/Desktop/resources/train-labels-idx1-ubyte'
    imagearray = idx2numpy.convert_from_file(filename)
    plt.imshow(imagearray[index], cmap=plt.cm.binary)
    plt.show()

def array_from_file(filename):
    return idx2numpy.convert_from_file(filename)"""

def main():
    #image_array = array_from_file('/Users/bigmac/Desktop/resources/train-images-idx3-ubyte')
    #num_array = array_from_file('/Users/bigmac/Desktop/resources/train-labels-idx1-ubyte')

    #data = np.array([(image_array[i], num_array[i]) for i in range(len(image_array))], dtype=object)
    #np.save('training_data.npy', data)

    arr = np.load('training_data.npy', allow_pickle=True)

    for i in range(9):
        plt.subplot(330 + 1 + i)
        #plt.imshow(image_array[i], cmap=plt.get_cmap('gray'))
        plt.imshow(arr[i+10][0], cmap=plt.get_cmap('gray'))
        #print(num_array[i])
        print(arr[i+10][1])

    #print(image_array.shape)
    #print(num_array.shape)
    print(arr.shape)
    plt.show()
    print("Done")



if __name__ == '__main__':
    main()
