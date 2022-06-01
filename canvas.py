from matplotlib import pyplot as plt
from skimage.io import imshow, show


def show_decompressed_images(d_images):
    fig = plt.figure(figsize=(20, 10))
    for i in range(0, len(d_images[0]), 1):
        epsilon = (2 ** i) * 5
        fig.add_subplot(4, 4, i + 1)
        plt.title(f"Decompressed image with e:{epsilon}")
        imshow(d_images[0][i], cmap='gray', vmin=0, vmax=255)
        fig.add_subplot(4, 4, i + 5)
        plt.title(f"Decompressed image with e:{epsilon}")
        imshow(d_images[1][i], cmap='gray', vmin=0, vmax=255)
        fig.add_subplot(4, 4, i + 9)
        plt.title(f"Decompressed image with e:{epsilon}")
        imshow(d_images[2][i], cmap='gray', vmin=0, vmax=255)
        fig.add_subplot(4, 4, i + 13)
        plt.title(f"Decompressed image with e:{epsilon}")
        imshow(d_images[3][i], cmap='gray', vmin=0, vmax=255)
    fig.savefig('Decompressed_images.jpg')
    show()


def show_compressed_images(c_images):
    fig = plt.figure(figsize=(20, 10))
    for i in range(0, len(c_images[0]), 1):
        epsilon = i * 5
        fig.add_subplot(4, 3, i + 1)
        plt.title(f"Compressed image with e:{epsilon}")
        imshow(c_images[0][i], cmap='gray')  # , vmin=0, vmax=255
        fig.add_subplot(4, 3, i + 4)
        plt.title(f"Compressed image with e:{epsilon}")
        imshow(c_images[1][i], cmap='gray')  # , vmin=0, vmax=255
        fig.add_subplot(4, 3, i + 7)
        plt.title(f"Compressed image with e:{epsilon}")
        imshow(c_images[2][i], cmap='gray')  # , vmin=0, vmax=255
        fig.add_subplot(4, 3, i + 10)
        plt.title(f"Compressed image with e:{epsilon}")
        imshow(c_images[3][i], cmap='gray')  # , vmin=0, vmax=255
    fig.savefig('Compressed_images.jpg')
    show()


def show_entropy_graph(epsilons, entropies):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 1, 1)
    plt.plot(epsilons, entropies[0], color='red', linestyle='-', linewidth=1)
    plt.plot(epsilons, entropies[1], color='blue', linestyle='-', linewidth=1)
    plt.plot(epsilons, entropies[2], color='green', linestyle='-', linewidth=1)
    plt.plot(epsilons, entropies[3], color='yellow', linestyle='-', linewidth=1)
    plt.legend(['predictor 1', 'predictor 2', 'predictor 3', 'predictor 4'])
    fig.savefig('Entropy_value_from_e.jpg')
    show()
