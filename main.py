import numpy as np
from skimage.io import imread

from util.canvas import show_decompressed_images, show_compressed_images, show_entropy_graph
from util.consts import PATH_TO_FILE, START_EPSILON, END_EPSILON, TASK_5_EPSILONS, TASK_7_EPSILONS
from util.difcode import MyDifCode, MyDifDecode
from util.entropy import get_entropy
from initialization import q3_array, q2_array, q1_array, q4_array, img_decompress1, img_decompress2, img_decompress3, \
    img_decompress4, epsilons, entropy_array1, entropy_array2, entropy_array3, entropy_array4, max_epsilon1, \
    max_epsilon2, max_epsilon3, max_epsilon4

x = imread(PATH_TO_FILE)

for e in range(START_EPSILON, END_EPSILON, 1):
    q1 = MyDifCode(x, e, 1)
    q2 = MyDifCode(x, e, 2)
    q3 = MyDifCode(x, e, 3)
    q4 = MyDifCode(x, e, 4)
    if e in TASK_7_EPSILONS:  # show compressed images for epsilon = 0, 5, 10
        q1_array.append(q1)
        q2_array.append(q2)
        q3_array.append(q3)
        q4_array.append(q4)
    y1 = MyDifDecode(q1, e, 1)
    y2 = MyDifDecode(q2, e, 2)
    y3 = MyDifDecode(q3, e, 3)
    y4 = MyDifDecode(q4, e, 4)
    if e in TASK_5_EPSILONS:  # show decompressed images for epsilon = 5, 10, 20, 40
        img_decompress1.append(y1)
        img_decompress2.append(y2)
        img_decompress3.append(y3)
        img_decompress4.append(y4)

    epsilons.append(e)
    entropy_array1.append(get_entropy(q1))
    entropy_array2.append(get_entropy(q2))
    entropy_array3.append(get_entropy(q3))
    entropy_array4.append(get_entropy(q4))

    # ============= ERROR RATE TO CONSOLE ================= #
    max_epsilon1.append(np.max(x - y1))
    max_epsilon2.append(np.max(x - y2))
    max_epsilon3.append(np.max(x - y3))
    max_epsilon4.append(np.max(x - y4))
    print(e)  # show current state optional

for i in range(0, len(max_epsilon1), 1):
    print(f"e: {epsilons[i]}\tmax1: {max_epsilon1[i]}\t max2: {max_epsilon2[i]:.5f}\t max3: {max_epsilon3[i]:.5f}\t max4: {max_epsilon4[i]}")

show_decompressed_images([img_decompress1, img_decompress2, img_decompress3, img_decompress4])
show_compressed_images([q1_array, q2_array, q3_array, q4_array])

show_entropy_graph(epsilons, [entropy_array1, entropy_array2, entropy_array3, entropy_array4])
