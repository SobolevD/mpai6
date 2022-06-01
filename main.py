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

q_arrays = [q1_array, q2_array, q3_array, q4_array]
dec_images = [img_decompress1, img_decompress2, img_decompress3, img_decompress4]
entropies = [entropy_array1, entropy_array2, entropy_array3, entropy_array4]
max_epsilons = [max_epsilon1, max_epsilon2, max_epsilon3, max_epsilon4]

for e in range(START_EPSILON, END_EPSILON, 1):
    epsilons.append(e)
    for r in range(1, 4, 1):
        q = MyDifCode(x, e, r)

        if e in TASK_7_EPSILONS: # show compressed images for epsilon = 0, 5, 10
            q_arrays[r-1].append(q)

        y = MyDifDecode(q, e, r)

        if e in TASK_5_EPSILONS:  # show decompressed images for epsilon = 5, 10, 20, 40
            dec_images[r-1].append(y)

        entropies[r-1].append(get_entropy(q))

        # ============= ERROR RATE TO CONSOLE ================= #
        max_epsilons[r-1].append(np.max(x - y))

        print(e)  # show current state optional

for i in range(0, len(max_epsilon1), 1):
    print(f"e: {epsilons[i]}\tmax1: {max_epsilon1[i]}\t max2: {max_epsilon2[i]:.5f}\t max3: {max_epsilon3[i]:.5f}\t max4: {max_epsilon4[i]}")

# ============= SHOW IMAGES ================= #
show_decompressed_images(dec_images)
show_compressed_images(q_arrays)
show_entropy_graph(epsilons, entropies)
