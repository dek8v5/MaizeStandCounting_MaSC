import os
import numpy as np
import cv2

def fragment_mosaic(img, fragments, save_path):
    #fragmenting mosaic to fragment size dimension for yolo input
    #output: coordinate of fragments and fragments saved in fragment save_path

    att = []

    for frag in fragments:
        print( f'fragment_{frag}')
        frag_dir = os.path.join(save_path, f'fragment_{frag}')
        print(frag_dir)
        if not os.path.exists(frag_dir):
            os.makedirs(frag_dir)

        height, width, channel = img.shape

        h = 0
        w = 0

        ht = int(np.floor(height / (frag * 0.9))) * int(frag * 0.9) + frag
        wt = int(np.floor(width / (frag * 0.9))) * int(frag * 0.9) + frag

        imgg = np.zeros((ht, wt, channel), dtype=img.dtype)
        imgg[:height, :width, :] = img

        h = []
        w = []

        for i in range(0, int(np.floor(ht / (0.9 * frag))) + 1):
            if i * (0.9 * frag) < ht:
                h.append(i * int(0.9 * frag))

        for i in range(0, int(np.floor(wt / (0.9 * frag))) + 1):
            if i * (0.9 * frag) < wt:
                w.append(i * int(0.9 * frag))

        att.append({
            'h': h,
            'w': w,
            'height': height,
            'width': width,
            'channel': channel
        })

        counter = 0

        for i in range(len(h) - 1 ):
            for j in range(len(w) - 1):
                
                fgs = imgg[h[i]:h[i] + frag, w[j]:w[j] + frag, :]
                img_seg = fgs.copy()
                fragment_filename = os.path.join(frag_dir, f'{frag}_fragment_{counter:06d}.png')
                cv2.imwrite(fragment_filename, img_seg)

                counter += 1

        attribute_filename = os.path.join(save_path, f'attributes_{frag}.npz')
        np.savez(attribute_filename, att=att)

    return att
