import os
import numpy as np

def mosaicback(directory, att, fragments):
    #it will mosaic back fragment from yolo and adjust the label with respect
    #to the whole mosaic
    #input: directory, attribute of fragments, and frag size
    #output: mosaic and label

    label_final_all = []

    for a in range(len(fragments)):
        print(a)
        print(att)
        frag = fragments[a]
        #just fixing directory set up, tidying up
        #img_dir = os.path.join(directory, 'yolo_result')
        img_dir = directory
        label_dir = os.path.join(img_dir, 'labels') 
        all_img = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        all_txt = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

        w_sep = att[a]['w']
        h_sep = att[a]['h']
        h_ori = att[a]['height']
        w_ori = att[a]['width']
        c_ori = att[a]['channel']

        label_final = []

        for k in range(len(all_img) // ((len(h_sep) - 1) * (len(w_sep) - 1))):

            for i in range(len(h_sep) - 1):
                for j in range(len(w_sep) - 1):
                    #index correction
                    z = (k * ((len(h_sep) - 1) * (len(w_sep) - 1))) + (i * (len(w_sep) - 1)) + j
                    img_path = os.path.join(img_dir, all_img[z])
                    txt_path = os.path.join(label_dir, all_txt[z])

                    #read txt file
                    with open(txt_path, 'r') as file:
                        lines = file.readlines()

                    label_original = [list(map(float, line.strip().split())) for line in lines]

                    #filtering prediction's label that does not make sense
                    label = [l for l in label_original if 0.01 < l[3] < 1 and 0.01 < l[4] < 1]

                    temp_label = np.zeros((len(label), 6))
                    temp_label[:, 0] = [l[0] for l in label]
                    temp_label[:, 1] = [(l[1] * frag + (w_sep[j] + 1)) for l in label]
                    temp_label[:, 2] = [(l[2] * frag + (h_sep[i] + 1)) for l in label]
                    temp_label[:, 3] = [(l[3] * frag) for l in label]
                    temp_label[:, 4] = [(l[4] * frag) for l in label]
                    temp_label[:, 5] = [l[5] for l in label]

                    label_final.extend(temp_label)

        label_final = np.array(label_final)
        norm_label_final = label_final.copy()
        norm_label_final[:, 1:5] = np.array([
            label_final[:, 1] / w_ori,
            label_final[:, 2] / h_ori,
            label_final[:, 3] / w_ori,
            label_final[:, 4] / h_ori
        ]).T

        np.savetxt(os.path.join(directory, f'mosaic_prediction_normalized_{frag}.txt'), norm_label_final, delimiter=' ', fmt='%f')

        label_final[:, 1:5] = np.array([
            label_final[:, 1] - label_final[:, 3] / 2,
            label_final[:, 2] - label_final[:, 4] / 2,
            label_final[:, 3],
            label_final[:, 4]
        ]).T.astype(int)

        label_final_all.extend(label_final)

    return np.array(label_final_all)
