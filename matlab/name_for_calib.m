clc;
clear;
close;

directory = '/data/e/stand_counts/MATLAB/11_9_cal_images/use/';
result_dir = '/data/e/stand_counts/MATLAB/11_9_cal_images/use/correct_name/';

all_img =  dir(fullfile(directory,'*.png'));

for k = 1:length(all_img)
    
    filename = fullfile(directory, all_img(k).name);

    img = imread(filename);
    
    imwrite(img,  strcat(result_dir, sprintf('Frame%03d.jpg', k)));
end