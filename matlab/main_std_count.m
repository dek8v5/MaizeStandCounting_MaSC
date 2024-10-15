% clc,
% clear,
% close all,
% 
% tic;
% frag = [1280];
% num_rows =  [4];
% num_range = 1;
% num_class = 3;
% custom_color = true;
% empty_row = [];
% format = 'png';
% each_range = false;
% 
% directory = '/data/e/stand_counts/stand_count_dataset/22r_sweet/for_DMC/';
% % result_dir = '/data/e/stand_counts/corn_detection/darknet/evaluation/evaluation_rgb_square/1280x1280/';
% raw =  dir(fullfile(directory, strcat('*.', string(format))));
% 
% if format=='png'
%     img = imread(fullfile(raw(1).folder, raw(1).name));
%     img = rot_radon(img);
% else
% % 
%     img = tif2png(fullfile(raw(1).folder, raw(1).name));
%     imwrite(img, strcat(directory, 'preprocessed_mosaic.png'));
% end
% 
% [h_ori, w_ori, c] = size(img);
% 
% att = fragment_mosaic(img, frag, directory);
% 
% [label_all] = mosaicback(directory, att, frag);

% filtered out the label (removeduplication from overlap area)
% non-max supression to get the best bounding box from the duplicates
label_all_filtered = label_all((label_all(:,2:5)>0 & label_all(:,2:5)<1));




[selectedBbox, selectedScore, index] = selectStrongestBbox(label_all(:, 2:5), label_all(:, 6), 'RatioType', 'Min');


filtered_label = label_all(index, :);

img_with_bbx = draw_bbox(img, filtered_label, num_class, custom_color, filtered_label);

center_crop = zeros(h_ori, w_ori);
lind = sub2ind(size(center_crop), filtered_label(:,3)+round(filtered_label(:,5)./2), filtered_label(:,2)+round(filtered_label(:,4)./2));
center_crop(lind) = 1;

[range_separator, img_w_range]= detect_range(img_with_bbx, center_crop, num_range, true, true, directory);

[row_separator, img_w_range_row] = detect_row(img_w_range, center_crop, num_rows, range_separator, each_range, true, true, directory);

% plot both ranges and row

figure;
imshow(uint8(img_w_range_row));

%
to_right=true;
num_border=0;

[img, count_per_row, final_row] = count_object(img_w_range_row, range_separator, row_separator, filtered_label, to_right, num_border, directory);

figure,
imshow(img);

imwrite(img, strcat(directory, 'final_stand_count_per_row.png'));
toc;

