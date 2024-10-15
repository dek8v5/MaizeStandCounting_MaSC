clc
clear
close all;


% read ground_truth file from 
txt_name = '/data/e/stand_counts/MATLAB/ground_truth/gt_fix/19r_30_400-405.csv';

fid = fopen(txt_name);
M = textscan(fid, '%s %d %d %d %d %s %s %s', 'Delimiter',',','HeaderLines',1); 
fclose(fid);

rowss = cell2mat(M{1,1});

row_int = str2double(string(rowss(:,size(rowss,2)-2:size(rowss,2))));

label = [row_int, M{1,2}-M{1,4}];

label_sorted = sortrows(label, 1);

writematrix(label_sorted, strcat('/data/e/stand_counts/MATLAB/ground_truth/gt_final/', '19r_30_gt_final.txt'), 'Delimiter',',');



%read prediction
txt_name = '/data/e/stand_counts/MATLAB/IMAGES/19r_30/pred_34_32.txt';

fid = fopen(txt_name);
pred_count = textscan(fid, '%d %d', 'Delimiter',','); 
fclose(fid);

pred_stand_count = [pred_count{1,1}, pred_count{1,2}];

%% evaluation
rmse = rmse_count(pred_stand_count, label_sorted, true, '/data/e/stand_counts/MATLAB/IMAGES/19r_30/');