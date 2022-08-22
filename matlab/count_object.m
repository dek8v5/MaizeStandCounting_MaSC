function [img, rowsz, final_pred] = count_object(img,range_sep,row_sep, label, to_right, num_border, directory)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
%% counting corn
[h_ori, w_ori, c] = size(img);

idx_h = find(range_sep==1);
h_sep = cat(1, 1, idx_h, h_ori); %range

counter = 1;

final_pred = [];

for i=1:length(h_sep)-1
       
       idx = find(row_sep(i,:)==1);
       w_sep=cat(1, 1, idx', w_ori);
       
       diff_w_sep_min = min(abs(w_sep(1:length(w_sep)-1) - w_sep(2:length(w_sep))));
       if diff_w_sep_min > 200
          diff_w_sep_min = 200; 
       end
       row_actual = 1;
       
       for k=1:size(w_sep,1)-1
       
           rowsz(counter, :) = [w_sep(k), w_sep(k+1), h_sep(i), h_sep(i+1)];
           
           corn_in_row = label(w_sep(k)<=label(:,2)+label(:,4)/2 & label(:,2)+label(:,4)/2 <= w_sep(k+1) & h_sep(i)<=label(:,3)+label(:,5)/2 & label(:,3)+label(:,5)/2<=h_sep(i+1), 1);

           if isempty(corn_in_row)
               std_count(counter) = 0;
               std(row_actual) = 0;
           else
               corn_in_row = corn_in_row + 1;
               std_count(counter) = sum(corn_in_row);
               std(row_actual) = sum(corn_in_row);
           end

           img = insertText(img, [round(w_sep(k+1)-((w_sep(k+1)-w_sep(k))/2)-round(diff_w_sep_min/2)), round(h_sep(i+1)-((h_sep(i+1)-h_sep(i))/2)-round(diff_w_sep_min/2)) ],std_count(counter),'FontSize', round(diff_w_sep_min*0.3) ,'BoxColor', 'red','BoxOpacity',0.4,'TextColor','white');
         
           counter = counter+1; 
           row_actual = row_actual+1;
       end
       
       row_actual = row_actual-1;
       
       if to_right==true 
           if mod(i,2)==0
                row_counter_final = [(num_border+1:row_actual-num_border)' flip(std(num_border+1:row_actual-num_border))'];
           else
                row_counter_final = [(num_border+1:row_actual-num_border)' std(num_border+1:row_actual-num_border)'];
           end
       else
           if mod(i,2)==0
                row_counter_final = [(num_border+1:row_actual-num_border)' std(num_border+1:row_actual-num_border)'];
           else
                row_counter_final = [(num_border+1:row_actual-num_border)' flip(std(num_border+1:row_actual-num_border))'];
           end
       end
       w_sep = [];
       final_pred = cat(1, final_pred, row_counter_final);
 end


rowsz = cat(2, rowsz, std_count');


save(strcat(directory, 'rows_coordinate_and_count.mat'), 'rowsz');
writematrix(final_pred, strcat(directory, 'pred_34_32.txt'), 'Delimiter',',');

% figure;
% imshow(uint8(img));
% imwrite(uint8(img), strcat(directory, 'webodm_yolo_detection.png'));
end

