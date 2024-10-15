function [img] = draw_bbox(img, label, num_class, custColor, label_print)
%draw bounding box for custom number of classes
%   input:  image for the bounding box
%           label for the mosaic, real dimention for the bbox info
%           number of class 

%color index based on number of class
ColMap = jet(256); % In the range 0-1
randomIndex = randi(length(ColMap), num_class,1);

if custColor == true
    ColMap = [ 255 2 255; 220 150 40; 70 230 70]/255; % In the range 0-1
    %ColMap = [ 255 2 2; 255 247 0; 7 138 0]/255; % In the range 0-1
    randomIndex = [1;2;3];
end


%drawing
for i=1:num_class
    if i==1
       label_str = string(strcat('corn', {' '}, num2str(label(label(:,1)==i-1, 6),'%0.2f'))); 
    elseif i==2
        label_str = string(strcat('double', {' '}, num2str(label(label(:,1)==i-1, 6),'%0.2f')));
    else
        label_str = string(strcat('triple', {' '}, num2str(label(label(:,1)==i-1, 6),'%0.2f')));
    end
    
    if label_print==true
        img = insertObjectAnnotation(img,'rectangle',label(label(:,1)==i-1, 2:5), label_str, 'TextBoxOpacity',0.7,'FontSize',50, 'Color', round(255 * ColMap(randomIndex(i), :)), 'LineWidth', 5);
    else
        img =  insertShape(img, 'rectangle', label(label(:,1)==i-1, 2:5), 'Color', round(255 * ColMap(randomIndex(i), :)), 'LineWidth', 5);
    end

        
            
            
end

end

