function [img] = tif2png(img)
%tif2png: converted tif image from webODM to png with 0 as background and
%rotated the image to vertical row orientation with radon transform
%   input img (tif)
%   output img (png)

[img, ~, alpha] = imread(img);


J = imresize(img(:,:,1:3),1);
alpha2 = imresize(img(:,:,4), 1);

img =J.*(alpha2/255);
% 
 img_bw = 2*img(:,:,2) - img(:,:,1) -img(:,:,3);
% 
% 
 th = graythresh(img_bw);
% 
img_bw = (img_bw>th);

%%%%%%%%%%%%%%%%%%% radon transform %%%%%%%%%%%%%%%%%%
%radon transform
 
theta = 1:180;
[Rad,xp] = radon(img_bw,theta);
plot(theta, sum(Rad, 1));

variance = var(Rad);
[max_angl, idx] = max(variance);
angl = idx;

figure,
plot(theta, variance, theta(idx), variance(idx), 'rx');
text(theta(idx)+5, variance(idx)-5, num2str(idx))
set(gca,'box','off')
title('Randon transform variance 0-180^{\circ}','Interpreter','tex');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

angl = 360-angl;

ExG_rot = imrotate(img_bw, angl);
img_rot = imrotate(img, angl);

img_rot = uint8(img_rot);
alpha2 = imrotate(alpha2, angl);


[row,col,c] = find(alpha2(:,:,1)~=0);

min_row = min(row);
max_row = max(row);
min_col = min(col);
max_col = max(col);

figure,
imshow(img_rot(min_row:max_row, min_col:max_col, :));

img = img_rot(min_row:max_row, min_col:max_col, :);
% % % % imwrite((img_rot(min_row:max_row, min_col:max_col, :)), strcat(directory, '34_597_preprocessed_no_alpha.png');%, 'Alpha', alpha2(min_row:max_row, min_col:max_col, :) );
end

