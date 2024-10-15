function [att] = fragment_mosaic(img, fragments, directory)
%fragmenting mosaic to fragement size dimension for yolo input
%output: coordinate of fragments and fragments saved in fregment directory

for k=1:length(fragments)
    frag = fragments(k); 
   
    frag_dir = strcat(directory, 'fragment_' , string(frag) , '/');

    if ~exist(strcat(frag_dir), 'dir')
        mkdir(frag_dir);
    end


    [height, width, channel] = size(img);

    h = 0;
    w = 0;

    ht = floor(height/(frag*0.9))*(frag*0.9)+frag;
    wt = floor(width/(frag*0.9))*(frag*0.9)+frag;

    imgg = zeros(ht,wt,channel);
    imgg(1:height, 1:width, 1:channel) = img; 

    for i=1:round(ht/(0.9*frag))
        if i*(0.9*frag) < ht
            h = cat(2, h, i*(0.9*frag));
        end
    end
    for i=1:round(wt/(0.9*frag))
        if i*(0.9*frag) < wt
            w = cat(2, w, i*(0.9*frag));
        end
    end

    h = cat(2, h);
    w = cat(2, w);

    att(k).h = h;
    att(k).w = w;
    att(k).height = height;
    att(k).width = width;
    att(k).channel = channel;



    counter = 1;

    for i=1:length(h)-1
        for j=1:length(w)-1
           %fragment the image
           fgs = imgg(h(i)+1:h(i)+frag, w(j)+1:w(j)+frag, :);
           [hg, wd, ch] = size(fgs);
           img_seg(1:hg, 1:wd, 1:ch) =  fgs;
           imwrite(uint8(img_seg), strcat(frag_dir, string(frag), sprintf('_fragment_%06d.png', counter)));

           img_seg = zeros(frag,frag,3);

           counter = counter+1;

        end 
    end

    save(strcat(directory, 'attributes_' , string(frag), '.mat'),'att');
end
end

