function [lm_row, img] = detect_range(img, center_crop, num_ranges, draw, plot_graph, directory)
%detect_range detecting ranges in the whole field and 
%   Detailed explanation goes here
%% ranges
sum_row = sum(center_crop, 2); 
[h_ori,w_ori, c] = size(img);

lm_range_len = 100000;
while lm_range_len > num_ranges-1
        sum_row = smoothdata(sum_row,'movmean', round(length(sum_row)/100));
        lm_row = islocalmin(sum_row);
        lm_range_len = size(find(lm_row==1),1);
end
lm_row = islocalmin(sum_row);

sum_row = rescale(sum_row);

if plot_graph == true
    fig = figure;
    subplot(1,2,1);
    x = 1:1:length(sum_row);
    plot(sum_row, x, sum_row(lm_row), x(lm_row), 'r*' );
    set(gca, 'YDir', 'reverse');
    set(gca,'box','off')
    title('plot sum green on height');
    saveas(fig,strcat(directory, '/range_plot.png'));
end

if draw ==true
    w_lines = int32([ones(length(find(lm_row==1)),1), find(lm_row==1), ones(length( find(lm_row==1)),1)*w_ori ,  find(lm_row==1)]);
    img = insertShape(img,'Line', w_lines , 'Color', [16,255,239]  ,'LineWidth',5); 
    
end



end

