function [separator, img] = detect_row(img, center_crop, num_row, range_separator, each_range, draw, plot_graph, directory)
%detect_row detecting row as a whole or based on range
%   Detailed explanation goes here
lm_row_len = 10000;

[h_ori, w_ori, c] =size(img);

idx = find(range_separator==1);
range_separator = cat(1, 1, idx, h_ori);

if each_range == true
    figs = figure;
    
    for i=1:length(range_separator)-1
        sum_col = sum(center_crop(range_separator(i):range_separator(i+1), :), 1);
        
        while lm_row_len > num_row(i)-1
            sum_col = smoothdata(sum_col, 'movmean', round(length(sum_col)/200));
            lm_col = islocalmin(sum_col);
            lm_row_len = size(find(lm_col==1),2);
            
        end

        if draw ==true
            h_lines = int32([ find(lm_col==1)', repmat(range_separator(i), length(find(lm_col==1)'), 1), find(lm_col==1)', repmat(range_separator(i+1), length(find(lm_col==1)'), 1)]);
            img = insertShape(img,'Line', h_lines, 'Color', [16,255,239] ,'LineWidth',5);
        end
        
        if plot_graph == true
            subplot(1, length(range_separator)-1, i)
            
            x = 1:1:length(sum_col);
            plot(x, sum_col, x(lm_col), sum_col(lm_col), 'r*');
            set(gca,'box','off')
            title('plot sum green on width');

            saveas(figs, strcat(directory, '/row_plot.png'));

        end 
        
        sum_col_final = sum_col;
        separator(i,:) = lm_col;
        sum_col = 0;
        lm_row = 0;
        lm_row_len = 10000;
        

    end
    
    sum_col = rescale(sum_col_final); 
    
else
    
    sum_col = sum(center_crop, 1);
    
    while lm_row_len > num_row-1
        
        sum_col = smoothdata(sum_col, 'movmean', round(length(sum_col)/200));
        lm_col = islocalmin(sum_col);
        lm_row_len = size(find(lm_col==1),2);

    end
    
    sum_col = rescale(sum_col);
    
    separator = repmat(lm_col, length(range_separator)-1, 1);
    
    if draw ==true
        h_lines = int32([ find(lm_col==1)', ones(length(find(lm_col==1)),1), find(lm_col==1)', ones(length(find(lm_col==1)),1)*h_ori]);
        img = insertShape(img,'Line', h_lines, 'Color', [16,255,239] ,'LineWidth',5);
 
    end
    
    if plot_graph == true
        figs = figure;
        x = 1:1:length(sum_col);
        plot(x, sum_col, x(lm_col), sum_col(lm_col), 'r*');
        set(gca,'box','off')
        title('plot sum green on width');

        saveas(figs,strcat(directory, '/row_plot.png'));
    end 

end

end

