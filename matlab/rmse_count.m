function [rmse] = rmse_count(pred, gt, plot_rmse, directory)
%RMSE_COUNT Summary of this function goes here
%   Detailed explanation goes here

rmse = sqrt(mean((pred(:,2)-gt(:,2)).^2));


if plot_rmse==true
    x = pred(:,2);
    y = (gt(:,2));
    sz = 25;
    c = linspace(1,10,length(x));
    scatter(x,y,sz,c,'filled')
    xlim([0 max(max(x,y))])
    ylim([0 max(max(x,y))])
end



if directory
    
end

end

