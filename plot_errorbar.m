function plot_errorbar(x1,y_data,varargin)

% Handling inputs if run from Matlab
p = inputParser;
addParameter(p,'color','k',@isstr);

parse(p,varargin{:})



y_mean = mean(y_data, 'omitnan');
y_std = std(y_data, 'omitnan');
plot(x1(:),y_mean,'o','color',p.Results.color), hold on
plot([x1(:),x1(:)],[y_mean-y_std,y_mean+y_std],'-','color',p.Results.color)
