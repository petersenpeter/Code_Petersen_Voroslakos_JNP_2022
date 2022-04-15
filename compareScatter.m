function stats = compareScatter(data,states)
    stats = {};
    fieldsToAnalize = fieldnames(data);
    k = 1;
    for i = 1:numel(fieldsToAnalize)
        if all(size(data.(fieldsToAnalize{i})) == [size(data.timestamps,1),1]) && isnumeric(data.(fieldsToAnalize{i}))
            subplot(3,3,k)
            x = data.temperature;
            y1 = data.(fieldsToAnalize{i});
            plot(x,y1,'.'), hold on
            if exist('states','var')
                plot(x(~states),y1(~states),'.r')
                plot(x(states),y1(states),'.b')
            end
            P1 = polyfit(x,y1,1); yfit = P1(1)*x+P1(2); plot(x,yfit,'-k','linewidth',2);
            [R,P] = corrcoef(x,y1);
            xlabel('Temperature'), ylabel(fieldsToAnalize{i})
            title({['Slope:' num2str(P1(1),3)],['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)]})
            stats.(fieldsToAnalize{i}).slope = P1(1);
            stats.(fieldsToAnalize{i}).R = R(2,1);
            stats.(fieldsToAnalize{i}).P = P(2,1);
            k = k+1;
        end
    end
end