function compareState(data,states)
    fieldsToAnalize = fieldnames(data);
    k = 1;
    for i = 1:numel(fieldsToAnalize)
        if all(size(data.(fieldsToAnalize{i})) == [size(data.timestamps,1),1]) && isnumeric(data.(fieldsToAnalize{i}))
            set1 = data.(fieldsToAnalize{i})(states);
            set2 = data.(fieldsToAnalize{i})(~states);
            [h,P] = kstest2(set1,set2);
            
            subplot(3,3,k)
            violinplot(data.(fieldsToAnalize{i}), states);
            ylabel(fieldsToAnalize{i})
            title(['KS-test = ',num2str(h),', P = ',num2str(P,3)])
            xlabel('Cooling')
            k = k+1;
        end
    end
end