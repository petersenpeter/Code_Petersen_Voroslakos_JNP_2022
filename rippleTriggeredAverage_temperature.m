function [fig8,fig9] = rippleTriggeredAverage_temperature(ripples,temperature,basename)
    sorting_field = {'peakAmplitude','duration','peakFrequency','temperature'};
    colors = {[0,0,0],[0,1,0],[0,0,1],[1,0,0]};
    fig8 = figure('name',basename);
    fig9 = figure('name',basename);
    window = temperature.sr*1200;
    idx = find(ripples.peaks<1200 | ripples.peaks+1200-1>ripples.peaks(end));
    ripples.peaks(idx) = [];
    ripples.timestamps(idx,:) = [];
    ripples.peakNormedPower(idx) = [];
    ripples.peakFrequency(idx) = [];
    ripples.duration(idx) = [];
    ripples.peakAmplitude(idx) = [];
    ripples.ISIs(idx) = [];
    for j= 1%:length(sorting_field)
        
        plt = [];
        idx_all = interp1(temperature.timestamps,[1:length(temperature.timestamps)],ripples.peaks,'nearest');
        nRipples = length(idx_all);
        [~,ind_sorted] = sort(ripples.(sorting_field{j}));
        intervals = {(1:round(nRipples/3)); (round(nRipples/3):round(2*nRipples/3)); (2*round(nRipples/3):nRipples)};
        idx_groups = {[1:nRipples];intervals{1};intervals{2};intervals{3}};
        
        
        time_axis = [-window+1:window]/temperature.sr;
        for i = 1:4
            clear psth_temperature
            startIndicies = idx_all(ind_sorted(idx_groups{i}))-window+1;
            stopIndicies = idx_all(ind_sorted(idx_groups{i}))+window;
            X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
            X = X(1:end-1);
            psth_temperature = reshape(temperature.data(X)',window*2,[]);
            psth_temperature = psth_temperature-mean(psth_temperature);
            if i == 1 && j == 1
                figure(fig8)
                %             subplot(1,3,1:2)
                %             plot(time_axis,psth_temperature), hold on,
                plot(time_axis,nanmean(psth_temperature,2),'k','linewidth',2), hold on,
                title('Ripple triggered average temperature'), xlabel('Time (s)'), ylabel('Temperature (C)'), grid on, ylim([-0.08,0.08])
                plot([0,0],[-0.1,0.1],'k'),plot([-1,1],[0,0],'k')
                %             subplot(1,3,3)
                %             histogram(diff(ripples.peaks),[2:100]/100,'Normalization','probability'), xlim([0,1]), xlabel('Time (s)'), ylabel('Probability'), grid on, title('Ripple ISI distribution')
            end
            figure(fig9)
            subplot(2,2,j), hold on
            patch([time_axis,flip(time_axis)], [nanmean(psth_temperature,2)+nansem(psth_temperature')';flip(nanmean(psth_temperature,2)-nansem(psth_temperature')')]',colors{i},'EdgeColor','none','FaceAlpha',.2)
            plt(i) = plot(time_axis,nanmean(psth_temperature,2),'color',colors{i}); hold on,
        end
        xlabel('Time (s)'), ylabel('Temperature (C)'), grid on, title(['Grouped and sorted by ',sorting_field{j}]),
        if j == length(sorting_field)
            legend(plt,{'All','1st third','2nd third','3rd third'})
        end
    end