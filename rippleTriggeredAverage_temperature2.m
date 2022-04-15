function [fig1] = rippleTriggeredAverage_temperature2(ripples,temperature,basename)
    colors = {[0,0,0],[0,1,0],[0,0,1],[1,0,0]};
    fig1 = figure('name',basename);
    
    window = temperature.sr*1200;
    idx = find(ripples.peaks<1200 | ripples.peaks+1200-1>ripples.peaks(end));
    ripples.peaks(idx) = [];
    ripples.timestamps(idx,:) = [];    

    idx_all = interp1(temperature.timestamps,[1:length(temperature.timestamps)],ripples.peaks,'nearest');
    
    time_axis = [-window+1:window]/temperature.sr;
    clear psth_temperature
    startIndicies = idx_all-window+1;
    stopIndicies = idx_all+window;
    X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
    X = X(1:end-1);
    psth_temperature = reshape(temperature.data(X)',window*2,[]);
    
    psth_temperature = psth_temperature-mean(psth_temperature);
    psth_temperature_sem = nanstd(psth_temperature')';
    psth_temperature_mean = nanmean(psth_temperature,2);
    figure(fig1)
    patch([time_axis,flip(time_axis)], [psth_temperature_mean+psth_temperature_sem;flip(psth_temperature_mean-psth_temperature_sem)]',[0,0,0],'EdgeColor','none','FaceAlpha',.2), hold on
    plot(time_axis,nanmean(psth_temperature,2),'color',[0,0,0]); hold on,
    xlabel('Time (s)'), ylabel('Temperature (C)'), grid on,
end