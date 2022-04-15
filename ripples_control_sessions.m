% Ripples control sessions
% Ripple analysis
clear all,
% close all

% MS10: 61 (Peter_MS10_170317_153237_concat) OK
%       62 (Peter_MS10_170314_163038) % gamma id: 62 OK
%       63 (Peter_MS10_170315_123936)
%       64 (Peter_MS10_170307_154746_concat) OK
% MS12: 78 (Peter_MS12_170714_122034_concat) OK
%       79 (Peter_MS12_170715_111545_concat) OK
%       80 (Peter_MS12_170716_172307_concat) OK
%       81 (Peter_MS12_170717_111614_concat) OK
%       83 (Peter_MS12_170719_095305_concat) OK
% MS13: 92 (Peter_MS13_171129_105507_concat) OK, Ripples processed
%       93 (Peter_MS13_171130_121758_concat) OK, Ripples processed
%       88 (Peter_MS13_171110_163224_concat) No good cooling behavior
%       91 (Peter_MS13_171128_113924_concat) OK
%       94 (Peter_MS13_171201_130527_concat) OK
% MS21: 126 (Peter_MS21_180629_110332_concat) OK, unit count:
%       140 (Peter_MS21_180627_143449_concat) OK, Ripples processed
%       143 (Peter_MS21_180719_155941_concat, control)
%       149 (Peter_MS21_180625_153927_concat) OK
%       153 (Peter_MS21_180712_103200_concat) OK
%       151 (Peter_MS21_180628_155921_concat) OK
%       159 (Peter_MS21_180807_122213_concat, control)
% MS22: 139 (Peter_MS22_180628_120341_concat) OK, Ripples processed
%       127 (Peter_MS22_180629_110319_concat) OK, Ripples processed
%       144 (Peter_MS22_180719_122813_concat, control)
%       168 (Peter_MS22_180720_110055_concat) OK
%       166 (Peter_MS22_180711_112912_concat) OK
% cd(basepath)
% [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
% basepath = '/Volumes/Samsung_T5/GlobusDataFolder/Temp_R04_20201027';
% basepath = 'Z:\Homes\voerom01\Temperature\Temp_R05\Temp_R05_20201219';
% basepath = '/Volumes/Samsung_T5/GlobusDataFolder/Temp_R04_20201114';

% Defining local paths
user_name = 'Peter';
if ismac
%     basepath_root = '/Volumes/Samsung_T5/GlobusDataFolder'
    basepath_root = '/Volumes/Peter_SSD_4/';
    local_path = '/Users/peterpetersen/Dropbox/Buzsakilab Postdoc/Matlab/';
elseif strcmp(user_name,'Peter')
%     basepath_root = 'Z:\Buzsakilabspace\LabShare\PeterPetersen\';
    basepath_root = 'D:\'
    local_path = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';

else
    basepath_root = 'D:\';
    local_path = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';
end

if exist([local_path 'sessions_metadata.mat'])
    load([local_path 'sessions_metadata.mat'],'sessions_metadata');
else
    DOCID = '1hGU_NPMt2wPanXKROXuPlj06saA1pFTA1391w44Tl5I';
    session_temp = GetGoogleSpreadsheet(DOCID)';
    sessions_metadata = session_temp(:,2:end)';
    sessions_metadata = cell2table(sessions_metadata);
    sessions_metadata.Properties.VariableNames = session_temp(:,1);
    save([local_path 'sessions_metadata.mat'],'sessions_metadata');
end

%% Ripples analysis nonREM 
sessions_control_peter = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170716_172307_concat', 'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS21_180628_155921_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', ...
    'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};

sessions_control_nonREM = {'Temp_R05_20201229','Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', 'Peter_MS22_180711_112912_concat'};
sessions_all = sessions_control_nonREM;

gausswin_steps = 160;

% i = 5
% basename = sessions_all{i};
% disp(' ')
% disp([num2str(i),'/',num2str(length(sessions_all)),': ', basename])
% 
% animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
% basepath = fullfile(basepath_root,animal_subject,basename);
% cd(basepath)
% StateExplorer
% TheStateEditor(basename)

for i = 1:numel(sessions_all)
    % basename = 'Peter_MS13_171130_121758_concat';
%     basename = 'Peter_MS13_171129_105507_concat';
    % basename = 'Peter_MS22_180628_120341_concat';
    
    basename = sessions_all{i};
    disp(' ')
    disp([num2str(i),'/',num2str(length(sessions_all)),': ', basename])
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    
    session = loadSession(basepath);
    load(fullfile(basepath,[basename,'.ripples.events.mat']));
    load(fullfile(basepath,[basename,'.temperature.timeseries.mat']));
    
    if isfield(temperature.states,'Cooling')
        temperature.states.cooling = temperature.states.Cooling;
    end
    
    if isfield(ripples,'flagged')
        ripples.peaks(ripples.flagged) = [];
        ripples.timestamps(ripples.flagged,:) = [];
        ripples.peakNormedPower(ripples.flagged) = [];
    end
    
    idx = find(InIntervals(ripples.peaks,temperature.states.cooling));
    ripples.peaks(idx) = [];
    ripples.timestamps(idx,:) = [];
    ripples.peakNormedPower(idx) = [];
    
    if isfield(temperature.states,'Bad')
        idx = find(InIntervals(ripples.peaks, temperature.states.Bad));
        ripples.peaks(idx) = [];
        ripples.timestamps(idx,:) = [];
        ripples.peakNormedPower(idx) = [];
    end
    
    if isfield(temperature.states,'CoolingInterval')
        disp('Removing Cooling Intervals')
        idx = find(InIntervals(ripples.peaks, temperature.states.CoolingInterval));
        ripples.peaks(idx) = [];
        ripples.timestamps(idx,:) = [];
        ripples.peakNormedPower(idx) = [];
    end
    
%     % Filtering by non-REM state
    SleepState = loadStruct('SleepState','states','session',session);
    idx = InIntervals(ripples.peaks,SleepState.ints.NREMstate);
    ripples.peaks = ripples.peaks(idx);
    ripples.timestamps = ripples.timestamps(idx,:);
    ripples.peakNormedPower = ripples.peakNormedPower(idx);
    
    idx2 = find(InIntervals(temperature.timestamps,temperature.states.cooling));
    temperature.timestamps(idx2) = [];
    temperature.data(idx2) = [];
    
    lfp = bz_GetLFP(ripples.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
    ripfiltlfp = bz_Filter(lfp.data,'passband',[110,180],'filter','fir1');
    [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
    % PlotRippleStats(ripples,maps,data,stats)
    ripples.peakFrequency = data.peakFrequency;
    ripples.duration = data.duration*1000;
    ripples.peakAmplitude = data.peakAmplitude/1000;
    ripples.ISIs = diff(ripples.peaks); ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
    
    t_minutes = [0:ceil(max(ripples.peaks)/30)];
    SleepState.idx.states(SleepState.idx.states ~= 3) = 0;
    SleepState.idx.states(SleepState.idx.states == 3) = 1;
    nREM_state = zeros(1,size(t_minutes,2)-1);
    for j = 1:numel(nREM_state)-1
        nREM_state(j) = sum(SleepState.idx.states((j-1)*30+1:j*30));
    end
    nREM_state(nREM_state==0)=1;
    
    ripple_rate = histcounts(ripples.peaks/30,t_minutes);
    ripple_rate = ripple_rate./nREM_state;
    %ripple_rate = ripple_rate/30;
    ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks/30);
    ripples.rate(ripples.rate>2)= 1;
    
    rem1 = rem(length(temperature.data),temperature.sr);
    temperature.timestamps1 = temperature.timestamps(1:temperature.sr:end);
    temperature.data1 = nanmean(reshape(temperature.data(1:length(temperature.data)-rem1),temperature.sr,[]));
    temperature.timestamps1 = temperature.timestamps1(1:length(temperature.data1));
    ripples.temperature = interp1(temperature.timestamps1,temperature.data1, ripples.peaks);
    
    fig1 = figure('name',basename,'position',[50,50,1200,900]); % Figure 1
    subplot(3,3,1:2)
    plot(ripples.peaks, ripples.peakFrequency,'.b'), hold on,
    plot(temperature.timestamps1,10*zscore(temperature.data1)+149,'r')
    % plot(temperature.timestamps(temperature.data>20),15*zscore(temperature.data(temperature.data>20))+147,'r')
    % plot(temperature.timestamps,temperature.data+100)
    plot(ripples.peaks, nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge'),'-g'),
    legend({'Ripples','Temperature (zscored & aligned)','Ripple freq running average'},'AutoUpdate','off')
    axis tight, %gridxy(cumsum(session.epochs.duration))
    title('Ripple frequency'), xlabel('Time (s)'), ylabel('Frequency (Hz)'),
    ylim([100,200])
    
    subplot(3,3,3)
    plot(ripples.temperature,ripples.peakFrequency,'.b'), hold on
    plot(ripples.temperature,nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge'),'.g')
    title('Ripple rate vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Ripple rate (Hz)')
    x = ripples.temperature;
    y1 = ripples.peakFrequency;
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
    text(0.05,1.1,['Slope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
    [R,P] = corrcoef(x,y1);
    text(0.05,1.15,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    ylim([100,200])
    x = ripples.temperature;
    y1 = nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge');
    [R,P] = corrcoef(x,y1);
    text(0.05,1.2,['R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    ylim([100,200])
    
    subplot(3,3,4:5)
    plot(ripples.peaks, ripples.rate,'.b'), hold on,
    plot(temperature.timestamps1,(zscore(temperature.data1)+2)/3,'r')
    plot(ripples.peaks, nanconv(ripples.rate,gausswin(gausswin_steps),'edge'),'-k'),
    axis tight,
    title('Ripple rate (Hz)'), xlabel('Time (s)'), ylabel('Frequency (Hz)'),
    
    subplot(3,3,6)
    plot(ripples.temperature,ripples.rate,'.b'), hold on
    title('Ripple rate vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Ripple rate (Hz)')
    x = ripples.temperature(~isnan(ripples.rate));
    y1 = ripples.rate(~isnan(ripples.rate));
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
    text(0.05,1.15,['Slope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
    [R,P] = corrcoef(x,y1);
    text(0.05,1.1,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    
    subplot(3,3,[7,8])
    plot(ripples.peaks, ripples.duration,'.b'), hold on,
    plot(temperature.timestamps1,zscore(temperature.data1)*10+60,'r')
    plot(ripples.peaks, nanconv(ripples.duration,gausswin(gausswin_steps),'edge'),'-g'),
    axis tight
    title('Ripple duration'), xlabel('Time (s)'), ylabel('Duration (ms)'),
    
    subplot(3,3,9)
    plot(ripples.temperature,ripples.duration,'.'), hold on
    plot(ripples.temperature,nanconv(ripples.duration,gausswin(gausswin_steps),'edge'),'.g')
    title('Ripple duration vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Duration (ms, log10)')
    x = ripples.temperature;
    y1 = ripples.duration;
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
    text(0.05,1.1,['Slope: ' num2str(P(1),3),' ms/degree'],'Color','k','Units','normalized')
    [R,P] = corrcoef(x,y1);
    text(0.05,1.15,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    x = ripples.temperature;
    y1 = nanconv(ripples.duration,gausswin(gausswin_steps),'edge');
    [R,P] = corrcoef(x,y1);
    text(0.05,1.2,['R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    
    saveas(fig1,fullfile(local_path, 'RipplesVsTemperature/Figures',[basename,'.ripplesControlStats1.png']))
    drawnow
    figure
    nonREM_stats = compareScatter(ripples);
     
    ripples_nonREM = ripples;
    saveStruct(ripples_nonREM,'events','session',session);
    save(fullfile(basepath,[basename,'.nonREM_stats.mat']), 'nonREM_stats');
    
    ripples_smooth = ripples;   
    ripples_smooth.peakFrequency = nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge');
    ripples_smooth.rate = nanconv(ripples.rate,gausswin(gausswin_steps),'edge');
    ripples_smooth.duration = nanconv(ripples.duration,gausswin(gausswin_steps),'edge');
    nonREM_stats_smooth = compareScatter(ripples_smooth);
    save(fullfile(basepath,[basename,'.nonREM_stats_smooth.mat']), 'nonREM_stats_smooth');
    
    ripples_nonREM_smooth = ripples_smooth;
    saveStruct(ripples_nonREM_smooth,'events','session',session);
end

%%
% Figure 2 group data nonREM
batchData = {};
k = 1;
for i = 1:numel(sessions_all)
    sessionID = i;
    basename = sessions_all{sessionID};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    load(fullfile(basepath,[basename,'.nonREM_stats.mat']), 'nonREM_stats');
    
    fieldsToAnalize = fieldnames(nonREM_stats);
    for j = 1:numel(fieldsToAnalize)
        batchData.(fieldsToAnalize{j}).slope(k) = nonREM_stats.(fieldsToAnalize{j}).slope;
        batchData.(fieldsToAnalize{j}).R(k) = nonREM_stats.(fieldsToAnalize{j}).R;
        batchData.(fieldsToAnalize{j}).P(k) = nonREM_stats.(fieldsToAnalize{j}).P;
    end
    k=k+1;
end

fig5 = figure('name','Slope');
fig6 = figure('name','R');
fig7 = figure('name','P');
fieldsToAnalize = fieldnames(nonREM_stats);
fieldsToAnalize = setdiff(fieldsToAnalize,{'cooling','timestamps','peaks','detectorinfo','noise','temperature'});
for j = 1:numel(fieldsToAnalize)
    figure(fig5)
    subplot(2,3,j)
    raincloud_plot(batchData.(fieldsToAnalize{j}).slope,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('Slope')
    figure(fig6)
    subplot(2,3,j)
    raincloud_plot(batchData.(fieldsToAnalize{j}).R,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('R')
    %     plot(1,batchData.(fieldsToAnalize{j}).R,'o'); hold on, title(fieldsToAnalize{j}), xlabel('R')
    figure(fig7)
    subplot(2,3,j)
    P1 = log10(batchData.(fieldsToAnalize{j}).P);
    P1(P1==-Inf)=-20;
    if any(P1>-20)
        raincloud_plot(P1,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('P')
    else
        plot(1,P1,'o'), hold on, title(fieldsToAnalize{j}), xlabel('P')
    end    
end
saveas(fig5,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_nonREM_Slope.png'))
saveas(fig6,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_nonREM_R.png'))
saveas(fig7,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_nonREM_P.png'))

%%
figure % Figure 2
subplot(2,3,[1,2])
plot(ripples.peaks, ripples.peakAmplitude,'x'), hold on,
plot(temperature.timestamps(temperature.data>20),zscore(temperature.data(temperature.data>20))/40+0.15,'r')
% plot(temperature.timestamps,temperature.data+100)
plot(ripples.peaks, nanconv(ripples.peakAmplitude,gausswin(gausswin_steps),'edge'),'-k'),
axis tight, %gridxy(cumsum(session.epochs.duration))
title('Ripple peakAmplitude'), xlabel('Time (s)'), ylabel('peakAmplitude'),
subplot(2,3,3)
plot(ripples.temperature,ripples.peakAmplitude,'.'), hold on
plot(ripples.temperature,nanconv(ripples.peakAmplitude,gausswin(gausswin_steps),'edge'),'.r')
title('Ripple peakAmplitude vs Temperature'), xlabel('Temperature (degree C)'), ylabel('peakAmplitude')

x = ripples.temperature(ripples.temperature>20);
y1 = ripples.peakAmplitude(ripples.temperature>20);
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(36.5,0.28,['Slope: ' num2str(P(1),3),' /degree'],'Color','k')
[R,P] = corrcoef(x,y1);
text(36.5,0.25,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')

x = ripples.temperature(ripples.temperature>35.5);
y1 = ripples.peakAmplitude(ripples.temperature>35.5);
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(36.5,0.28,['Slope: ' num2str(P(1),3),' /degree'],'Color','k')
[R,P] = corrcoef(x,y1);
text(36.5,0.25,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')

subplot(2,3,[4,5])
plot(ripples.peaks, ripples.ISIs,'x'), hold on,
plot(temperature.timestamps(temperature.data>20),zscore(temperature.data(temperature.data>20))/2+0.15,'r')
% plot(temperature.timestamps,temperature.data+100)
plot(ripples.peaks, nanconv(ripples.ISIs,gausswin(gausswin_steps),'edge'),'-k'),
axis tight, %gridxy(cumsum(session.epochs.duration))
title('Ripple ISIs (log10)'), xlabel('Time (s)'), ylabel('ISIs (log10)'),
subplot(2,3,6)
plot(ripples.temperature,ripples.ISIs,'.'), hold on
plot(ripples.temperature,nanconv(ripples.ISIs,gausswin(gausswin_steps),'edge'),'.r')
title('Ripple ISI vs Temperature'), xlabel('Temperature (degree C)'), ylabel('ISIs (log10)')
x = ripples.temperature(ripples.temperature>20);
y1 = ripples.ISIs(ripples.temperature>20);
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(33,0.28,['Slope: ' num2str(P(1),3),' /degree'],'Color','k')
[R,P] = corrcoef(x,y1);
text(33,0.25,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')

x = ripples.temperature(ripples.temperature>35.5);
y1 = ripples.ISIs(ripples.temperature>35.5);
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(36.5,0.28,['Slope: ' num2str(P(1),3),' /degree'],'Color','k')
[R,P] = corrcoef(x,y1);
text(36.5,0.25,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')


% Sort by duration vs amplitude of ripple
[~,dursort]=sort(data.duration,1,'descend');
[~,ampsort]=sort(data.peakAmplitude,1,'descend'); axis tight

figure, % Figure 3
subplot 221
imagesc(maps.amplitude(ampsort,:))
title('SPW-R Amplitude: sorted by amplitude')
subplot 222
imagesc(maps.amplitude(dursort,:))
title('SPW-R Amplitude: sorted by duration')
subplot 223
imagesc(maps.ripples(ampsort,:))
title('SPW-R Filtered Signal: sorted by amplitude')
subplot 224
imagesc(maps.ripples(dursort,:))
title('SPW-R Filtered Signal: sorted by duration')

%% % Theta triggered average of the temperature
phasezero_times = find(diff(theta.phase>0)==1)/theta.sr;
phasezero_speed = interp1(animal.time,animal.speed,phasezero_times);
phasezero_temperature = interp1(animal.time,animal.temperature,phasezero_times);
phasezero_theta_freq = interp1([1:length(theta.freq)]/theta.sr_freq,theta.freq,phasezero_times);
phasezero_theta_power = interp1([1:length(theta.power)]/theta.sr_freq,theta.power,phasezero_times);
phasezero_speed_thresholded = find(phasezero_speed>30 & phasezero_temperature>36);

figure(1004)
colors = {[0,0,0],[0,1,0],[0,0,1],[1,0,0]};
window = temperature.sr;
time_axis = [-window+1:window]/temperature.sr;
sorting_field = {'Time','Running speed','Theta freq','Theta power'};
for j= 1:length(sorting_field)
    plt = [];
    idx_all = interp1(temperature.timestamps,[1:length(temperature.timestamps)],phasezero_times(phasezero_speed_thresholded),'nearest');
    nEvents = length(idx_all);
    if j == 1
        ind_sorted = 1:nEvents;
    elseif j == 2
        [~,ind_sorted] = sort(phasezero_speed(phasezero_speed_thresholded));
    elseif j == 3
        [~,ind_sorted] = sort(phasezero_theta_freq(phasezero_speed_thresholded));
    elseif j == 4
        [~,ind_sorted] = sort(phasezero_theta_power(phasezero_speed_thresholded));
    end
    
    %     [~,ind_sorted] = sort(ripples.(sorting_field{j}));
    intervals = {(1:round(nEvents/3)); (round(nEvents/3):round(2*nEvents/3)); (2*round(nEvents/3):nEvents)};
    idx_groups = {[1:nEvents];intervals{1};intervals{2};intervals{3}};
    
    for i = 1:4
        clear psth_temperature
        startIndicies = idx_all(ind_sorted(idx_groups{i}))-window+1;
        stopIndicies = idx_all(ind_sorted(idx_groups{i}))+window;
        X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
        X = X(1:end-1);
        psth_temperature = reshape(temperature.data(X)',window*2,[]);
        psth_temperature = psth_temperature-mean(psth_temperature);
        if i == 1 && j == 1
            figure(1003)
            plot(time_axis,psth_temperature), hold on, plot(time_axis,nanmean(psth_temperature,2),'k','linewidth',2)
            title('Theta phase triggered average temperature'), xlabel('Time (s)'), ylabel('Temperature (C)'), grid on, ylim([-0.08,0.08])
            plot([0,0],[-0.1,0.1],'k'),plot([-1,1],[0,0],'k')
        end
        figure(1004)
        subplot(2,2,j), hold on
        patch([time_axis,flip(time_axis)], [nanmean(psth_temperature,2)+nansem(psth_temperature')';flip(nanmean(psth_temperature,2)-nansem(psth_temperature')')]',colors{i},'EdgeColor','none','FaceAlpha',.2)
        plt(i) = plot(time_axis,nanmean(psth_temperature,2),'color',colors{i}); hold on,
    end
    xlabel('Time (s)'), ylabel('Temperature (C)'), grid on, title(['Grouped and sorted by ',sorting_field{j}]),
    if j == 1%length(sorting_field)
        legend(plt,{'All','1st third','2nd third','3rd third'})
    end
end

%% % Temperature triggered by trial start
events = animal.time(trials.start(find(trials.cooling~=2)));
events_duration = trials.end-trials.start;
figure(1006)
colors = {[0,0,0],[0,1,0],[0,0,1],[1,0,0]};
window = 4*temperature.sr;
time_axis = [-window+1:window]/temperature.sr;
sorting_field = {'time','duration'};
for j= 1:length(sorting_field)
    plt = [];
    idx_all = interp1(temperature.timestamps,[1:length(temperature.timestamps)],events,'nearest');
    nEvents = length(idx_all);
    if j == 1
        ind_sorted = 1:nEvents;
    elseif j == 2
        [~,ind_sorted] = sort(events_duration(find(trials.cooling~=2)));
    elseif j == 3
        [~,ind_sorted] = sort(phasezero_theta_freq(phasezero_speed_thresholded));
    elseif j == 4
        [~,ind_sorted] = sort(phasezero_theta_power(phasezero_speed_thresholded));
    end
    
    %     [~,ind_sorted] = sort(ripples.(sorting_field{j}));
    intervals = {(1:round(nEvents/3)); (round(nEvents/3):round(2*nEvents/3)); (2*round(nEvents/3):nEvents)};
    idx_groups = {[1:nEvents];intervals{1};intervals{2};intervals{3}};
    
    for i = 1:4
        clear psth_temperature
        startIndicies = idx_all(ind_sorted(idx_groups{i}))-window+1;
        stopIndicies = idx_all(ind_sorted(idx_groups{i}))+window;
        X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
        X = X(1:end-1);
        psth_temperature = reshape(temperature.data(X)',window*2,[]);
        psth_temperature = psth_temperature-mean(psth_temperature);
        if i == 1 && j == 1
            figure(1005)
            plot(time_axis,psth_temperature), hold on, plot(time_axis,nanmean(psth_temperature,2),'k','linewidth',2)
            title('Trial start triggered average temperature'), xlabel('Time (s)'), ylabel('Temperature (C)'), grid on, ylim([-0.08,0.08])
            plot([0,0],[-0.1,0.1],'k'),plot([-1,1],[0,0],'k')
        end
        figure(1006)
        subplot(1,2,j), hold on
        patch([time_axis,flip(time_axis)], [nanmean(psth_temperature,2)+nansem(psth_temperature')';flip(nanmean(psth_temperature,2)-nansem(psth_temperature')')]',colors{i},'EdgeColor','none','FaceAlpha',.2)
        plt(i) = plot(time_axis,nanmean(psth_temperature,2),'color',colors{i}); hold on,
    end
    xlabel('Time (s)'), ylabel('Temperature (C)'), grid on, title(['Grouped and sorted by ',sorting_field{j}]),
    if j == 1%length(sorting_field)
        legend(plt,{'All','1st third','2nd third','3rd third'})
    end
end

%% Temperature effects from single cell spiking (triggered average of temperature)
for i = 1:10%spikes.numcells
    i
    idx = round(spikes.times{i}*1250);
    idx(idx<501 | idx>numel(temperature.timestamps)-501) = [];
    temperature_psth = zeros(601,numel(idx));
    for j = 1:numel(idx)
        temperature_psth(:,j) = temperature.data(idx(j)-100:idx(j)+500);
    end
    figure, plot(temperature_psth-mean(temperature_psth)), hold on
    plot(mean(temperature_psth-mean(temperature_psth),2),'k','linewidth',2), ylim([-0.1,0.1]),title(['Unit ' num2str(i)])
end

%% Temperature effects from average firing rate of single cells (triggered average of temperature)
% Generating continues representation of the raster actvity
% Defining kernel
sig = 2000;
x_val = -2000:5:2000;
kernel = 1/(sqrt(2*pi)*sig)*exp(-((x_val).^2)/(2*sig.^2));
figure, plot(kernel)

time_bins = 0:0.01:ceil(max(spikes.spindices(:,1)));
spikes_presentation = zeros(spikes.numcells,numel(time_bins));
for i = 1:spikes.numcells
    idx = round(spikes.times{i}*100);
    spikes_presentation(i,idx) = 1;
    % Convoluting the spike times with a 250ms gaussian convolution
    spikes_presentation(i,:) = conv(spikes_presentation(i,:),kernel','same');
end
imagesc(spikes_presentation)

% Filtering the temperature to detect faster dynamic
Fc = [0.0001,0.1];
[b1,a1]=butter(2,Fc*2/100,'bandpass'); % 'high' pass filter
temperature.filt = filtfilt(b1,a1,temperature.data);
figure,
subplot(2,1,1)
plot(temperature.data)
subplot(2,1,2)
plot(temperature.filt)

%% Temperature vs behavior (trial onsets, theta)
% Getting behavior
basepath = 'Z:\Homes\voerom01\UM\Kyounghwan\Passive_probe\R2W3_10A2\R2W3_10A2_20191016';
csv_file = 'R2W3_10A2_20191016_linear001_000.csv';

session = sessionTemplate(basepath);
% session = loadSession(basepath); % Loading session info

% Loading optitrack data
optitrack = optitrack2buzcode(session);

% Loading TTL pulses
intanDig = intanDigital2buzcode(session);

% Generating lineartrack data
lineartrack = optitrack;
lineartrack.offsets = [110,-8,0];
lineartrack.rotation = +0.5; % in degrees
lineartrack.timestamps = intanDig.on{session.inputs.OptitrackTTL.channels};
lineartrack.position.x = lineartrack.position.x+lineartrack.offsets(1);
lineartrack.position.y = lineartrack.position.y+lineartrack.offsets(2);
lineartrack.position.z = lineartrack.position.z+lineartrack.offsets(3);

if ~isempty(lineartrack.rotation)
    x = lineartrack.position.x;
    y = lineartrack.position.y;
    X = x*cosd(lineartrack.rotation) - y*sind(lineartrack.rotation);
    Y = x*sind(lineartrack.rotation) + y*cosd(lineartrack.rotation);
    lineartrack.position.x = X;
    lineartrack.position.y = Y;
end
lineartrack.linearized.position = lineartrack.position.x;
figure,
plot(lineartrack.position.x,lineartrack.position.y)
% lineartrack.linearized.speed
% lineartrack.linearized.acceleration
lineartrack.boundaries.start = 10;
lineartrack.boundaries.end = 190;
lineartrack.trials = getTrials_lineartrack(lineartrack.linearized.position,lineartrack.boundaries.start,lineartrack.boundaries.end);

% Loading temperature
temperature = intan2buzcode(session,'temperature','timeseries','adc',1);

%% Control and temperature
sessions_control_misi = {'Temp_R05_20201219','Temp_R05_20201228','Temp_R05_20201229','Temp_R05_20210101'};
sessions_control_peter = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170716_172307_concat', 'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS21_180628_155921_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', ...
    'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};
% 'Peter_MS10_170317_153237_concat', 'Peter_MS10_170314_163038', 'Peter_MS10_170315_123936', 'Peter_MS10_170307_154746_concat', 'Peter_MS13_171110_163224_concat', 'Peter_MS21_180807_122213_concat', 'Peter_MS21_180719_155941_concat', 'Peter_MS21_180625_153927_concat', 'Peter_MS22_180719_122813_concat'
% 'Temp_R04_20201023','Temp_R04_20201024', 'Temp_R04_20201112','Temp_R05_20210102','Temp_R04_20201113', 'Temp_R04_20201114', 'Temp_R04_20201027',
sessions_control_peter2 = {'Peter_MS21_180719_155941_concat','Peter_MS21_180807_122213_concat','Peter_MS22_180719_122813_concat'};
sessions_all = [sessions_control_peter,sessions_control_misi];

sessions_control_nonREM = {'Temp_R05_20201219','Temp_R05_20201229','Temp_R05_20210101','Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', 'Peter_MS22_180711_112912_concat'};
% sessions_all = sessions_control_nonREM;
% sessions_control_peter: 1:14,16,17 complete,
% sessions_all = sessions_control_peter2

for i = 1:numel(sessions_all)
    sessionID = i;
    basename = sessions_all{sessionID};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp';
        temperature.timestamps = temperature.time';
    end
    % [b1, a1] = butter(3, 2/temperature.sr*2, 'low');
    % temperature.filter = filtfilt(b1, a1, temperature.data);
    
    if ~isempty(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)})
        cooling_interval = [str2num(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)}),str2num(sessions_metadata.Cooling_end{strcmp(sessions_metadata.Session_name,basename)})];
    else
        cooling_interval = [0,temperature.timestamps(end)];
    end
    control_interval = [1,cooling_interval(1)-100;cooling_interval(2)+100,temperature.timestamps(end)];
    fig5 = figure('name',basename);
    subplot(2,1,1)
    plot(temperature.timestamps,temperature.data), title(basename), hold on
    % plot(temperature.timestamps,temperature.filter), hold on
    
    if ~isempty(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)})
        session.channelTags.Ripple.channels = str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
    end
    
    if exist(fullfile(basepath,[basename,'.ripples.events.mat']))
        disp('Loading existing ripples file')
        load(fullfile(basepath,[basename,'.ripples.events.mat']))
        if ~isfield(ripples,'flagged')
            disp('Detecting ripples again... hold on!')
            detectRipples = true;
        else
            detectRipples = false;
        end
    else
        detectRipples = true;
    end
    if detectRipples
        % ripples = ce_FindRipples(session,'thresholds', [45 70]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
        ripples = ce_FindRipples(session,'thresholds', [18 45]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
        
    end
    if ~isfield(ripples,'flagged')
        ripples.flagged = [];
    end   
    
    % ripples2 = ce_FindRipples(session,'thresholds', [1 2], 'passband', [80 240], 'EMGThresh', 1, 'durations', [20 150],'saveMat',true);
    figure,plot(ripples.peakNormedPower,'.'), hold on, plot(ripples.flagged,ripples.peakNormedPower(ripples.flagged),'or')
    
    ripples.peaks(ripples.flagged) = [];
    ripples.timestamps(ripples.flagged,:) = [];
    ripples.peakNormedPower(ripples.flagged) = [];
    
    % Filtering by non-REM state
    SleepState = loadStruct('SleepState','states','session',session);
    idx = InIntervals(ripples.peaks,SleepState.ints.NREMstate);
    ripples.peaks = ripples.peaks(idx);
    ripples.timestamps = ripples.timestamps(idx,:);
    ripples.peakNormedPower = ripples.peakNormedPower(idx);
    
    ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);
    
    idx = setdiff(find((ripples.peaks <= cooling_interval(1) | ripples.peaks >= cooling_interval(2)) & ripples.temperature > 32 & ripples.temperature < 38),ripples.flagged);
    
    ripples.peaks = ripples.peaks(idx);
    ripples.timestamps = ripples.timestamps(idx,:);
    ripples.peakNormedPower = ripples.peakNormedPower(idx);
    ripples.temperature = ripples.temperature(idx);
    
    lfp = bz_GetLFP(session.channelTags.Ripple.channels-1,'basepath',basepath,'basename',basename);
    ripfiltlfp = bz_Filter(lfp.data,'passband',[110,180],'filter','fir1');
    [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
    % PlotRippleStats(ripples,maps,data,stats)
    ripples.peakFrequency = data.peakFrequency;
    ripples.duration = data.duration*1000;
    ripples.peakAmplitude = data.peakAmplitude/1000;
    ripples.ISIs = diff(ripples.peaks); ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
    t_minutes = [0:ceil(max(ripples.peaks)/30)];
    ripple_rate = histcounts(ripples.peaks/30,t_minutes)/30;
    ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks/30);
    
    figure(fig5)
    plot(control_interval',[33,33],'-k','linewidth',2), xlabel('Time (sec)'), ylabel('Rippe frequency (Hz)')
    subplot(2,1,2)
    plot(ripples.peaks,ripples.peakFrequency,'.'), hold on
    plot(control_interval',[120,120],'-k','linewidth',2), xlabel('Time (sec)'), ylabel('Rippe frequency (Hz)')
    saveas(fig5,fullfile([local_path,'/RipplesVsTemperature/Figures'],[basename,'.temperature_control.png']))
    
    fig6 = figure('name',basename);
    control_stats = compareScatter(ripples);
    saveas(fig6,fullfile([local_path, '/RipplesVsTemperature/Figures'],[basename,'.ripples_control_Stats1.png']))
    
    %     fig7 = figure('name',basename);
    %     compareState(ripples,ripples.cooling)
    %     saveas(fig7,fullfile([local_path, '/RipplesVsTemperature/Figures'],[basename,'.ripples_control_Stats2.png']))
    ripples_control = ripples;
    saveStruct(ripples_control,'events','session',session);
    save(fullfile(basepath,[basename,'.control_stats.mat']), 'control_stats');
    drawnow
    %     close all
end

% BATCH
batchData = {};
k = 1;
for i = 1:numel(sessions_all)
    sessionID = i;
    basename = sessions_all{sessionID};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    load(fullfile(basepath,[basename,'.control_stats.mat']), 'control_stats');
    
    control_stats
    fieldsToAnalize = fieldnames(control_stats);
    for j = 1:numel(fieldsToAnalize)
        batchData.(fieldsToAnalize{j}).slope(k) = control_stats.(fieldsToAnalize{j}).slope;
        batchData.(fieldsToAnalize{j}).R(k) = control_stats.(fieldsToAnalize{j}).R;
        batchData.(fieldsToAnalize{j}).P(k) = control_stats.(fieldsToAnalize{j}).P;
    end
    k=k+1;
end

fig5 = figure('name','Slope');
fig6 = figure('name','R');
fig7 = figure('name','P');
fieldsToAnalize = fieldnames(control_stats);
fieldsToAnalize = setdiff(fieldsToAnalize,{'cooling','timestamps','peaks','detectorinfo','noise','temperature'});
for j = 1:numel(fieldsToAnalize)
    figure(fig5)
    subplot(2,3,j)
    raincloud_plot(batchData.(fieldsToAnalize{j}).slope,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('Slope')
    figure(fig6)
    subplot(2,3,j)
    raincloud_plot(batchData.(fieldsToAnalize{j}).R,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('R')
    %     plot(1,batchData.(fieldsToAnalize{j}).R,'o'); hold on, title(fieldsToAnalize{j}), xlabel('R')
    figure(fig7)
    subplot(2,3,j)
    P1 = log10(batchData.(fieldsToAnalize{j}).P);
    P1(P1==-Inf)=-20;
    if any(P1>-20)
        raincloud_plot(P1,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('P')
    else
        plot(1,P1,'o'), hold on, title(fieldsToAnalize{j}), xlabel('P')
    end
    
end
saveas(fig5,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_Slope.png'))
saveas(fig6,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_R.png'))
saveas(fig7,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_P.png'))

%% % Ripple triggered average of the temperature NEW
for i = 1:numel(sessions_all) % SleepScoreMaster TODO 20:21
    basename = sessions_all{i};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    temperature = loadStruct('temperature','timeseries','session',session);
    temperature.sr
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp';
        temperature.timestamps = temperature.time';
        temperature.data = temperature.data(1:temperature.sr:end);
        temperature.timestamps = temperature.timestamps(1:temperature.sr:end);
    end
    if isfield(temperature,'time')
        temperature.time = temperature.time(1:temperature.sr:end);
        temperature.temp = temperature.temp(1:temperature.sr:end);
    end
    temperature.sr = 1;
    
    ripples = loadStruct('ripples','events','session',session);
    ripples.peaks(ripples.flagged) = [];
    
    ripples.timestamps(ripples.flagged,:) = [];
    ripples.peakNormedPower(ripples.flagged) = [];
    ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);
    
    [fig8] = rippleTriggeredAverage_temperature2(ripples,temperature,basename);
    saveas(fig8,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.rippleTriggeredAverage_temperature3.png']))
    %     saveas(fig9,fullfile(local_path, 'RipplesVsTemperature/Figures',[basename,'.rippleTriggeredAverage_temperature4.png']))
end

%% Waveforms : amplitude, width, asymmetry
% Required the dat file for waveform extraction
sessions_control_peter = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170716_172307_concat', 'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS21_180628_155921_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', ...
    'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};
sessions_all = sessions_control_peter;
temperature_batch = [];
for ii = 5:numel(sessions_all)
    basename = sessions_all{ii};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    %     sessions = db_load_sessions('sessionName',basename);
    %     session = sessions{1};
    %     basepath = session.general.basePath;
    cd(basepath)
    disp(['*** Calculating waveform metrics: ', basename,'. ', num2str(k),'/', num2str(length(sessions_all)),' sessions'])
    session = loadSession(basepath,basename); % Loading session info
    disp('Loading spikes')
    spikes = loadSpikes('session',session); % Loading spikes
    disp('Extracting waveforms')
    spikes = getWaveformsFromDat(spikes,session,'nPull',1000,'keepWaveforms_filt',true); % Pulling waveforms
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
        temperature.timestamps = temperature.time;
    end
    temperature.bins = 32:0.2:37.5;
    
    spikes.temperature = cellfun(@(X) interp1(temperature.timestamps,temperature.data,X),spikes.times,'UniformOutput',false);
    j = 0;
    temperature_means_all = [];
    for i = 1:spikes.numcells
        if j == 0 | j == 25
            figure;
            j = 0;
        end
        j = j+1;
        spike_metrics = permute(range(spikes.waveforms.filt{i}(spikes.maxWaveformCh1(i),:,:),2),[3,2,1]);
        [~,ia,~] = intersect(spikes.times{i},spikes.waveforms.times{i});
        spikes_temperature = spikes.temperature{i}(ia);
        
        subplot(5,5,j)
        plot(spikes_temperature,spike_metrics,'.'), hold on
        temperature_means = [];
        for k = 1:numel(temperature.bins)-1
            temperature_means(k) = nanmean(spike_metrics(find(spikes_temperature>temperature.bins(k) & spikes_temperature < temperature.bins(k+1))));
        end
        plot(temperature.bins(1:end-1)+0.1,temperature_means,'r'),xlim([temperature.bins(1),temperature.bins(end)]),%ylim([0,10])
        temperature_means_all(i,:) = temperature_means./nanmean(temperature_means);
    end
    temperature_batch(ii).temperature_means_all = temperature_means_all;
    figure, plot(temperature.bins(1:end-1),(temperature_means_all)'), hold on
    plot(temperature.bins(1:end-1),nanmean((temperature_means_all))','-k','linewidth',2)
    % errorbar(temperature.bins(1:end-1),nanmedian((temperature_means_all))',nanstd((temperature_means_all))')
    xlabel('Brain temperature (degrees C)'), ylabel('Spike amplitude'), set(gca, 'YScale', 'log')
end

%% Spikes vs temperature
% Temporal dynamical changes: Firing rates, burstiness, CV2
sessions_control_peter = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170716_172307_concat', 'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS21_180628_155921_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', ...
    'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};
sessions_all = sessions_control_peter;
temperature_batch = [];
for ii = 1:numel(sessions_all)
    basename = sessions_all{ii};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    %     basepath = fullfile('Z:\Homes\peterp03\IntanData\',animal_subject,basename);
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    disp(['*** Calculating waveform metrics: ', basename,'. ', num2str(ii),'/', num2str(length(sessions_all)),' sessions'])
    session = loadSession(basepath,basename); % Loading session info
    disp('Loading spikes')
    spikes = loadSpikes('session',session); % Loading spikes
    
    cell_metrics = loadCellMetrics('session',session); % Loading spikes
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
        temperature.timestamps = temperature.time;
    end
    temperature.bins = 34:0.2:38;
    temperature.data(temperature.data<34) = nan;
    spikes.temperature = cellfun(@(X) interp1(temperature.timestamps,temperature.data,X),spikes.times,'UniformOutput',false);
    spikes.ISIs = cellfun(@diff,spikes.times,'UniformOutput',false);
    spikes.meanISI = cellfun(@(X) [(X(1:end-1)+X(2:end))./2;0],spikes.ISIs,'UniformOutput',false);
    spikes.meanInstantRate = cellfun(@(X) [1./((X(1:end-1)+X(2:end))./2);0],spikes.ISIs,'UniformOutput',false);
    spikes.meanInstantRateSmooth = cellfun(@(X) [1./movmean(((X(1:end-1)+X(2:end))./2),10);0],spikes.ISIs,'UniformOutput',false);
    spikes.CV2 = cellfun(@(X) [2.*abs(X(2:end)-X(1:end-1))./(X(2:end)+X(1:end-1));0],spikes.ISIs ,'UniformOutput',false);
    
    figure, plot(temperature.timestamps,temperature.data)
    
    fields2process = {'meanInstantRateSmooth'}; % 'amplitudes','meanISI','CV2'
    for iii = 1:numel(fields2process)
        %         figure
        j = 0;
        temperature_means_all = [];
        for i = 1:spikes.numcells
            %             if j == 0 | j == 25
            %                 figure;
            %                 j = 0;
            %             end
            j = j+1;
            spike_metrics = spikes.(fields2process{iii}){i};
            %             subplot(5,5,j)
            %             plot(spikes.temperature{i}(1:numel(spike_metrics)),spike_metrics,'.'), hold on
            temperature_means = [];
            for k = 1:numel(temperature.bins)-1
                temperature_means(k) = nanmean(spike_metrics(find(spikes.temperature{i}(1:end-1)>temperature.bins(k) & spikes.temperature{i}(1:end-1) < temperature.bins(k+1))));
            end
            %             plot(temperature.bins(1:end-1)+0.1,temperature_means,'r'),xlim([temperature.bins(1),temperature.bins(end)]),%ylim([0,10])
            temperature_means_all(i,:) = temperature_means;
        end
        temperature_batch{ii} = temperature_means_all';
        
        % KiloSort amplitude
        figure, plot(temperature.bins(1:end-1),(temperature_means_all)'), hold on
        plot(temperature.bins(1:end-1),nanmean((temperature_means_all))','-k','linewidth',2)
        % errorbar(temperature.bins(1:end-1),nanmedian((temperature_means_all))',nanstd((temperature_means_all))')
        xlabel('Brain temperature (degrees C)'), ylabel(fields2process{iii}), %set(gca, 'YScale', 'log')
        title(basename)
    end
end

temperature_batch2 = [temperature_batch{:}];
figure, plot(temperature.bins(1:end-1),temperature_batch2), hold on
plot(temperature.bins(1:end-1),nanmean(temperature_batch2,2),'-k','linewidth',2)
xlabel('Brain temperature (degrees C)'), ylabel(fields2process{iii}), set(gca, 'YScale', 'log')


%% Average ripple spectrogram highest vs lowest temperature events

sessions_control_nonREM = {'Temp_R05_20201229','Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', 'Peter_MS22_180711_112912_concat'};
sessions_all = sessions_control_nonREM;

freqband = 'ripple';
switch freqband
    case 'theta'
        freqlist = [4:0.025:10];
        Fpass = [4,10];
        sr_theta = animal.sr;
        caxis_range = [0,1.];
    case 'gamma'
        freqlist = [30:5:100];
        Fpass = [30,100];
        sr_theta = 400;
        caxis_range = [0,0.45];
    case 'ripple'
        freqlist = [80:2:220];
        Fpass = [80,220];
        sr_theta = 1250;
        caxis_range = [0,0.45];
end
running_window = 20;
theta_samples_pre = 80;
theta_samples_post = 80;

% basename = 'Peter_MS22_180711_112912_concat';

ripple_spectrograms1 = [];
ripple_spectrograms2 = [];
for ii = numel(sessions_all)
    basename = sessions_all{ii};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    %     basepath = fullfile('Z:\Homes\peterp03\IntanData\',animal_subject,basename);
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    disp(['*** Calculating waveform metrics: ', basename,'. ', num2str(ii),'/', num2str(length(sessions_all)),' sessions'])
    session = loadSession(basepath,basename); % Loading session info
    
    load(fullfile(basepath,[basename,'.ripples.events.mat']))
    load(fullfile(basepath,[basename,'.temperature.timeseries.mat']))
    %  load(fullfile(basepath,[basename,'.ripples_cooling.events.mat']))
    signal = LoadBinary([basename '.lfp'],'nChannels',session.extracellular.nChannels,'precision','int16','frequency',session.extracellular.srLfp);
    signal = session.extracellular.leastSignificantBit * double(signal(:,ripples.detectorinfo.detectionchannel+1));
%     signal = session.extracellular.leastSignificantBit * double(LoadBinary([basename '.lfp'],'nChannels',session.extracellular.nChannels,'channels',ripples.detectorinfo.detectionchannel,'precision','int16','frequency',session.extracellular.srLfp));
    
    window_time = [-theta_samples_pre:theta_samples_post]/sr_theta;
    window_stim = [theta_samples_pre-sr_theta/2:theta_samples_pre+sr_theta/2];
    window_prestim = [1:theta_samples_pre-sr_theta/2-1];
    Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)];
    [btheta,atheta] = butter(3,Wn_theta);
    signal_filtered = filtfilt(btheta,atheta,signal);
    %[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
    %wt2 = abs(wt)'; clear wt
    wt = spectrogram(signal_filtered,running_window,running_window-1,freqlist,sr_theta);
    wt2 = [zeros(length(freqlist),running_window/2-1),abs(wt), zeros(length(freqlist),running_window/2)]; clear wt
    
    % Preparing ripples struct
    if ~isempty(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)})
        cooling_interval = [str2num(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)}),str2num(sessions_metadata.Cooling_end{strcmp(sessions_metadata.Session_name,basename)})];
    else
        cooling_interval = [0,temperature.timestamps(end)];
    end
    
    if isfield(temperature.states,'Cooling')
        temperature.states.cooling = temperature.states.Cooling;
    end
    
    if isfield(ripples,'flagged')
        ripples.peaks(ripples.flagged) = [];
        ripples.timestamps(ripples.flagged,:) = [];
        ripples.peakNormedPower(ripples.flagged) = [];
    end
    
    idx = find(InIntervals(ripples.peaks,temperature.states.cooling));
    ripples.peaks(idx) = [];
    ripples.timestamps(idx,:) = [];
    ripples.peakNormedPower(idx) = [];
    
    ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);
    
    % Getting the lowest and highest temperature ripples
    [tem,idx] = sort(ripples.temperature);
    ripples.low_temp = idx(1:200);
    ripples.high_temp = idx(end-200:end);
    opto_peaks1 = round(sr_theta*ripples.peaks(ripples.low_temp));
    theta_triggered1 = [];
    lfp_average1 = [];
    for i = 1:length(opto_peaks1)
        theta_triggered1(:,:,i) = wt2(:,opto_peaks1(i)-theta_samples_pre:opto_peaks1(i)+theta_samples_post);
        lfp_average1(:,i) = signal_filtered(opto_peaks1(i)-theta_samples_pre:opto_peaks1(i)+theta_samples_post);
    end
    
    % Cooling
    opto_peaks2 = round(sr_theta*ripples.peaks(ripples.high_temp));
    theta_triggered2 = [];
    lfp_average2 = [];
    for i = 1:length(opto_peaks2)
        theta_triggered2(:,:,i) = wt2(:,opto_peaks2(i)-theta_samples_pre:opto_peaks2(i)+theta_samples_post);
        lfp_average2(:,i) = signal_filtered(opto_peaks2(i)-theta_samples_pre:opto_peaks2(i)+theta_samples_post);
    end
    
    % Figure for all sessions
    figure,
    subplot(2,2,1)
    imagesc(window_time,freqlist,mean(theta_triggered1,3)), set(gca,'YDir','normal'), title(['Control ' basename],'interpreter','none'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    ptemp = mean(theta_triggered1,3);
    
    ripple_spectrograms1(:,:,ii) =  mean(theta_triggered1,3);
    ripple_spectrograms2(:,:,ii) =  mean(theta_triggered2,3);
    
    hold on, %plot([-100:100]/200,-cos([-100:100]/100*pi+pi)+freqlist(1)+1,'w'), %plot([-0.5,-0.5;0.5,0.5]',[freqlist(1),freqlist(end)],'--w'), %caxis(caxis_range)
    xlim([-0.04,0.04])
    
    subplot(2,2,2)
    imagesc(window_time,freqlist,mean(theta_triggered2,3)), set(gca,'YDir','normal'), title(['Cooling '],'interpreter','none'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    hold on, %plot([-100:100]/200,-cos([-100:100]/100*pi+pi)+freqlist(1)+1,'w'), %plot([-0.5,-0.5;0.5,0.5]',[freqlist(1),freqlist(end)],'--w'), %caxis(caxis_range)
    xlim([-0.04,0.04])
    subplot(2,2,3)
    ptemp1 = mean(theta_triggered1,3);
    [~, idx1] = max(ptemp1(:,80));
    ptemp2 = mean(theta_triggered2,3);
    [~, idx2] = max(ptemp2(:,80));
    plot(ptemp1(:,80),freqlist,'r'), hold on, plot(ptemp1(idx1,80),freqlist(idx1),'xk')
    plot(ptemp2(:,80),freqlist,'b'), plot(ptemp2(idx2,80),freqlist(idx2),'xk')
    legend({'Control', 'Heating'})
    subplot(2,2,4)
    plot(window_time,mean(lfp_average1,2),'-r'), hold on
    plot(window_time,mean(lfp_average2,2),'-b')
    xlim([-0.04,0.04])
    drawnow
    clear wt2
    % end
    

    ripples_intervals_low_temp = getIntervalsFromDat(ripples.peaks(ripples.low_temp),session,'nPull',Inf);
    ripples_intervals_high_temp = getIntervalsFromDat(ripples.peaks(ripples.high_temp),session,'nPull',Inf);

    figure,
    plot(ripples_intervals_low_temp.timeInterval_all,ripples_intervals_low_temp.filtIntervals_all(ripples.detectorinfo.detectionchannel1,:),'b'); hold on
    plot(ripples_intervals_high_temp.timeInterval_all,ripples_intervals_high_temp.filtIntervals_all(ripples.detectorinfo.detectionchannel1,:),'r');
    xlim([-20,20])

end

%% Plotting average spectrogram across group

ripple_spectrograms1_mean = mean(ripple_spectrograms1,3);
ripple_spectrograms2_mean = mean(ripple_spectrograms2,3);
 % Figure for all sessions
    figure,
    subplot(2,2,1)
    imagesc(window_time,freqlist,ripple_spectrograms1_mean), set(gca,'YDir','normal'), title(['Control ' basename],'interpreter','none'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    
    hold on, %plot([-100:100]/200,-cos([-100:100]/100*pi+pi)+freqlist(1)+1,'w'), %plot([-0.5,-0.5;0.5,0.5]',[freqlist(1),freqlist(end)],'--w'), %caxis(caxis_range)
    xlim([-0.02,0.02])
    ylim([100,200])
    subplot(2,2,2)
    imagesc(window_time,freqlist,ripple_spectrograms2_mean), set(gca,'YDir','normal'), title(['Cooling '],'interpreter','none'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    hold on, %plot([-100:100]/200,-cos([-100:100]/100*pi+pi)+freqlist(1)+1,'w'), %plot([-0.5,-0.5;0.5,0.5]',[freqlist(1),freqlist(end)],'--w'), %caxis(caxis_range)
    xlim([-0.02,0.02])
    ylim([100,200])
    
    subplot(2,2,3)
    ptemp1 = mean(theta_triggered1,3);
    [~, idx1] = max(ptemp1(:,80));
    ptemp2 = mean(theta_triggered2,3);
    [~, idx2] = max(ptemp2(:,80));
    plot(ptemp1(:,80),freqlist,'r'), hold on, plot(ptemp1(idx1,80),freqlist(idx1),'xk')
    plot(ptemp2(:,80),freqlist,'b'), plot(ptemp2(idx2,80),freqlist(idx2),'xk')
    legend({'Control', 'Heating'})
    subplot(2,2,4)
    plot(window_time,mean(lfp_average1,2),'-r'), hold on
    plot(window_time,mean(lfp_average2,2),'-b')
    xlim([-0.02,0.02])
    drawnow
    clear wt2
    % end
    
%% Awake ripples
for i = 1:numel(sessions_all)
    % basename = 'Peter_MS13_171130_121758_concat';
    % basename = 'Peter_MS13_171129_105507_concat';
    % basename = 'Peter_MS22_180628_120341_concat';
    
    basename = sessions_all{i};
    disp(' ')
    disp([num2str(i),'/',num2str(length(sessions_all)),': ', basename])
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    
    session = loadSession(basepath);
    load(fullfile(basepath,[basename,'.ripples.events.mat']));
    load(fullfile(basepath,[basename,'.temperature.timeseries.mat']));
    
    if isfield(temperature.states,'Cooling')
        temperature.states.cooling = temperature.states.Cooling;
    end
    
    if isfield(ripples,'flagged')
        ripples.peaks(ripples.flagged) = [];
        ripples.timestamps(ripples.flagged,:) = [];
        ripples.peakNormedPower(ripples.flagged) = [];
    end
    
    idx = find(InIntervals(ripples.peaks,temperature.states.cooling));
    ripples.peaks(idx) = [];
    ripples.timestamps(idx,:) = [];
    ripples.peakNormedPower(idx) = [];
    
%     % Filtering by non-REM state
    SleepState = loadStruct('SleepState','states','session',session);
    idx = InIntervals(ripples.peaks,SleepState.ints.WAKEstate);
    ripples.peaks = ripples.peaks(idx);
    ripples.timestamps = ripples.timestamps(idx,:);
    ripples.peakNormedPower = ripples.peakNormedPower(idx);
    
    idx2 = find(InIntervals(temperature.timestamps,temperature.states.cooling));
    temperature.timestamps(idx2) = [];
    temperature.data(idx2) = [];
    
    if isfield(temperature.states,'Bad')
        idx = find(InIntervals(ripples.peaks, temperature.states.Bad));
        ripples.peaks(idx) = [];
        ripples.timestamps(idx,:) = [];
        ripples.peakNormedPower(idx) = [];
    end
    
    if isfield(temperature.states,'CoolingInterval')
        disp('Removing Cooling Intervals')
        idx = find(InIntervals(ripples.peaks, temperature.states.CoolingInterval));
        ripples.peaks(idx) = [];
        ripples.timestamps(idx,:) = [];
        ripples.peakNormedPower(idx) = [];
    end
    
    lfp = bz_GetLFP(ripples.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
    ripfiltlfp = bz_Filter(lfp.data,'passband',[110,180],'filter','fir1');
    [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
    % PlotRippleStats(ripples,maps,data,stats)
    ripples.peakFrequency = data.peakFrequency;
    ripples.duration = data.duration*1000;
    ripples.peakAmplitude = data.peakAmplitude/1000;
    ripples.ISIs = diff(ripples.peaks); ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
    
    t_minutes = [0:ceil(max(ripples.peaks)/30)];
    SleepState.idx.states(SleepState.idx.states ~= 1) = 0;
    nREM_state = zeros(1,size(t_minutes,2)-1);
    for j = 1:numel(nREM_state)-1
        nREM_state(j) = sum(SleepState.idx.states((j-1)*30+1:j*30));
    end
    nREM_state(nREM_state==0)=1;
    
    ripple_rate = histcounts(ripples.peaks/30,t_minutes);
    ripple_rate = ripple_rate./nREM_state;
    %ripple_rate = ripple_rate/30;
    ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks/30);
    ripples.rate(ripples.rate>2)= 1;
    
    rem1 = rem(length(temperature.data),temperature.sr);
    temperature.timestamps1 = temperature.timestamps(1:temperature.sr:end);
    temperature.data1 = nanmean(reshape(temperature.data(1:length(temperature.data)-rem1),temperature.sr,[]));
    temperature.timestamps1 = temperature.timestamps1(1:length(temperature.data1));
    ripples.temperature = interp1(temperature.timestamps1,temperature.data1, ripples.peaks);
    
    fig1 = figure('name',basename,'position',[50,50,1200,900]); % Figure 1
    subplot(3,3,1:2)
    plot(ripples.peaks, ripples.peakFrequency,'.b'), hold on,
    plot(temperature.timestamps,10*zscore(temperature.data)+149,'r')
    % plot(temperature.timestamps(temperature.data>20),15*zscore(temperature.data(temperature.data>20))+147,'r')
    % plot(temperature.timestamps,temperature.data+100)
    plot(ripples.peaks, nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge'),'-g'),
    legend({'Ripples','Temperature (zscored & aligned)','Ripple freq running average'},'AutoUpdate','off')
    axis tight, %gridxy(cumsum(session.epochs.duration))
    title('Ripple frequency'), xlabel('Time (s)'), ylabel('Frequency (Hz)'),
    ylim([100,200])
    
    subplot(3,3,3)
    plot(ripples.temperature,ripples.peakFrequency,'.b'), hold on
    plot(ripples.temperature,nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge'),'.g')
    title('Ripple rate vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Ripple rate (Hz)')
    x = ripples.temperature;
    y1 = ripples.peakFrequency;
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
    text(0.05,1.1,['Slope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
    [R,P] = corrcoef(x,y1);
    text(0.05,1.15,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    ylim([100,200])
    x = ripples.temperature;
    y1 = nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge');
    [R,P] = corrcoef(x,y1);
    text(0.05,1.2,['R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    ylim([100,200])
    
    subplot(3,3,4:5)
    plot(ripples.peaks, ripples.rate,'.b'), hold on,
    plot(temperature.timestamps,(zscore(temperature.data)+2)/3,'r')
    plot(ripples.peaks, nanconv(ripples.rate,gausswin(gausswin_steps),'edge'),'-k'),
    axis tight,
    title('Ripple rate (Hz)'), xlabel('Time (s)'), ylabel('Frequency (Hz)'),
    
    subplot(3,3,6)
    plot(ripples.temperature,ripples.rate,'.b'), hold on
    title('Ripple rate vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Ripple rate (Hz)')
    x = ripples.temperature(~isnan(ripples.rate));
    y1 = ripples.rate(~isnan(ripples.rate));
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
    text(0.05,1.15,['Slope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
    [R,P] = corrcoef(x,y1);
    text(0.05,1.1,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    
    subplot(3,3,[7,8])
    plot(ripples.peaks, ripples.duration,'.b'), hold on,
    plot(temperature.timestamps,zscore(temperature.data)*10+60,'r')
    plot(ripples.peaks, nanconv(ripples.duration,gausswin(gausswin_steps),'edge'),'-g'),
    axis tight
    title('Ripple duration'), xlabel('Time (s)'), ylabel('Duration (ms)'),
    
    subplot(3,3,9)
    plot(ripples.temperature,ripples.duration,'.'), hold on
    plot(ripples.temperature,nanconv(ripples.duration,gausswin(gausswin_steps),'edge'),'.g')
    title('Ripple duration vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Duration (ms, log10)')
    x = ripples.temperature;
    y1 = ripples.duration;
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
    text(0.05,1.1,['Slope: ' num2str(P(1),3),' ms/degree'],'Color','k','Units','normalized')
    [R,P] = corrcoef(x,y1);
    text(0.05,1.15,['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    x = ripples.temperature;
    y1 = nanconv(ripples.duration,gausswin(gausswin_steps),'edge');
    [R,P] = corrcoef(x,y1);
    text(0.05,1.2,['R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
    
    saveas(fig1,fullfile(local_path, 'RipplesVsTemperature/Figures',[basename,'.ripplesControlStats1.png']))
    drawnow
    figure
    wake_stats = compareScatter(ripples);
    ripples_awake = ripples;
    saveStruct(ripples_awake,'events','session',session);
    save(fullfile(basepath,[basename,'.wake_stats.mat']), 'wake_stats');
    
    ripples_smooth = ripples;   
    ripples_smooth.peakFrequency = nanconv(ripples.peakFrequency,gausswin(gausswin_steps),'edge');
    ripples_smooth.rate = nanconv(ripples.rate,gausswin(gausswin_steps),'edge');
    ripples_smooth.duration = nanconv(ripples.duration,gausswin(gausswin_steps),'edge');
    ripples_awake_smooth = ripples_smooth;
    saveStruct(ripples_awake_smooth,'events','session',session);
    wake_stats_smooth = compareScatter(ripples_smooth);
    save(fullfile(basepath,[basename,'.wake_stats_smooth.mat']), 'wake_stats_smooth');
end

%%
% Figure 2 group data awake
batchData = {};
k = 1;
for i = 1:numel(sessions_all)
    sessionID = i;
    basename = sessions_all{sessionID};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    load(fullfile(basepath,[basename,'.wake_stats.mat']), 'wake_stats');
    
    fieldsToAnalize = fieldnames(wake_stats);
    for j = 1:numel(fieldsToAnalize)
        batchData.(fieldsToAnalize{j}).slope(k) = wake_stats.(fieldsToAnalize{j}).slope;
        batchData.(fieldsToAnalize{j}).R(k) = wake_stats.(fieldsToAnalize{j}).R;
        batchData.(fieldsToAnalize{j}).P(k) = wake_stats.(fieldsToAnalize{j}).P;
    end
    k=k+1;
end

fig5 = figure('name','Slope');
fig6 = figure('name','R');
fig7 = figure('name','P');
fieldsToAnalize = fieldnames(wake_stats);
fieldsToAnalize = setdiff(fieldsToAnalize,{'cooling','timestamps','peaks','detectorinfo','noise','temperature'});
for j = 1:numel(fieldsToAnalize)
    figure(fig5)
    subplot(2,3,j)
    raincloud_plot(batchData.(fieldsToAnalize{j}).slope,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('Slope')
    figure(fig6)
    subplot(2,3,j)
    raincloud_plot(batchData.(fieldsToAnalize{j}).R,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('R')
    %     plot(1,batchData.(fieldsToAnalize{j}).R,'o'); hold on, title(fieldsToAnalize{j}), xlabel('R')
    figure(fig7)
    subplot(2,3,j)
    P1 = log10(batchData.(fieldsToAnalize{j}).P);
    P1(P1==-Inf)=-20;
    if any(P1>-20)
        raincloud_plot(P1,'box_on', 1,'box_dodge', 1,'box_dodge_amount', 0.2); hold on, title(fieldsToAnalize{j}), xlabel('P')
    else
        plot(1,P1,'o'), hold on, title(fieldsToAnalize{j}), xlabel('P')
    end    
end
saveas(fig5,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_awake_Slope.png'))
saveas(fig6,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_awake_R.png'))
saveas(fig7,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_control_awake_P.png'))


%% Comparing nonREM with Awake

batchData_nonREM = {};
batchData_awake = {};
batchData_nonREM_smooth = {};
batchData_awake_smooth = {};
k = 1;

fieldsToAnalize = {'peakFrequency','duration','rate','peakAmplitude'};
for i = 1:numel(sessions_all)
    sessionID = i;
    basename = sessions_all{sessionID};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    load(fullfile(basepath,[basename,'.wake_stats.mat']), 'wake_stats');
    load(fullfile(basepath,[basename,'.nonREM_stats.mat']), 'nonREM_stats');
    
    ripples_nonREM = loadStruct('ripples_nonREM','events','session',session);
    ripples_awake = loadStruct('ripples_awake','events','session',session);    
    
    % Smoothed
    load(fullfile(basepath,[basename,'.wake_stats_smooth.mat']), 'wake_stats_smooth');
    load(fullfile(basepath,[basename,'.nonREM_stats_smooth.mat']), 'nonREM_stats_smooth');
    
    ripples_nonREM_smooth = loadStruct('ripples_nonREM_smooth','events','session',session);
    ripples_awake_smooth = loadStruct('ripples_awake_smooth','events','session',session);
    
%     fieldsToAnalize = fieldnames(wake_stats);
    for j = 1:numel(fieldsToAnalize)
        batchData_nonREM.(fieldsToAnalize{j}).mean(k) = mean(ripples_nonREM.(fieldsToAnalize{j}));
        batchData_nonREM.(fieldsToAnalize{j}).median(k) = median(ripples_nonREM.(fieldsToAnalize{j}));
        batchData_nonREM.(fieldsToAnalize{j}).R(k) = nonREM_stats.(fieldsToAnalize{j}).R;
        
        batchData_awake.(fieldsToAnalize{j}).mean(k) = mean(ripples_awake.(fieldsToAnalize{j}));
        batchData_awake.(fieldsToAnalize{j}).median(k) = median(ripples_awake.(fieldsToAnalize{j}));
        batchData_awake.(fieldsToAnalize{j}).R(k) = wake_stats.(fieldsToAnalize{j}).R;
        
        % Smoothing
        batchData_nonREM_smooth.(fieldsToAnalize{j}).mean(k) = mean(ripples_nonREM_smooth.(fieldsToAnalize{j}));
        batchData_nonREM_smooth.(fieldsToAnalize{j}).median(k) = median(ripples_nonREM_smooth.(fieldsToAnalize{j}));
        batchData_nonREM_smooth.(fieldsToAnalize{j}).R(k) = nonREM_stats_smooth.(fieldsToAnalize{j}).R;
        
        batchData_awake_smooth.(fieldsToAnalize{j}).mean(k) = mean(ripples_awake_smooth.(fieldsToAnalize{j}));
        batchData_awake_smooth.(fieldsToAnalize{j}).median(k) = median(ripples_awake_smooth.(fieldsToAnalize{j}));
        batchData_awake_smooth.(fieldsToAnalize{j}).R(k) = wake_stats_smooth.(fieldsToAnalize{j}).R;
    end
    k=k+1;
end

fig5 = figure('name','mean (normal)');
fig6 = figure('name','median (normal)');
fig7 = figure('name','r-values (normal)');
% fieldsToAnalize = fieldnames(wake_stats);
% fieldsToAnalize = setdiff(fieldsToAnalize,{'cooling','timestamps','peaks','detectorinfo','noise','temperature'});
for j = 1:numel(fieldsToAnalize)
    figure(fig5) % Mean
    subplot(2,3,j)
    plot(1,batchData_nonREM.(fieldsToAnalize{j}).mean,'.k','markersize',20), hold on
    plot(2,batchData_awake.(fieldsToAnalize{j}).mean,'.r','markersize',20)
    plot([1; 2],[batchData_nonREM.(fieldsToAnalize{j}).mean; batchData_awake.(fieldsToAnalize{j}).mean],'-k' )
    % Error bars
    plot_errorbar(0.75,batchData_nonREM.(fieldsToAnalize{j}).mean)
    plot_errorbar(2.25,batchData_awake.(fieldsToAnalize{j}).mean)
      
    [h,p] = kstest2(batchData_nonREM.(fieldsToAnalize{j}).mean, batchData_awake.(fieldsToAnalize{j}).mean);
    text(0.05,-0.18,['KS-p=' num2str(p,3)],'Color','k','Units','normalized')
    text(0,0.9,['mean=' num2str(mean(batchData_nonREM.(fieldsToAnalize{j}).mean, 'omitnan'),3)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData_awake.(fieldsToAnalize{j}).mean, 'omitnan'),3)],'Color','k','Units','normalized','HorizontalAlignment', 'right')
    xlim([0.5 2.5]), xticks([1,2]), xticklabels({'nonREM','Awake'})
    title(fieldsToAnalize{j})
    
    figure(fig6) % Median
    subplot(2,3,j)
    plot(1,batchData_nonREM.(fieldsToAnalize{j}).median,'.k','markersize',20), hold on
    plot(2,batchData_awake.(fieldsToAnalize{j}).median,'.r','markersize',20)
    % Error bars
    plot_errorbar(0.75,batchData_nonREM.(fieldsToAnalize{j}).median)
    plot_errorbar(2.25,batchData_awake.(fieldsToAnalize{j}).median)
    plot([1; 2],[batchData_nonREM.(fieldsToAnalize{j}).median; batchData_awake.(fieldsToAnalize{j}).median],'-k' )
    [h,p] = kstest2(batchData_nonREM.(fieldsToAnalize{j}).median, batchData_awake.(fieldsToAnalize{j}).median);
    text(0.05,-0.18,['KS-p=' num2str(p,3)],'Color','k','Units','normalized')
    text(0,0.9,['mean=' num2str(mean(batchData_nonREM.(fieldsToAnalize{j}).median, 'omitnan'),3)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData_awake.(fieldsToAnalize{j}).median, 'omitnan'),3)],'Color','k','Units','normalized','HorizontalAlignment', 'right')
    xlim([0.5 2.5]), xticks([1,2]), xticklabels({'nonREM','Awake'})
    title(fieldsToAnalize{j})
    
    figure(fig7) % R-values
    subplot(2,3,j)
    plot(1,batchData_nonREM.(fieldsToAnalize{j}).R,'.k','markersize',20), hold on
    plot(2,batchData_awake.(fieldsToAnalize{j}).R,'.r','markersize',20),ylim([-1 1])
    
    % Error bars
    plot_errorbar(0.75,batchData_nonREM.(fieldsToAnalize{j}).R)
    plot_errorbar(2.25,batchData_awake.(fieldsToAnalize{j}).R)
    plot([1; 2],[batchData_nonREM.(fieldsToAnalize{j}).R; batchData_awake.(fieldsToAnalize{j}).R],'-k' )
    [h,p] = kstest2(batchData_nonREM.(fieldsToAnalize{j}).R, batchData_awake.(fieldsToAnalize{j}).R);
    text(0.05,-0.18,['KS-p=' num2str(p,3)],'Color','k','Units','normalized')
    text(0,0.9,['mean=' num2str(mean(batchData_nonREM.(fieldsToAnalize{j}).R, 'omitnan'),3)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData_awake.(fieldsToAnalize{j}).R, 'omitnan'),3)],'Color','k','Units','normalized','HorizontalAlignment', 'right')
    xlim([0.5 2.5]), xticks([1,2]), xticklabels({'nonREM','Awake'})
    title(fieldsToAnalize{j})
end

% Smoothing
fig8 = figure('name',['mean (smooth: ', num2str(gausswin_steps),')']);
fig9 = figure('name',['median (smooth: '  num2str(gausswin_steps)]);
fig10 = figure('name',['r-values (smooth: '  num2str(gausswin_steps),')']);

% fieldsToAnalize = fieldnames(wake_stats);
% fieldsToAnalize = setdiff(fieldsToAnalize,{'cooling','timestamps','peaks','detectorinfo','noise','temperature'});
for j = 1:numel(fieldsToAnalize)
    figure(fig8) % Mean
    subplot(2,3,j)
    plot(1,batchData_nonREM_smooth.(fieldsToAnalize{j}).mean,'.k','markersize',20), hold on
    plot(2,batchData_awake_smooth.(fieldsToAnalize{j}).mean,'.r','markersize',20)
    plot([1; 2],[batchData_nonREM_smooth.(fieldsToAnalize{j}).mean; batchData_awake_smooth.(fieldsToAnalize{j}).mean],'-k' )
    % Error bars
    plot_errorbar(0.75,batchData_nonREM_smooth.(fieldsToAnalize{j}).mean)
    plot_errorbar(2.25,batchData_awake_smooth.(fieldsToAnalize{j}).mean)
      
    [h,p] = kstest2(batchData_nonREM_smooth.(fieldsToAnalize{j}).mean, batchData_awake_smooth.(fieldsToAnalize{j}).mean);
    text(0.05,-0.18,['KS-p=' num2str(p,3)],'Color','k','Units','normalized')
    text(0,0.9,['mean=' num2str(mean(batchData_nonREM_smooth.(fieldsToAnalize{j}).mean, 'omitnan'),3)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData_awake_smooth.(fieldsToAnalize{j}).mean, 'omitnan'),3)],'Color','k','Units','normalized','HorizontalAlignment', 'right')
    xlim([0.5 2.5]), xticks([1,2]), xticklabels({'nonREM','Awake'})
    title(fieldsToAnalize{j})
    
    figure(fig9) % Median
    subplot(2,3,j)
    plot(1,batchData_nonREM_smooth.(fieldsToAnalize{j}).median,'.k','markersize',20), hold on
    plot(2,batchData_awake_smooth.(fieldsToAnalize{j}).median,'.r','markersize',20)
    % Error bars
    plot_errorbar(0.75,batchData_nonREM_smooth.(fieldsToAnalize{j}).median)
    plot_errorbar(2.25,batchData_awake_smooth.(fieldsToAnalize{j}).median)
    plot([1; 2],[batchData_nonREM_smooth.(fieldsToAnalize{j}).median; batchData_awake_smooth.(fieldsToAnalize{j}).median],'-k' )
    [h,p] = kstest2(batchData_nonREM_smooth.(fieldsToAnalize{j}).median, batchData_awake_smooth.(fieldsToAnalize{j}).median);
    text(0.05,-0.18,['KS-p=' num2str(p,3)],'Color','k','Units','normalized')
    text(0,0.9,['mean=' num2str(mean(batchData_nonREM_smooth.(fieldsToAnalize{j}).median, 'omitnan'),3)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData_awake_smooth.(fieldsToAnalize{j}).median, 'omitnan'),3)],'Color','k','Units','normalized','HorizontalAlignment', 'right')
    xlim([0.5 2.5]), xticks([1,2]), xticklabels({'nonREM','Awake'})
    title(fieldsToAnalize{j})
    
    figure(fig10) % R-values
    subplot(2,3,j)
    plot(1,batchData_nonREM_smooth.(fieldsToAnalize{j}).R,'.k','markersize',20), hold on
    plot(2,batchData_awake_smooth.(fieldsToAnalize{j}).R,'.r','markersize',20)
    
    % Error bars
    plot_errorbar(0.75,batchData_nonREM_smooth.(fieldsToAnalize{j}).R)
    plot_errorbar(2.25,batchData_awake_smooth.(fieldsToAnalize{j}).R),ylim([-1 1])
    plot([1; 2],[batchData_nonREM_smooth.(fieldsToAnalize{j}).R; batchData_awake_smooth.(fieldsToAnalize{j}).R],'-k' )
    [h,p] = kstest2(batchData_nonREM_smooth.(fieldsToAnalize{j}).R, batchData_awake_smooth.(fieldsToAnalize{j}).R);
    text(0.05,-0.18,['KS-p=' num2str(p,3)],'Color','k','Units','normalized')
    text(0,0.9,['mean=' num2str(mean(batchData_nonREM_smooth.(fieldsToAnalize{j}).R, 'omitnan'),3)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData_awake_smooth.(fieldsToAnalize{j}).R, 'omitnan'),3)],'Color','k','Units','normalized','HorizontalAlignment', 'right')
    xlim([0.5 2.5]), xticks([1,2]), xticklabels({'nonREM','Awake'})
    title(fieldsToAnalize{j})
end


%% Getting ripples from raw dat file
ripples2 = ripples;
ripples2.peaks(ripples2.flagged);
ripples_intervals = getIntervalsFromDat(ripples2.peaks,session,'nPull',1000);

figure, plot(ripples_intervals.timeInterval_all,ripples_intervals.filtIntervals_all(ripples.detectorinfo.detectionchannel1,:));
