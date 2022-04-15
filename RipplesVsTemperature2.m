%% % Ripple analysis
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
    basepath_root = 'D:\';
    local_path = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';
else
    basepath_root = 'D:\';
    local_path = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';
end


if exist([local_path 'sessions_metadata.mat'],'file')
    load([local_path 'sessions_metadata.mat'],'sessions_metadata');
else
    DOCID = '1hGU_NPMt2wPanXKROXuPlj06saA1pFTA1391w44Tl5I';
    session_temp = GetGoogleSpreadsheet(DOCID)';
    sessions_metadata = session_temp(:,2:end)';
    sessions_metadata = cell2table(sessions_metadata);
    sessions_metadata.Properties.VariableNames = session_temp(:,1);
    save([local_path 'sessions_metadata.mat'],'sessions_metadata');
end
% Temp_R05_20201219 Cooling
% Temp_R05_20201228 Cooling
% Temp_R05_20201229 Cooling
% Temp_R05_20210101 Cooling
% Temp_R05_20210102 Cooling
% Temp_R05_20210129 Cooling
% Temp_R05_20210130 Heating 
% Temp_R05_20210130_overnight Cooling
% Temp_R05_20210131 Heating
% Temp_R07_20210215 Cooling/Heating
% Temp_R07_20210216 Heating
% Temp_R07_20210217 Heating
% Temp_R07_20210219 Cooling/Heating
% Temp_R08_20210224 Heating
% Temp_R08_20210304 Cooling/Heating
% Temp_R08_20210305 Cooling/Heating
% Temp_R08_20210306 Heating
% Temp_R08_20210307 Cooling
% Temp_R09_20210404 Heating
% Temp_R09_20210407 Heating
sessions_cooling = {'Temp_R05_20201219','Temp_R05_20201228','Temp_R05_20201229','Temp_R05_20210101','Temp_R05_20210102','Temp_R05_20210129',...
    'Temp_R05_20210130_overnight','Temp_R07_20210215','Temp_R07_20210219','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210307'};
sessions_heating = {'Temp_R05_20210130','Temp_R05_20210131','Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217','Temp_R07_20210219',...
    'Temp_R08_20210224','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210306','Temp_R09_20210404','Temp_R09_20210407'};
% sessions = {'Temp_R05_20201219','Temp_R05_20201228','Temp_R05_20201229','Temp_R05_20210101','Temp_R05_20210102','Temp_R05_20210129','Temp_R05_20210130','Temp_R05_20210130_overnight','Temp_R05_20210131'};
% sessions = {'Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217','Temp_R07_20210219','Temp_R08_20210224','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210306','Temp_R08_20210307','Temp_R09_20210404','Temp_R09_20210407'};
% Control sessions: 'Temp_R08_20210223', 'Temp_R08_20210227'
sessions_excluded = {'Temp_R04_20201114','Temp_R04_20201112','Temp_R04_20201113'}; % Too low temperature manipulations to easily process them
% No ripples: 'Temp_R04_20201027','Temp_R04_20201023','Temp_R04_20201024',
sessions_excluded2 = {}; % No temperature reading? 

%% COOLING Single session analysis of ripples and temperature 
sessions = sessions_cooling;
for i = 1:numel(sessions)
    sessionID = i;
    basename = sessions{sessionID};
    disp(['Processing ' basename])
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    [b1, a1] = butter(3, 2/temperature.sr*2, 'low');
    temperature.filter = filtfilt(b1, a1, temperature.data);
    
    if ~isempty(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)})
        cooling_interval = [str2num(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)}),str2num(sessions_metadata.Cooling_end{strcmp(sessions_metadata.Session_name,basename)})];
    else
        cooling_interval = [0,temperature.timestamps(end)];
    end
    fig1 = figure('name',basename);
    subplot(2,1,1)
    plot(temperature.timestamps,temperature.data,'k'), title(basename), hold on, axis tight
    ylabel('Temperature (°C)')
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    % New method to determine the cooling intervals/states
    if str2num(sessions_metadata.Cooling{strcmp(sessions_metadata.Session_name,basename)})== 0
        warning('NO COOLING DATA')
    else
        
    if isfield(temperature.states,'Cooling')
        Cooling = temperature.states.Cooling;
        plot(Cooling,[35,35],'-b','linewidth',2)
    end
    
    if isfield(temperature.states,'CoolingControl')
        CoolingControl = temperature.states.CoolingControl;
        plot(CoolingControl,[36,36],'-g','linewidth',2)
    end

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    if ~(isfield(session,'channelTags') && isfield(session.channelTags,'Ripple'))
        if ~isempty(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)})
            session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
        else
            warning('No ripple channel assigned')
        end
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
%         ripples = ce_FindRipples(session,'thresholds', [45 70]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',true,'absoluteThresholds',true,'show','off');
        ripples = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off','noise',18);

    end
    if ~isfield(ripples,'flagged')
        ripples.flagged = [];
    end
    
    ripples.peaks(ripples.flagged) = [];
    ripples.timestamps(ripples.flagged,:) = [];
    ripples.peakNormedPower(ripples.flagged) = [];

    idx = find(InIntervals(ripples.peaks,CoolingControl));
    ripples.peaks = ripples.peaks(idx);
    ripples.timestamps = ripples.timestamps(idx,:);
    ripples.peakNormedPower = ripples.peakNormedPower(idx);
    
%     % Filtering by non-REM state
%     SleepState = loadStruct('SleepState','states','session',session);
%     idx = find(~InIntervals(ripples.peaks,SleepState.ints.NREMstate));
%     ripples.peaks = ripples.peaks(idx);
%     ripples.timestamps = ripples.timestamps(idx,:);
%     ripples.peakNormedPower = ripples.peakNormedPower(idx);
    
    lfp = bz_GetLFP(session.channelTags.Ripple.channels-1,'basepath',basepath);
    ripfiltlfp = bz_Filter(lfp.data,'passband',[100 200],'filter','fir1');
    [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
    % PlotRippleStats(ripples,maps,data,stats)
    ripples.peakFrequency = data.peakFrequency;
    ripples.duration = data.duration*1000;
    ripples.peakAmplitude = data.peakAmplitude/1000;
    ripples.ISIs = diff(ripples.peaks); ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
    t_minutes = [0:ceil(max(ripples.peaks)/30)];
    ripple_rate = histcounts(ripples.peaks/30,t_minutes)/30;
    ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks/30);
    ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);
    ripples.cooling = InIntervals(ripples.peaks,Cooling); 
    
    subplot(2,1,2)
    plot(ripples.peaks,ripples.peakFrequency,'.'), hold on
    plot(Cooling,[120,120],'-k'), xlabel('Time (sec)'), ylabel('Rippe frequency (Hz)')
    saveas(fig1,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.temperature.png']))
    
    fig2 = figure('name',basename);
    compareScatter(ripples,ripples.cooling) 
    saveas(fig2,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesCoolingStats1.png']))
    
    fig3 = figure('name',basename);
    compareState(ripples,ripples.cooling)
    saveas(fig3,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesCoolingStats2.png']))
    ripples_cooling = ripples;
    saveStruct(ripples_cooling,'events','session',session);
    close all
    end
end

%% % Ripple heating 
sessions = sessions_heating;
for i = 1:numel(sessions)
    sessionID = i;
    basename = sessions{sessionID};
    disp(['Processing ' basename])
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    [b1, a1] = butter(3, 2/temperature.sr*2, 'low');
    temperature.filter = filtfilt(b1, a1, temperature.data);
    
%     if ~isempty(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)})
%         cooling_interval = [str2num(sessions_metadata.Cooling_start{strcmp(sessions_metadata.Session_name,basename)}),str2num(sessions_metadata.Cooling_end{strcmp(sessions_metadata.Session_name,basename)})];
%     else
%         cooling_interval = [0,temperature.timestamps(end)];
%     end
    fig1 = figure('name',basename);
    subplot(2,1,1)
    plot(temperature.timestamps,temperature.data,'k'), title(basename), hold on, axis tight

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    % New method to determine the cooling intervals/states
    if str2num(sessions_metadata.Heating{strcmp(sessions_metadata.Session_name,basename)})== 0
        warning('NO HEATING DATA')
    else
    if isfield(temperature.states,'Heating')
        Heating = temperature.states.Heating;
        plot(Heating,[36,36],'-r','linewidth',2)

    end

    if isfield(temperature.states,'HeatingControl')
        HeatingControl = temperature.states.HeatingControl;
        plot(HeatingControl,[38,38],'-g','linewidth',2)
    end
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    if ~(isfield(session,'channelTags') && isfield(session.channelTags,'Ripple'))
        if ~isempty(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)})
            session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
        else
            warning('No ripple channel assigned')
        end
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
%         ripples = ce_FindRipples(session,'thresholds', [45 70]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
        ripples = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
    end
    if ~isfield(ripples,'flagged')
        ripples.flagged = [];
    end
    
    ripples.peaks(ripples.flagged) = [];
    ripples.timestamps(ripples.flagged,:) = [];
    ripples.peakNormedPower(ripples.flagged) = [];
    
%     % Filtering by non-REM state
%     SleepState = loadStruct('SleepState','states','session',session);
%     idx = find(~InIntervals(ripples.peaks,SleepState.ints.NREMstate));
%     ripples.peaks = ripples.peaks(idx);
%     ripples.timestamps = ripples.timestamps(idx,:);
%     ripples.peakNormedPower = ripples.peakNormedPower(idx);
%     
    idx = find(InIntervals(ripples.peaks,HeatingControl));
    ripples.peaks = ripples.peaks(idx);
    ripples.timestamps = ripples.timestamps(idx,:);
    ripples.peakNormedPower = ripples.peakNormedPower(idx);

    lfp = bz_GetLFP(session.channelTags.Ripple.channels-1,'basepath',basepath);
    ripfiltlfp = bz_Filter(lfp.data,'passband',[100 200],'filter','fir1');
    [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
    ripples.peakFrequency = data.peakFrequency;
    ripples.duration = data.duration*1000;
    ripples.peakAmplitude = data.peakAmplitude/1000;
    ripples.ISIs = diff(ripples.peaks); ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
    t_minutes = [0:ceil(max(ripples.peaks)/30)];
    ripple_rate = histcounts(ripples.peaks/30,t_minutes)/30;
    ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks/30);
    ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);
    ripples.heating = InIntervals(ripples.peaks,Heating); 
    ylabel('Temperature (°C)')
    
    subplot(2,1,2)
    plot(ripples.peaks,ripples.peakFrequency,'.'), hold on
    plot(Heating,[120,120],'-k'), xlabel('Time (sec)'), ylabel('Rippe frequency (Hz)')
    saveas(fig1,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.temperature2.png']))
    
    fig2 = figure('name',basename);
    compareScatter(ripples,ripples.heating) 
    saveas(fig2,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesHeatingStats1.png']))
    
    fig3 = figure('name',basename);
    compareState(ripples,ripples.heating)
    saveas(fig3,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesHeatingStats2.png']))
    ripples_heating = ripples;
    saveStruct(ripples_heating,'events','session',session);
    close all
    end
end

%% BATCH of cooling
batchData = {};
k = 1;
colors = {};
sessions = sessions_cooling;
for i = 1:numel(sessions)
    sessionID = i;
    basename = sessions{sessionID};
%     cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
%     cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    ripples_cooling = loadStruct('ripples_cooling','events','session',session);
    batch_ripples(i).ripples_cooling = ripples_cooling;
    data = ripples_cooling;
    fieldsToAnalize = fieldnames(data);
    states = data.cooling;
    fieldsToAnalize = setdiff(fieldsToAnalize,{'cooling','timestamps','peaks','detectorinfo','noise'});
    for j = 1:numel(fieldsToAnalize)
        if all(size(data.(fieldsToAnalize{j})) == [size(data.timestamps,1),1]) && isnumeric(data.(fieldsToAnalize{j}))
            batchData.(fieldsToAnalize{j}).set1(k) = nanmean(data.(fieldsToAnalize{j})(states));
            batchData.(fieldsToAnalize{j}).set2(k) = nanmean(data.(fieldsToAnalize{j})(~states));
            [h,P] = kstest2(data.(fieldsToAnalize{j})(states),data.(fieldsToAnalize{j})(~states));
            batchData.(fieldsToAnalize{j}).h(k) = h;
            batchData.(fieldsToAnalize{j}).P(k) = P;
        end
    end
    colors{i} = '-k';
    k=k+1;
end

fieldsToAnalize = fieldnames(batchData);
fig4 = figure;
for j = 1:numel(fieldsToAnalize)
    subplot(3,3,j)
    
    for k = 1:numel(batchData.(fieldsToAnalize{j}).set1)
        plot([2;1],[batchData.(fieldsToAnalize{j}).set1(k);batchData.(fieldsToAnalize{j}).set2(k)],['-',colors{k}],'linewidth',1),hold on
        P = batchData.(fieldsToAnalize{j}).P(k);
        if P<0.003
            plot([2;1],[batchData.(fieldsToAnalize{j}).set1(k);batchData.(fieldsToAnalize{j}).set2(k)],colors{k},'linewidth',2),hold on
        elseif P<0.05
            plot([2;1],[batchData.(fieldsToAnalize{j}).set1(k);batchData.(fieldsToAnalize{j}).set2(k)],colors{k},'linewidth',1),hold on
        end
    end
    plot(2,batchData.(fieldsToAnalize{j}).set1,'or'), plot(1,batchData.(fieldsToAnalize{j}).set2,'ob')
    % Error bars
    plot_errorbar(0.75,batchData.(fieldsToAnalize{j}).set2)
    plot_errorbar(2.25,batchData.(fieldsToAnalize{j}).set1)
    text(0,0.9,['mean=' num2str(mean(batchData.(fieldsToAnalize{j}).set2, 'omitnan'),4)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData.(fieldsToAnalize{j}).set1, 'omitnan'),4)],'Color','k','Units','normalized','HorizontalAlignment', 'right')

    xlim([0.5,2.5]),ylabel(fieldsToAnalize{j})
    [p1,h1] = signrank(batchData.(fieldsToAnalize{j}).set1',batchData.(fieldsToAnalize{j}).set2');
    title(['h=',num2str(h1),', p=',num2str(p1,3)]);
end
saveas(fig4,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripplesCooling.png'));

%% BATCH of heating
batchData = {};
k = 1;
colors = {};
sessions_heating1 = {'Temp_R05_20210130','Temp_R05_20210131','Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217','Temp_R07_20210219','Temp_R08_20210224','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210306','Temp_R09_20210404','Temp_R09_20210407'};
% sessions = sessions_heating1;
sessions = sessions_heating;
for i = 1:numel(sessions)
    sessionID = i;
    basename = sessions{sessionID};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    ripples_heating = loadStruct('ripples_heating','events','session',session);
    batch_ripples(i).ripples_heating = ripples_heating;
    data = ripples_heating;
    fieldsToAnalize = fieldnames(data);
    states = data.heating;
    fieldsToAnalize = setdiff(fieldsToAnalize,{'heating','timestamps','peaks','detectorinfo','noise','time'});
    for j = 1:numel(fieldsToAnalize)
        if all(size(data.(fieldsToAnalize{j})) == [size(data.timestamps,1),1]) && isnumeric(data.(fieldsToAnalize{j}))
            batchData.(fieldsToAnalize{j}).set1(k) = nanmean(data.(fieldsToAnalize{j})(states));
            batchData.(fieldsToAnalize{j}).set2(k) = nanmean(data.(fieldsToAnalize{j})(~states));
            [h,P] = kstest2(data.(fieldsToAnalize{j})(states),data.(fieldsToAnalize{j})(~states));
            batchData.(fieldsToAnalize{j}).h(k) = h;
            batchData.(fieldsToAnalize{j}).P(k) = P;
        end
    end
    colors{i} = '-k';
    k=k+1;
end

fieldsToAnalize = fieldnames(batchData);
fig4 = figure;
for j = 1:numel(fieldsToAnalize)
    subplot(3,3,j)
    
    for k = 1:numel(batchData.(fieldsToAnalize{j}).set1)
        plot([2;1],[batchData.(fieldsToAnalize{j}).set1(k);batchData.(fieldsToAnalize{j}).set2(k)],['-',colors{k}],'linewidth',1),hold on
        
        P = batchData.(fieldsToAnalize{j}).P(k);
        if P<0.003
            plot([2;1],[batchData.(fieldsToAnalize{j}).set1(k);batchData.(fieldsToAnalize{j}).set2(k)],colors{k},'linewidth',2),hold on
        elseif P<0.05
            plot([2;1],[batchData.(fieldsToAnalize{j}).set1(k);batchData.(fieldsToAnalize{j}).set2(k)],colors{k},'linewidth',1),hold on
        end
    end
    plot(2,batchData.(fieldsToAnalize{j}).set1,'or'), plot(1,batchData.(fieldsToAnalize{j}).set2,'ob')
    % Error bars
    plot_errorbar(0.75,batchData.(fieldsToAnalize{j}).set2)
    plot_errorbar(2.25,batchData.(fieldsToAnalize{j}).set1)
    text(0,0.9,['mean=' num2str(mean(batchData.(fieldsToAnalize{j}).set2, 'omitnan'),4)],'Color','k','Units','normalized')
    text(1,0.1,['mean=' num2str(mean(batchData.(fieldsToAnalize{j}).set1, 'omitnan'),4)],'Color','k','Units','normalized','HorizontalAlignment', 'right')
    
    xlim([0.5,2.5]),ylabel(fieldsToAnalize{j})
    [p1,h1] = signrank(batchData.(fieldsToAnalize{j}).set1',batchData.(fieldsToAnalize{j}).set2');
    title(['h=',num2str(h1),', p=',num2str(p1,3)]);
end
saveas(fig4,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripplesHeating.png'));

%% Average ripple spectrogram with and without cooling
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
sessions = sessions_cooling;
% for iii = 4%:numel(sessions)
    iii = 6
    % Settint session
    sessionID = iii;
    basename = sessions{sessionID};
%     basename = 'Temp_R08_20210306'
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
%     load(fullfile(basepath,[basename,'.ripples.events.mat']))
    load(fullfile(basepath,[basename,'.ripples_cooling.events.mat']))
    ripples = ripples_cooling; %ripples_heating
%     load(fullfile(basepath,[basename,'.ripples_cooling.events.mat']))
    signal = session.extracellular.leastSignificantBit * double(LoadBinary([basename '.lfp'],'nChannels',session.extracellular.nChannels,'channels',ripples.detectorinfo.detectionchannel1,'precision','int16','frequency',session.extracellular.srLfp));
    
    running_window = 20;
    theta_samples_pre = 80;
    theta_samples_post = 80;
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
    
    % Control
    opto_peaks1 = round(sr_theta*ripples.peaks(~ripples.cooling));
    theta_triggered1 = [];
    lfp_average1 = [];
    for i = 1:length(opto_peaks1)
        theta_triggered1(:,:,i) = wt2(:,opto_peaks1(i)-theta_samples_pre:opto_peaks1(i)+theta_samples_post);
        lfp_average1(:,i) = signal_filtered(opto_peaks1(i)-theta_samples_pre:opto_peaks1(i)+theta_samples_post);
    end
    % Cooling
    opto_peaks2 = round(sr_theta*ripples.peaks(ripples.cooling));
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


%% State scoring NEW
for i = 1:numel(sessions) % SleepScoreMaster TODO 20:21
    sessionID = i;
    basename = sessions_all{sessionID};
    cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    
    % State scoring
    if ~exist(fullfile(basepath,[basename,'.SleepState.states.mat']),'file')
        SleepScoreMaster(basepath);
    end
    TheStateEditor(basename);
end

%%
temperature = StateExplorer(temperature);
saveStruct(temperature,'timeseries','session',session);
SleepScoreMaster(basepath);
TheStateEditor(basename);

%% Ripple frequency changes in non-REM sleep
sessions_control_peter = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170716_172307_concat', 'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS21_180628_155921_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', ...
    'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};
% Problems: 3
% not good state scoring: 9,10,13

sessions_cooling = {'Temp_R05_20201219','Temp_R05_20201228','Temp_R05_20201229','Temp_R05_20210101','Temp_R05_20210102','Temp_R05_20210129','Temp_R05_20210130_overnight','Temp_R07_20210215','Temp_R07_20210219','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210307'};
sessions_heating = {'Temp_R05_20210130','Temp_R05_20210131','Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217','Temp_R07_20210219','Temp_R08_20210224','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210306','Temp_R09_20210404','Temp_R09_20210407'};

sessionID = 12;
basename = sessions_heating{sessionID};
cooling = {};
animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
basepath = fullfile(basepath_root,animal_subject,basename);
cd(basepath)
session = loadSession(basepath,basename); % Loading session info
temperature = loadStruct('temperature','timeseries','session',session);
ripples = loadStruct('ripples','events','session',session);
SleepState = loadStruct('SleepState','states','session',session);


%%
if ~isfield(temperature,'data')
    temperature.data = temperature.temp;
    temperature.timestamps = temperature.time;
end

% temperature = StateExplorer(temperature);
% saveStruct(temperature,'timeseries','session',session);
% SleepScoreMaster(basepath);
% TheStateEditor(basename);
if ~isfield(ripples,'flagged')
    ripples.flagged = [];
end

ripples.peaks(ripples.flagged) = [];
ripples.timestamps(ripples.flagged,:) = [];
ripples.peakNormedPower(ripples.flagged) = [];

idx = find(InIntervals(ripples.peaks, temperature.states.cooling));
ripples.peaks(idx) = [];
ripples.timestamps(idx,:) = [];
ripples.peakNormedPower(idx) = [];

if isfield(temperature.states,'Bad')
    idx = find(InIntervals(ripples.peaks, temperature.states.Bad));
    ripples.peaks(idx) = [];
    ripples.timestamps(idx,:) = [];
    ripples.peakNormedPower(idx) = [];
end
intervals = SleepState.ints.NREMstate;
idx = find(InIntervals(ripples.peaks,intervals));
ripples.peaks = ripples.peaks(idx);
ripples.timestamps = ripples.timestamps(idx,:);
ripples.peakNormedPower = ripples.peakNormedPower(idx);

ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);

lfp = bz_GetLFP(ripples.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
ripfiltlfp = bz_Filter(lfp.data,'passband',[80 240],'filter','fir1');
[maps,data,stats] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
% PlotRippleStats(ripples,maps,data,stats)
ripples.peakFrequency = data.peakFrequency;
ripples.duration = log10(data.duration*1000);
ripples.peakAmplitude = data.peakAmplitude/1000;
ripples.ISIs = diff(ripples.peaks); 
ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
ripples_nrem = ripples;
saveStruct(ripples_nrem,'events','session',session);

% figure
% plot(ripples.temperature,ripples.peakFrequency,'.')

fig6 = figure('name',basename);
control_stats = compareScatter(ripples_nrem);
saveas(fig6,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripples_nrem_Stats1.png']))

%% Detecting ripples again

session_redo = unique([sessions_heating,sessions_cooling]);
for i = 13:numel(session_redo)
    basename = session_redo{i};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    session.channelTags.Ripple.channels = str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
    if ~isempty(sessions_metadata.Rippe_ref_ch_idx1{strcmp(sessions_metadata.Session_name,basename)})
        session.channelTags.RippleRef.channels = str2num(sessions_metadata.Rippe_ref_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
    end
    load(fullfile(basepath,[basename,'.ripples.events.mat']))
    
    ripples_new = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
    ripples_new.flagged = [];
    if isfield(session.channelTags,'RippleRef')
        session.channelTags.Ripple.channels = session.channelTags.RippleRef.channels;
        ripples_ref = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
        idx = find(InIntervals(ripples_new.peaks,ripples_ref.timestamps));
        ripples_new.flagged = unique([ripples_new.flagged;idx]);
        disp(['Ripples flagged by ref channel: ' num2str(numel(idx))])
    end
    
    if isfield(ripples,'flagged')
        flagged_events = ripples.timestamps(ripples.flagged,:);
        idx = find(InIntervals(ripples_new.peaks,flagged_events));
        ripples_new.flagged = unique([ripples_new.flagged;idx]);
        disp(['Ripples previousy flagged: ' num2str(numel(idx))])
    end
    
    ripples = ripples_new;
    saveStruct(ripples,'events','session',session);
    
    NeuroScope2('session',session,'events','ripples')
end