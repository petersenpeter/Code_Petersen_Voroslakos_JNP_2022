% Ripples vs temperature
% Ipsi vs contralateral cooling/heating

% Ipsi and contralateral recorded sessions:
% Temp_R07_20210215 Cooling/Heating
% Temp_R07_20210216 Heating
% Temp_R07_20210217 Heating
% Temp_R07_20210219 Cooling/Heating
% Temp_R08_20210224 Heating
% Temp_R08_20210304 Cooling/Heating
% Temp_R08_20210305 Cooling/Heating
% Temp_R08_20210306 Heating
% Temp_R08_20210307 Cooling

clear all

user_name = 'Peter';

if ismac
%     basepath_root = '/Volumes/Samsung_T5/GlobusDataFolder'
    basepath_root = '/Volumes/Peter_SSD_4/';
    local_path = '/Users/peterpetersen/Dropbox/Buzsakilab Postdoc/Matlab/';
elseif strcmp(user_name,'Peter')
%     basepath_root = 'Z:\Buzsakilabspace\LabShare\PeterPetersen\';
    basepath_root = 'D:\';
    local_path_pc = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';

else
    basepath_root = 'D:\';
    local_path_pc = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';
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

sessions_cooling = {'Temp_R07_20210215','Temp_R07_20210219','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210307'};
sessions_heating = {'Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217','Temp_R07_20210219','Temp_R08_20210224','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210306'};

%% Cooling
extension =  {'ripples_cooling_ipsi','ripples_cooling_contra'};
extension1 =  {'ripples_ipsi','ripples_contra'};
for i = 1:numel(sessions_cooling)
    sessionID = i;
    basename = sessions_cooling{sessionID};
    disp(['Processing ipsi-/contralateral: ' basename,' '])
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
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Cooling intervals/states
    
    if isfield(temperature.states,'Cooling')
        Cooling = temperature.states.Cooling;
        plot(Cooling',[35,35],'o-b','linewidth',1.5)
    end
    
    if isfield(temperature.states,'CoolingControl')
        CoolingControl = temperature.states.CoolingControl;
        plot(CoolingControl',[36,36],'o-g','linewidth',1.5)
    end
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    for j = 1:2
        disp(['Processing: ' extension1{j}])
        if j == 1
            % Ipsilateral
            session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
        else
            % Contralateral
            session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1_contra{strcmp(sessions_metadata.Session_name,basename)});
        end
        if ~(isfield(session,'channelTags') && isfield(session.channelTags,'Ripple'))
            if ~isempty(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)})
                session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
            else
                warning('No ripple channel assigned')
            end
        end
        if exist(fullfile(basepath,[basename,'.',extension1{j},'.events.mat']))
            disp('Loading existing ripples file')
            load(fullfile(basepath,[basename,'.',extension1{j},'.events.mat']))
            if j == 1
                ripples = ripples_ipsi;
            else
                ripples = ripples_contra;
            end
            if ~isfield(ripples,'flagged')
                disp('Detecting ripples again... hold on!')
                detectRipples = true;
            else
                detectRipples = false;
            end
        else
            detectRipples = true;
        end
        
%         detectRipples = true;
        if detectRipples
            disp(['Ripple channel: ' session.channelTags.Ripple.channels])
            if j == 1
                ripples_ipsi = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
                saveStruct(ripples_ipsi,'events','session',session);
                NeuroScope2('session',session,'events','ripples_ipsi')
            else
                ripples_contra = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
                saveStruct(ripples_contra,'events','session',session);
                NeuroScope2('session',session,'events','ripples_contra')
            end
            if j == 1
                ripples = ripples_ipsi;
            else
                ripples = ripples_contra;
            end
        end
        
%         NeuroScope2('session',session,'events',extension)
        if isfield(ripples,'flagged')
            ripples.peaks(ripples.flagged) = [];
            ripples.timestamps(ripples.flagged,:) = [];
            ripples.peakNormedPower(ripples.flagged) = [];
        end
        if isfield(ripples,'time')
            ripples = rmfield(ripples,'time');
        end
        idx = find(InIntervals(ripples.peaks,CoolingControl));
        ripples.peaks = ripples.peaks(idx);
        ripples.timestamps = ripples.timestamps(idx,:);
        ripples.peakNormedPower = ripples.peakNormedPower(idx);
        
        lfp = bz_GetLFP(session.channelTags.Ripple.channels-1,'basepath',basepath);
        ripfiltlfp = bz_Filter(lfp.data,'passband',[100 200],'filter','fir1');
        [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
        % PlotRippleStats(ripples,maps,data,stats)
        ripples.peakFrequency = data.peakFrequency;
        ripples.duration = data.duration*1000;
        ripples.peakAmplitude = data.peakAmplitude/1000;
        ripples.ISIs = diff(ripples.peaks); ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
        ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);
        
        t_minutes = [0:ceil(max(ripples.peaks)/30)];
        ripple_rate = histcounts(ripples.peaks/30,t_minutes)/30;
        ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks/30);
        ripples.cooling = InIntervals(ripples.peaks,Cooling);
        
        ylabel('Temperature (°C)')
        
        fig2 = figure('name',[basename,' ' extension{j}]);
        compareScatter(ripples,ripples.cooling)
        
        fig3 = figure('name',[basename,' ' extension{j}]);
        compareState(ripples,ripples.cooling)
        
        if j == 1
            ripples_cooling_ipsi = ripples;
            saveStruct(ripples_cooling_ipsi,'events','session',session);
            saveas(fig2,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesCoolingStats1.ripples_ipsi.png']))
            saveas(fig3,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesCoolingStats2.ripples_ipsi.png']))
        else
            ripples_cooling_contra = ripples;
            saveStruct(ripples_cooling_contra,'events','session',session);
            saveas(fig2,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesCoolingStats1.ripples_contra.png']))
            saveas(fig3,fullfile(local_path, 'RipplesVsTemperature/Figures',[basename,'.ripplesCoolingStats2.ripples_contra.png']))
        end
    end
end

%% BATCH of cooling

batchData_ipsi_contra = {};
extension =  {'ripples_cooling_ipsi','ripples_cooling_contra'};
colors = {'b','r'};
k = [];
k(1) = 1;
k(2) = 1;
for i = 1:numel(sessions_cooling)
    sessionID = i;
    basename = sessions_cooling{sessionID};
    % cooling = {};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    % cooling.threshold = str2num(sessions_metadata.Cooling_threshold{strcmp(sessions_metadata.Session_name,basename)});
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    for ii = 1:2
        data = loadStruct(extension{ii},'events','session',session);
        fieldsToAnalize = fieldnames(data);
        states = data.cooling;
        fieldsToAnalize = setdiff(fieldsToAnalize,{'heating','cooling','timestamps','peaks','detectorinfo','noise'});
        for j = 1:numel(fieldsToAnalize)
            if all(size(data.(fieldsToAnalize{j})) == [size(data.timestamps,1),1]) && isnumeric(data.(fieldsToAnalize{j}))
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).set1(k(ii)) = mean(data.(fieldsToAnalize{j})(states));
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).set2(k(ii)) = mean(data.(fieldsToAnalize{j})(~states));
                [h,P] = kstest2(data.(fieldsToAnalize{j})(states),data.(fieldsToAnalize{j})(~states));
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).h(k(ii)) = h;
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).P(k(ii)) = P;
            end
        end
        k(ii) = k(ii)+1;
    end
end

fig4 = figure;
for ii = 1:2
    batchData = batchData_ipsi_contra{ii};
    fieldsToAnalize = fieldnames(batchData);
    
    for j = 1:numel(fieldsToAnalize)
        subplot(3,3,j)
        data2process = batchData.(fieldsToAnalize{j});
        for k = 1:numel(data2process.set1)
            plot([2;1],[data2process.set1(k);data2process.set2(k)],['-',colors{ii}],'linewidth',1),hold on
            P = data2process.P(k);
            if P<0.003
                plot([2;1],[data2process.set1(k);data2process.set2(k)],colors{ii},'linewidth',2),hold on
            elseif P<0.05
                plot([2;1],[data2process.set1(k);data2process.set2(k)],colors{ii},'linewidth',1),hold on
            end
        end
        plot(2,data2process.set1,'or'), plot(1,data2process.set2,'ob')
        xlim([0.5,2.5]),ylabel(fieldsToAnalize{j})
%         [p, h, stats] = signrank(x,y,varargin);
        [p1,h1] = signrank(data2process.set1',data2process.set2');
        [p3, h3] = ranksum(data2process.set1',data2process.set2');
         [p2,h2] = kstest2(data2process.set1',data2process.set2');
        text(0.02,ii-1,['h=',num2str(h1),', p=',num2str(p1,3)],'Units','normalized','color',colors{ii});
        xticks([1,2]),xticklabels({'base','cooling'})
    end
    
    subplot(3,3,j+1)
    data2process = batchData.peakFrequency;
    data2process2 = batchData.peakFrequency.set2;
    data2process1 = batchData.peakFrequency.set1;
%     data2process2 = data2process2-data2process2;
    
    for k = 1:numel(data2process.set1)
        plot([2;1],[data2process1(k);data2process2(k)],['-',colors{ii}],'linewidth',1),hold on
        P = data2process.P(k);
        if P<0.003
            plot([2;1],[data2process1(k);data2process2(k)],colors{ii},'linewidth',2),hold on
        elseif P<0.05
            plot([2;1],[data2process1(k);data2process2(k)],colors{ii},'linewidth',1),hold on
        end
    end
    plot(2,data2process1,'or'), plot(1,data2process2,'ob')
    xlim([0.5,2.5]),ylabel('Frequency difference')
    [p1,h1] = kstest2(data2process1',data2process2');
    [p2,h2] = kstest2(data2process1',data2process2');
    text(0.02,ii-1,['h=',num2str(h1),', p=',num2str(p1,3)],'Units','normalized','color',colors{ii});
    xticks([1,2]),xticklabels({'base','cooling'})
end
saveas(fig4,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripplesCooling_ipsi_contra_analysis.png'));

%% Heating
extension =  {'ripples_heating_ipsi','ripples_heating_contra'};
extension1 =  {'ripples_ipsi','ripples_contra'};
for i = 1:numel(sessions_heating)
    basename = sessions_heating{i};
    disp(['Processing ipsi-/contralateral: ' basename,' '])
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    [b1, a1] = butter(3, 2/temperature.sr*2, 'low');
    temperature.filter = filtfilt(b1, a1, temperature.data);
    
    fig1 = figure('name',basename);
    subplot(2,1,1)
    plot(temperature.timestamps,temperature.data,'k'), title(basename), hold on, axis tight
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Heating intervals/states
    
    if isfield(temperature.states,'Heating')
        Heating = temperature.states.Heating;
        plot(Heating',[35,35],'o-b','linewidth',1.5)
    end
    
    if isfield(temperature.states,'HeatingControl')
        HeatingControl = temperature.states.HeatingControl;
        plot(HeatingControl',[36,36],'o-g','linewidth',1.5)
    end
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    for j = 1:2
        disp(['Processing: ' extension1{j}])
        if j == 1
            % Ipsilateral
            session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
        else
            % Contralateral
            session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1_contra{strcmp(sessions_metadata.Session_name,basename)});
        end
        if ~(isfield(session,'channelTags') && isfield(session.channelTags,'Ripple'))
            if ~isempty(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)})
                session.channelTags.Ripple.channels=str2num(sessions_metadata.Ripple_ch_idx1{strcmp(sessions_metadata.Session_name,basename)});
            else
                warning('No ripple channel assigned')
            end
        end
        if exist(fullfile(basepath,[basename,'.',extension1{j},'.events.mat']))
            disp('Loading existing ripples file')
            load(fullfile(basepath,[basename,'.',extension1{j},'.events.mat']))
            if j == 1
                ripples = ripples_ipsi;
            else
                ripples = ripples_contra;
            end
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
            disp(['Ripple channel: ' session.channelTags.Ripple.channels])
            if j == 1
                ripples_ipsi = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
                saveStruct(ripples_ipsi,'events','session',session);
                NeuroScope2('session',session,'events','ripples_ipsi')
            else
                ripples_contra = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, 'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');
                saveStruct(ripples_contra,'events','session',session);
                NeuroScope2('session',session,'events','ripples_contra')
            end
        % NeuroScope2('session',session,'events',extension)
            if j == 1
                ripples = ripples_ipsi;              
            else
                ripples = ripples_contra;
            end
        end
        
        if isfield(ripples,'flagged')
            ripples.peaks(ripples.flagged) = [];
            ripples.timestamps(ripples.flagged,:) = [];
            ripples.peakNormedPower(ripples.flagged) = [];
        end
        
        if isfield(ripples,'time')
            ripples = rmfield(ripples,'time');
        end
        idx = find(InIntervals(ripples.peaks,HeatingControl));
        ripples.peaks = ripples.peaks(idx);
        ripples.timestamps = ripples.timestamps(idx,:);
        ripples.peakNormedPower = ripples.peakNormedPower(idx);
        
        lfp = bz_GetLFP(session.channelTags.Ripple.channels-1,'basepath',basepath);
        ripfiltlfp = bz_Filter(lfp.data,'passband',[100 200],'filter','fir1');
        [~,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
        % PlotRippleStats(ripples,maps,data,stats)
        ripples.peakFrequency = data.peakFrequency;
        ripples.duration = data.duration*1000;
        ripples.peakAmplitude = data.peakAmplitude/1000;
        ripples.ISIs = diff(ripples.peaks); ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
        ripples.temperature = interp1(temperature.timestamps,temperature.data, ripples.peaks);
        
        t_minutes = [0:ceil(max(ripples.peaks)/30)];
        ripple_rate = histcounts(ripples.peaks/30,t_minutes)/30;
        ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks/30);
        ripples.heating = InIntervals(ripples.peaks,Heating); 
        
        ylabel('Temperature (°C)')
        
        fig2 = figure('name',[basename,' ' extension{j}]);
        compareScatter(ripples,ripples.heating)
        
        fig3 = figure('name',[basename,' ' extension{j}]);
        compareState(ripples,ripples.heating);
        
        if j == 1
            ripples_heating_ipsi = ripples;
            saveStruct(ripples_heating_ipsi,'events','session',session);
            saveas(fig2,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesHeatingStats1.ripples_ipsi.png']))
            saveas(fig3,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesHeatingStats2.ripples_ipsi.png']))
        else
            ripples_heating_contra = ripples;
            saveStruct(ripples_heating_contra,'events','session',session);
            saveas(fig2,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesHeatingStats1.ripples_contra.png']))
            saveas(fig3,fullfile(local_path,'RipplesVsTemperature/Figures',[basename,'.ripplesHeatingStats2.ripples_contra.png']))
        end
    end
end

%% Batch heating
batchData_ipsi_contra = {};
extension =  {'ripples_heating_ipsi','ripples_heating_contra'};
colors = {'r','k'};
k = [];
k(1) = 1;
k(2) = 1;
for i = 1:numel(sessions_heating)
    basename = sessions_heating{i};
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    basepath = fullfile(basepath_root,animal_subject,basename);
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    for ii = 1:2
        data = loadStruct(extension{ii},'events','session',session);
        fieldsToAnalize = fieldnames(data);
        states = data.heating;
        fieldsToAnalize = setdiff(fieldsToAnalize,{'heating','cooling','timestamps','peaks','detectorinfo','noise'});
        for j = 1:numel(fieldsToAnalize)
            if all(size(data.(fieldsToAnalize{j})) == [size(data.timestamps,1),1]) && isnumeric(data.(fieldsToAnalize{j}))
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).set1(k(ii)) = nanmean(data.(fieldsToAnalize{j})(states));
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).set2(k(ii)) = nanmean(data.(fieldsToAnalize{j})(~states));
                [h,P] = kstest2(data.(fieldsToAnalize{j})(states),data.(fieldsToAnalize{j})(~states));
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).h(k(ii)) = h;
                batchData_ipsi_contra{ii}.(fieldsToAnalize{j}).P(k(ii)) = P;
            end
        end
        k(ii) = k(ii)+1;
    end
end

fig4 = figure;
for ii = 1:2
    batchData = batchData_ipsi_contra{ii};
    fieldsToAnalize = fieldnames(batchData);
    
    for j = 1:numel(fieldsToAnalize)
        subplot(3,3,j)
        data2process = batchData.(fieldsToAnalize{j});
        for k = 1:numel(data2process.set1)
            plot([2;1],[data2process.set1(k);data2process.set2(k)],['-',colors{ii}],'linewidth',1),hold on
            P = data2process.P(k);
            if P<0.003
                plot([2;1],[data2process.set1(k);data2process.set2(k)],colors{ii},'linewidth',2),hold on
            elseif P<0.05
                plot([2;1],[data2process.set1(k);data2process.set2(k)],colors{ii},'linewidth',1),hold on
            end
        end
        plot(2,data2process.set1,'or'), plot(1,data2process.set2,'ob')
        xlim([0.5,2.5]),ylabel(fieldsToAnalize{j})
        [p1,h1] = signrank(data2process.set1',data2process.set2');
        text(0.02,0.05+(ii-1)*0.9 ,['h=',num2str(h1),', p=',num2str(p1,3)],'Units','normalized','color',colors{ii});
        xticks([1,2]),xticklabels({'base','heating'})
        plot([2.1,0.9],[mean(data2process.set1),mean(data2process.set2)],'s-','linewidth',2,'color',colors{ii})
    end
    
    subplot(3,3,j+1)
    data2process = batchData.peakFrequency;
    data2process2 = batchData.peakFrequency.set2;
    data2process1 = batchData.peakFrequency.set1;% - data2process2;
%     data2process2 = data2process2-data2process2;
    
    for k = 1:numel(data2process.set1)
        plot([2;1],[data2process1(k);data2process2(k)],['-',colors{ii}],'linewidth',1),hold on
        P = data2process.P(k);
        if P<0.003
            plot([2;1],[data2process1(k);data2process2(k)],colors{ii},'linewidth',2),hold on
        elseif P<0.05
            plot([2;1],[data2process1(k);data2process2(k)],colors{ii},'linewidth',1),hold on
        end
    end
    
    plot(2,data2process1,'or'), plot(1,data2process2,'ob')
    xlim([0.5,2.5]),ylabel('Frequency difference')
    [p1,h1] = signrank(data2process1',data2process2');
    text(0.02,0.05+(ii-1)*0.9,['h=',num2str(h1),', p=',num2str(p1,3)],'Units','normalized','color',colors{ii});
    xticks([1,2]),xticklabels({'base','heating'})
    plot([2.1,0.9],[mean(data2process1),mean(data2process2)],'s-','linewidth',2,'color',colors{ii})
end
saveas(fig4,fullfile(local_path,'RipplesVsTemperature/Figures','batch_ripples_heating_ipsi_contra_analysis.png'));
