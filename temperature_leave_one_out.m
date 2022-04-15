% Leave on out analysis - prediction of ripple frequency
%
% Predictors:
% Temperature
% Ripple rate
% State
% hi-pascoh
% Theta ratio
% Movement
% theta/delta ratio
% Power spectrum slope
clear all

local_path_pc = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';
local_path_mac = '/Users/peterpetersen/Dropbox/Buzsakilab Postdoc/Matlab/';
local_path = local_path_pc;
user_name = 'Peter';
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

sessions_control_peter = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170716_172307_concat', 'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS21_180628_155921_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180629_110319_concat', ...
    'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};

% Problems: 3
% not good state scoring: 9,10,13

sessions_cooling = {'Temp_R05_20201219','Temp_R05_20201228','Temp_R05_20201229','Temp_R05_20210101','Temp_R05_20210102','Temp_R05_20210129','Temp_R05_20210130_overnight','Temp_R07_20210215','Temp_R07_20210219','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210307'};
sessions_heating = {'Temp_R05_20210130','Temp_R05_20210131','Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217','Temp_R07_20210219','Temp_R08_20210224','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210306','Temp_R09_20210404','Temp_R09_20210407'};
sessions_misi = [sessions_cooling,sessions_heating];
sessions = sessions_control_peter;

for k = 1:numel(sessions)
    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    
    for i = 1:floor(numel(temperature.data)/(temperature.sr))
        temperature.data_slow(i) = nanmean(temperature.data(temperature.sr*(i-1)+1:i*temperature.sr));
        temperature.timestamps_slow(i) = i;
    end
    
    % States
    SleepState = loadStruct('SleepState','states','session',session);
    
    % Ripples
    ripples = loadStruct('ripples','events','session',session);
    ripples.peaks(ripples.flagged)=[];
    ripples.timestamps(ripples.flagged,:)=[];
    if isfield(ripples,'time')
        ripples.time(ripples.flagged)=[];
    end
    ripples.peakNormedPower(ripples.flagged)=[];
        
    lfp = bz_GetLFP(ripples.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
    ripfiltlfp = bz_Filter(lfp.data,'passband',[100 220],'filter','fir1');
    [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples);
    ripples.peakFrequency = data.peakFrequency;
    ripples.duration = data.duration*1000;
    ripples.peakAmplitude = data.peakAmplitude/1000;
    ripples.ISIs = diff(ripples.peaks); 
    ripples.ISIs = log10([ripples.ISIs;ripples.ISIs(end)]);
%     ripples.ISIs2 = 1./diff(ripples.peaks);
%     ripples.rate =  1./min([ripples.ISIs2(1:end-1),ripples.ISIs2(2:end)]');
%     ripples.rate = ripples.rate([1,1:end,end])';

    t_minutes = [0:5:ceil(max(ripples.peaks))+5];
    ripple_rate = histcounts(ripples.peaks,t_minutes)/5;
    ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks);
    ripples.temperature = interp1(temperature.timestamps_slow,temperature.data_slow, ripples.peaks);
    ripples.states = interp1(SleepState.idx.timestamps,SleepState.idx.states, ripples.peaks);
    
    % theta delta ratio
    [spectrogram,t,f] = MTSpectrogram([lfp.timestamps,double(lfp.data)]);
%     figure, imagesc(t,f,log10(spectrogram));
    bands = SpectrogramBands(spectrogram,f,'theta',[5,12]);
    ripples.theta_delta_ratio = interp1(t,bands.ratio, ripples.peaks);
    
    % power_spectrum_slope
    f_span = 66:3277;
    % fooof_results = fooof(f(f_span), spectrogram(f_span,1), [5 300],[],true)
    x = f(f_span);
    power_spectrum_slope1 = [];
    power_spectrum_slope2 = [];
    for i = 1:size(spectrogram,2)
        y = spectrogram(f_span,i);
        fit_fcn = @(b,x) x.^b(1) .* exp(b(2));                                  % Objective Function
        RNCF = @(b) norm(y - fit_fcn(b,x));                                     % Residual Norm Cost Function
        B = fminsearch(RNCF, [1; 1]);                                           % Estimate Parameters
        p = polyfit(log(x),log(y),1);
        
        power_spectrum_slope1(i) = B(1);
        power_spectrum_slope2(i) = p(1);
    end
    ripples.power_spectrum_slope1 = interp1(t,power_spectrum_slope1, ripples.peaks);
    ripples.power_spectrum_slope2 = interp1(t,power_spectrum_slope2, ripples.peaks);
    
    ripples_limited = ripples;
    saveStruct(ripples_limited,'events','session',session);
end

%%

sessions = [sessions_control_peter,sessions_misi];
% sessions = sessions_misi;
sessions = sessions_control_peter;
mld_results1 = [];
mld_results2 = [];
mld_results3 = [];
kk = 1;

for k = 1:numel(sessions)
    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    
    ripples_limited = loadStruct('ripples_limited','events','session',session);
    ripples = ripples_limited;
    if isfield(temperature.states,'Bad')
        idx = find(~InIntervals(ripples.peaks,temperature.states.cooling) & ~InIntervals(ripples.peaks,temperature.states.Bad));
    else
        idx = find(~InIntervals(ripples.peaks,temperature.states.cooling));
    end
    
    % Ripple frequency - leave one out
    X0 = [ripples.temperature,ripples.states,ripples.rate,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];
    X1 = [ripples.states,ripples.rate,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];          % out: temperature
    X2 = [ripples.temperature,ripples.states,ripples.rate,ripples.theta_delta_ratio];                % out: power_spectrum_slope
    X3 = [ripples.temperature,ripples.rate,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];     % out: states
    X4 = [ripples.temperature,ripples.states,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];   % out: rate
    X5 = [ripples.temperature,ripples.states,ripples.rate,ripples.power_spectrum_slope2];                % out: theta_delta_ratio
    
    mdl0 = fitlm(X0(idx,:),ripples.peakFrequency(idx));
    mdl1 = fitlm(X1(idx,:),ripples.peakFrequency(idx));
    mdl2 = fitlm(X2(idx,:),ripples.peakFrequency(idx));
    mdl3 = fitlm(X3(idx,:),ripples.peakFrequency(idx));
    mdl4 = fitlm(X4(idx,:),ripples.peakFrequency(idx));
    mdl5 = fitlm(X5(idx,:),ripples.peakFrequency(idx));
    
    mld_results1(:,kk) = [mdl1.RMSE,mdl2.RMSE,mdl3.RMSE,mdl4.RMSE,mdl5.RMSE]-mdl0.RMSE;
    
    % Ripple rate
    X0 = [ripples.temperature,ripples.states,ripples.peakFrequency,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];
    X1 = [ripples.states,ripples.peakFrequency,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];          % out: temperature
    X2 = [ripples.temperature,ripples.states,ripples.peakFrequency,ripples.theta_delta_ratio];                    % out: power_spectrum_slope2
    X3 = [ripples.temperature,ripples.peakFrequency,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];     % out: states
    X4 = [ripples.temperature,ripples.states,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];            % out: peakFrequency
    X5 = [ripples.temperature,ripples.states,ripples.peakFrequency,ripples.power_spectrum_slope2];                % out: theta_delta_ratio
    
    mdl0 = fitlm(X0(idx,:),ripples.rate(idx));
    mdl1 = fitlm(X1(idx,:),ripples.rate(idx));
    mdl2 = fitlm(X2(idx,:),ripples.rate(idx));
    mdl3 = fitlm(X3(idx,:),ripples.rate(idx));
    mdl4 = fitlm(X4(idx,:),ripples.rate(idx));
    mdl5 = fitlm(X5(idx,:),ripples.rate(idx));
    
    mld_results2(:,kk) = [mdl1.RMSE,mdl2.RMSE,mdl3.RMSE,mdl4.RMSE,mdl5.RMSE]-mdl0.RMSE;
    
    % Error by single metric
    mdl1 = fitlm(ripples.temperature(idx),ripples.peakFrequency(idx));
    mdl2 = fitlm(ripples.power_spectrum_slope2(idx),ripples.peakFrequency(idx));
    mdl3 = fitlm(ripples.states(idx),ripples.peakFrequency(idx));
    mdl4 = fitlm(ripples.rate(idx),ripples.peakFrequency(idx));
    mdl5 = fitlm(ripples.theta_delta_ratio(idx),ripples.peakFrequency(idx));
    
    mld_results3(:,kk) = [mdl1.RMSE,mdl2.RMSE,mdl3.RMSE,mdl4.RMSE,mdl5.RMSE];
    mld_results3(:,kk) = mld_results3(:,kk)-mean(mld_results3(:,kk));
    kk = kk+1;
    
    figure
    subplot(2,3,1)
    plot(ripples.temperature(idx),ripples.peakFrequency(idx),'.'), xlabel('Temperature'), ylabel('Frequency'), title(basename)
    subplot(2,3,2)
    plot(ripples.states(idx),ripples.peakFrequency(idx),'.'), xlabel('states')
    subplot(2,3,3)
    plot(ripples.theta_delta_ratio(idx),ripples.peakFrequency(idx),'.'), xlabel('theta delta ratio')
    subplot(2,3,4)
    plot(ripples.power_spectrum_slope2(idx),ripples.peakFrequency(idx),'.'), xlabel('power-spectrum-slope')
    subplot(2,3,5)
    plot(ripples.temperature(idx),ripples.power_spectrum_slope1(idx),'.'), ylabel('power-spectrum-slope1'), xlabel('Temperature')
    subplot(2,3,6)
    plot(ripples.temperature(idx),ripples.power_spectrum_slope2(idx),'.'), ylabel('power-spectrum-slope2'), xlabel('Temperature')
end

labels1 = {'temperature','power-spectrum-slope','states','rate','theta-delta-ratio'};
labels2 = {'temperature','power-spectrum-slope','states','frequency','theta-delta-ratio'};

figure
subplot(2,1,1)
plot(mld_results1), hold on, title(basename)
xticks([1:5]), xticklabels(labels1), title('Ripple frequency - leave one out'), xlabel('Leave one out'), ylabel('RMSE'), xlim([0.5,5.5])

x = [1:5];
y = mean(mld_results1');
err = std(mld_results1');
subplot(2,1,2)
errorbar(x,y,err), ylabel('RMSE'), xlim([0.5,5.5])
xticks([1:5]), xticklabels(labels2),

% Ripple rate
figure
subplot(2,1,1)
plot(mld_results2), hold on, title(basename)
xticks([1:5]), xticklabels(labels2), title('Ripple rate'), xlabel('Leave one out'), ylabel('RMSE'), xlim([0.5,5.5])

x = [1:5];
y = mean(mld_results2');
err = std(mld_results2');
subplot(2,1,2)
errorbar(x,y,err), ylabel('RMSE'), xlim([0.5,5.5])
xticks([1:5]), xticklabels(labels2),

% Individual fits
figure
subplot(2,1,1)
plot(mld_results3), hold on, title(basename)
xticks([1:5]), xticklabels(labels1), title('Ripple frequency, single predictor'), xlabel('Individual fits'), ylabel('RMSE'), xlim([0.5,5.5])

x = [1:5];
y = mean(mld_results3');
err = std(mld_results3');
subplot(2,1,2)
errorbar(x,y,err), ylabel('RMSE'), xlim([0.5,5.5])
xticks([1:5]), xticklabels(labels1),

%%
% for k = 1:numel(sessions)
    k = 1;
    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
%     session = loadSession(basepath,basename); % Loading session info

    TheStateEditor(basename);
% end

%% Generating a PSTH for REM/Awake/nonREM/Micro arousal brain states onsets
% TheStateEditor(basename);
% Not good: Peter_MS12_170716_172307_concat (k=3)
% Not good: Peter_MS21_180629_110332_concat (k=10)
% Not good: Peter_MS21_180628_155921_concat (k=13)
% Not good: Peter_MS22_180629_110319_concat (k=15)

sessions = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', 'Peter_MS12_170717_111614_concat', ...
    'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat',...
    'Temp_R05_20210101','Temp_R07_20210215','Temp_R07_20210219','Temp_R05_20210130','Temp_R08_20210307','Temp_R05_20210131',...
    'Temp_R08_20210224','Temp_R08_20210306','Temp_R09_20210404'};

figure, 
temp_rem_onset_all = [];
theta_delta_ratio_rem_onset_all = [];
accelerometer_rem_onset_all = [];

% State to analyze
% stateData = 'REMstate';
% stateData = 'NREMstate';
% stateData = 'WAKEstate';
stateData = 'Intermediatestate';

% Not complete: 13
for k = 1:numel(sessions)
    
    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
%     TheStateEditor(basename);
% end
% %%
% for k = 1:numel(sessions)
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    
    temperature.data_slow = [];
    temperature.timestamps_slow = [];
    for i = 1:floor(numel(temperature.data)/(temperature.sr))
        temperature.data_slow(i) = nanmean(temperature.data(temperature.sr*(i-1)+1:i*temperature.sr));
        temperature.timestamps_slow(i) = i;
    end
    temperature_StateEditor = temperature.data_slow;
    manipulation_intervals = [];
    if isfield(temperature.states,'cooling')
        for i = 1:size(temperature.states.cooling,1)
            temperature_StateEditor(floor(temperature.states.cooling(i,1)):ceil(temperature.states.cooling(i,2))) = nan;
        end
        manipulation_intervals = [manipulation_intervals;temperature.states.cooling];
    end
    if isfield(temperature.states,'Cooling')
        for i = 1:size(temperature.states.Cooling,1)
            temperature_StateEditor(floor(temperature.states.Cooling(i,1)):ceil(temperature.states.Cooling(i,2))) = nan;
        end
        manipulation_intervals = [manipulation_intervals;temperature.states.Cooling];
    end
    if isfield(temperature.states,'Heating')
        for i = 1:size(temperature.states.Heating,1)
            temperature_StateEditor(floor(temperature.states.Heating(i,1)):ceil(temperature.states.Heating(i,2))) = nan;
        end
        manipulation_intervals = [manipulation_intervals;temperature.states.Heating];
    end
    if isfield(temperature.states,'Bad')
        for i = 1:size(temperature.states.Bad,1)
            temperature_StateEditor(floor(temperature.states.Bad(i,1))+1:ceil(temperature.states.Bad(i,2))) = nan;
        end
        manipulation_intervals = [manipulation_intervals;temperature.states.Bad];
    end
    if isfield(temperature.states,'weird')
        for i = 1:size(temperature.states.weird,1)
            temperature_StateEditor(floor(temperature.states.weird(i,1))+1:ceil(temperature.states.weird(i,2))) = nan;
        end
        manipulation_intervals = [manipulation_intervals;temperature.states.weird];
    end
    save('temperature_StateEditor.mat','temperature_StateEditor')
    
    % theta delta ratio
    ripples = loadStruct('ripples','events','session',session);
    lfp = bz_GetLFP(ripples.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
    [spectrogram,t,f] = MTSpectrogram([lfp.timestamps,double(lfp.data)],'step',1);
    bands = SpectrogramBands(spectrogram,f,'theta',[5,12]);
    
    % Motion 
    load([basename,'.eegstates.mat']);
    
    % States
    SleepState = loadStruct('SleepState','states','session',session);
    
    if isfield(SleepState.ints,stateData)
    state_intervals = SleepState.ints.(stateData);
    REMstate2 = [state_intervals(:,1)-120,state_intervals(:,2)+180];
    manipulation_intervals(:,2) = manipulation_intervals(:,2)+300;
    idx1 = InIntervals(REMstate2(:,1),manipulation_intervals);
    idx2 = InIntervals(REMstate2(:,2),manipulation_intervals);
    idx3 = InIntervals(REMstate2(:,1),[-150,120]);
    idx4 = InIntervals(REMstate2(:,2),temperature.timestamps(end)-[120,0]);
    state_intervals(find(idx1 | idx2 | idx3 | idx4 ),:) = [];
    t_interval = -120:180;
    temp_rem_onset = nan(301,size(state_intervals,1));
    theta_delta_ratio_rem_onset = nan(301,size(state_intervals,1));
    motion_rem_onset = nan(301,size(state_intervals,1));
    for i = 1:size(state_intervals,1)
        pretimes = -(state_intervals(:,2)-state_intervals(i,1));
        pretimes(pretimes<0)=inf;
        
        state_begin = min([min(pretimes),120]);
        state_end = min([diff(state_intervals(i,:)),180]);
        
        temp_rem_onset((121-state_begin):state_end+121,i) = temperature.data_slow(state_intervals(i,1)-state_begin:state_intervals(i,1)+state_end);
        % Theta delta ratio
        ratio_start = find(t>(state_intervals(i,1)-state_begin),1);
        ratio_end = find(t>(state_intervals(i,1)+state_end),1);
        if ~isempty(ratio_start) & ~isempty(ratio_end)
            theta_delta_ratio_rem_onset((121-state_begin):state_end+121,i) = bands.ratio(ratio_start:ratio_end);
        end
        % Motion
        if ~isempty(StateInfo.motion)
            ratio_start = find(t>(state_intervals(i,1)-state_begin),1);
            ratio_end = find(t>(state_intervals(i,1)+state_end),1);
            if ~isempty(ratio_start) & ~isempty(ratio_end)
                motion_rem_onset((121-state_begin):state_end+121,i) = StateInfo.motion(ratio_start:ratio_end);
            end
        end
    end
    temp_rem_onset = temp_rem_onset-temp_rem_onset(121,:);
    
    subplot(3,2,1) % Temperature trials
    plot(t_interval,temp_rem_onset), hold on
    temp_rem_onset_all = [temp_rem_onset_all,temp_rem_onset];
    
    subplot(3,2,3) % Theta-delta ratio
    plot(t_interval,theta_delta_ratio_rem_onset), hold on
    theta_delta_ratio_rem_onset_all = [theta_delta_ratio_rem_onset_all,theta_delta_ratio_rem_onset];
    
    subplot(3,2,5) % accelerometer data
    plot(t_interval,motion_rem_onset), hold on
    accelerometer_rem_onset_all = [accelerometer_rem_onset_all,motion_rem_onset];
    
    else
       disp(['State data not available: ' stateData])
    end
end
subplot(3,2,3) % Theta-delta ratio
title(basename,'interpreter','none'), xlabel('Time(s)'),ylabel('theta-delta-ratio'), xlim([-120,180])

subplot(3,2,1)
plot(t_interval,nanmean(temp_rem_onset_all')','k','linewidth',2), title(stateData,'interpreter','none'), xlabel('Time(s)'), ylabel('Temperature (C)'), hold on, axis tight, xlim([-120,180])

y = nanmean(temp_rem_onset_all');
err = nanstd(temp_rem_onset_all');
subplot(3,2,2)
errorbarPatch(t_interval,y,err,'b'), xlabel('Time(s)'), title('','interpreter','none'), grid on, xlim([-120,180])

y = nanmean(theta_delta_ratio_rem_onset_all');
err = nanstd(theta_delta_ratio_rem_onset_all');
subplot(3,2,4)
% plot(t_interval,y,'b')
errorbarPatch(t_interval,y,err,'b'), 
xlabel('Time(s)'), title('','interpreter','none'), grid on, xlim([-120,180])

y = nanmean(accelerometer_rem_onset_all');
err = nanstd(accelerometer_rem_onset_all');
subplot(3,2,6)
% plot(t_interval,y,'r')
errorbarPatch(t_interval,y,err,'r'), 
xlabel('Time(s)'), title('','interpreter','none'), grid on,ylim([0 1]), xlim([-120,180])

%% Brain temperature autocorrelogram
sessions = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180627_143449_concat', 'Peter_MS21_180712_103200_concat', ...
    'Peter_MS22_180628_120341_concat', 'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};

r_all = [];
for k = 1:numel(sessions)

    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    
    temperature.data_slow = [];
    temperature.timestamps_slow = [];
    for i = 1:floor(numel(temperature.data)/(temperature.sr))
        temperature.data_slow(i) = nanmean(temperature.data(temperature.sr*(i-1)+1:i*temperature.sr));
        temperature.timestamps_slow(i) = i;
    end
    temperature_StateEditor = temperature.data_slow;
    
    manipulation_intervals = [];
    if isfield(temperature.states,'cooling')
        manipulation_intervals = [manipulation_intervals;temperature.states.cooling];
    end
    if isfield(temperature.states,'Cooling')
        manipulation_intervals = [manipulation_intervals;temperature.states.Cooling];
    end
    if isfield(temperature.states,'Heating')
        manipulation_intervals = [manipulation_intervals;temperature.states.Heating];
    end
    if isfield(temperature.states,'Bad')
        for i = 1:size(temperature.states.Bad,1)
            temperature_StateEditor(floor(temperature.states.Bad(i,1))+1:ceil(temperature.states.Bad(i,2))) = nan;
        end
        manipulation_intervals = [manipulation_intervals;temperature.states.Bad];
    end
    if isfield(temperature.states,'weird')
        manipulation_intervals = [manipulation_intervals;temperature.states.weird];
    end
    temperature.data_slow5 = temperature.data_slow;
    idx12 = find(diff(temperature.data_slow)>0.25);
    for i = 1:numel(idx12)
        temperature.data_slow5(idx12(i)+1:end) = temperature.data_slow5(idx12(i)+1:end)+temperature.data_slow5(idx12(i))-temperature.data_slow5(idx12(i)+1);
    end
    
    
    temperature.data_slow4 = temperature.data_slow5;
%     idx1 = InIntervals(temperature.timestamps_slow,manipulation_intervals);
    for i = 1:size(manipulation_intervals,1)
        t_start = find(temperature.timestamps_slow>manipulation_intervals(i,1),1);
        t_duration = ceil(diff(manipulation_intervals(i,:)));
        temperature.data_slow4(t_start:t_start+t_duration) = nan;
        if ~isempty(temperature.data_slow4(t_start+t_duration+1:end))
            temperature.data_slow4(t_start+t_duration+1:end) = temperature.data_slow4(t_start+t_duration+1:end)+temperature.data_slow4(t_start-1)-temperature.data_slow4(t_start+t_duration+1);
        end
    end
%     temperature.timestamps_slow2 = temperature.timestamps_slow;
%     temperature.timestamps_slow2(isnan(temperature.data_slow4)) = [];
    temperature.data_slow4(isnan(temperature.data_slow4)) = [];
    idx12 = find(abs(diff(temperature.data_slow4))>0.1);
    for i = 1:numel(idx12)
        temperature.data_slow4(idx12(i)+1:end) = temperature.data_slow4(idx12(i)+1:end)+temperature.data_slow4(idx12(i))-temperature.data_slow4(idx12(i)+1);
    end

    temperature.data_slow3 = temperature.data_slow4-nanmean(temperature.data_slow4);
    [r,lags] = xcorr(temperature.data_slow3,temperature.data_slow3,3600,'normalized');
    figure, 
    subplot(3,1,1)
    plot(temperature.timestamps_slow,temperature.data_slow5)
    subplot(3,1,2)
    plot(temperature.data_slow3), title(basename)
    subplot(3,1,3)
    plot(lags,r), hold on
    r_all(k,:) = r;
    drawnow
end

figure
subplot(1,2,1)
plot(lags,r_all), hold on
plot(lags,nanmean(r_all),'k','linewidth',2),title('Temperature autocorrelogram','interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), hold on, axis tight

y = nanmean(r_all);
err = nanstd(r_all);
subplot(1,2,2)
errorbarPatch(lags,y,err,'b'), xlabel('Time(s)'),title('','interpreter','none'), axis tight

xticks([-1800,0,1800])
xticklabels([-0.5,0,0.5])

%% Autocorr within states
sessions = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180627_143449_concat', 'Peter_MS21_180712_103200_concat', ...
    'Peter_MS22_180628_120341_concat', 'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};

autocorrr_all = [];
window_duration = 100;
for k = 1:numel(sessions)

    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    
    temperature.data_slow = [];
    temperature.timestamps_slow = [];
    for i = 1:floor(numel(temperature.data)/(temperature.sr))
        temperature.data_slow(i) = nanmean(temperature.data(temperature.sr*(i-1)+1:i*temperature.sr));
        temperature.timestamps_slow(i) = i;
    end
    temperature_StateEditor = temperature.data_slow;
    
    manipulation_intervals = [];
    if isfield(temperature.states,'cooling')
        manipulation_intervals = [manipulation_intervals;temperature.states.cooling];
    end
    if isfield(temperature.states,'Cooling')
        manipulation_intervals = [manipulation_intervals;temperature.states.Cooling];
    end
    if isfield(temperature.states,'Heating')
        manipulation_intervals = [manipulation_intervals;temperature.states.Heating];
    end
    if isfield(temperature.states,'Bad')
        for i = 1:size(temperature.states.Bad,1)
            temperature_StateEditor(floor(temperature.states.Bad(i,1))+1:ceil(temperature.states.Bad(i,2))) = nan;
        end
        manipulation_intervals = [manipulation_intervals;temperature.states.Bad];
    end
    if isfield(temperature.states,'weird')
        manipulation_intervals = [manipulation_intervals;temperature.states.weird];
    end
    temperature.data_slow5 = temperature.data_slow;
    idx12 = find(diff(temperature.data_slow)>0.25);
    for i = 1:numel(idx12)
        temperature.data_slow5(idx12(i)+1:end) = temperature.data_slow5(idx12(i)+1:end)+temperature.data_slow5(idx12(i))-temperature.data_slow5(idx12(i)+1);
    end
        
    temperature.data_slow4 = temperature.data_slow5;
%     idx1 = InIntervals(temperature.timestamps_slow,manipulation_intervals);
    for i = 1:size(manipulation_intervals,1)
        t_start = find(temperature.timestamps_slow>manipulation_intervals(i,1),1);
        t_duration = ceil(diff(manipulation_intervals(i,:)));
        temperature.data_slow4(t_start:t_start+t_duration) = nan;
        if ~isempty(temperature.data_slow4(t_start+t_duration+1:end))
            temperature.data_slow4(t_start+t_duration+1:end) = temperature.data_slow4(t_start+t_duration+1:end)+temperature.data_slow4(t_start-1)-temperature.data_slow4(t_start+t_duration+1);
        end
    end
%     temperature.timestamps_slow2 = temperature.timestamps_slow;
%     temperature.timestamps_slow2(isnan(temperature.data_slow4)) = [];
%     temperature.data_slow4(isnan(temperature.data_slow4)) = [];
    idx12 = find(abs(diff(temperature.data_slow4))>0.1);
    for i = 1:numel(idx12)
        temperature.data_slow4(idx12(i)+1:end) = temperature.data_slow4(idx12(i)+1:end)+temperature.data_slow4(idx12(i))-temperature.data_slow4(idx12(i)+1);
    end

    
    % States
    SleepState = loadStruct('SleepState','states','session',session);
    states_names = {'WAKEstate','NREMstate','Intermediatestate','REMstate'};
    autocorrr12 = {};
    for j = 1:numel(states_names)
        intervals_in_states = SleepState.ints.(states_names{j});
        intervals_in_states = intervals_in_states(diff(intervals_in_states')>40,:);
        autocorrr12.(states_names{j}).r = nan(size(intervals_in_states,1),2*window_duration+1);
        for jj = 1:size(intervals_in_states,1)
            intervals_in_states_idx = intervals_in_states(jj,1):intervals_in_states(jj,2);
            temp_data = temperature.data_slow(intervals_in_states_idx);
            [r,lags] = xcorr(temp_data-nanmean(temp_data),temp_data-nanmean(temp_data),min([floor(numel(temp_data)/2),window_duration]),'normalized');
            autocorrr12.(states_names{j}).r(jj,lags+window_duration+1) = r;
            autocorrr12.(states_names{j}).t = lags;
        end
        autocorrr12.(states_names{j}).r(autocorrr12.(states_names{j}).r==0)=nan;
        autocorrr12.(states_names{j}).r_mean = nanmean(autocorrr12.(states_names{j}).r);
%         figure,
%         subplot(2,1,1)
%         plot(autocorrr12.(states_names{j}).r'), title(states_names{j})
%         subplot(2,1,2)
%         plot(autocorrr12.(states_names{j}).r_mean)
        
        autocorrr_all.(states_names{j}).r(k,:) = autocorrr12.(states_names{j}).r_mean;
    end
    figure
    for j = 1:numel(states_names)
        plot(autocorrr12.(states_names{j}).r_mean), hold on
    end
    legend(states_names)        
end

figure;
clrs = {'b','g','y','r'};
for j = 1:numel(states_names)    
    lags = -window_duration:window_duration;
    r_all = autocorrr_all.(states_names{j}).r;
    y = nanmean(r_all);
    err = nanstd(r_all);
%     plot(r_all), hold on
    
    subplot(1,2,1)
    plot(lags,y,clrs{j}), hold on
%     plot(lags,nanmean(r_all),'k','linewidth',2),title('Temperature autocorrelogram','interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), hold on, axis tight
    legend(states_names);
    
    subplot(1,2,2)
    errorbarPatch(lags,y,err,clrs{j}), xlabel('Time(s)'),title('','interpreter','none'); axis tight
end


%% Temperature PSTH trial onset
sessions = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180627_143449_concat', 'Peter_MS21_180712_103200_concat', ...
    'Peter_MS22_180628_120341_concat', 'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};

temp_trial_onset_all = [];
behavior_trial_onset_all = [];
for k = 1:numel(sessions)

    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    
    % Behavior
    animal = loadStruct('animal','behavior','session',session);
    trials = loadStruct('trials','behavior','session',session);
    
    % trial PSTH
    n_trials = trials.total;
    t_interval = (-temperature.sr*2:temperature.sr*4)/temperature.sr;
    t_interval2 = (-animal.sr*2:animal.sr*4)/animal.sr;
    temp_trial_onset = nan(temperature.sr*6+1,n_trials);
    behavior_trial_onset = nan(animal.sr*6+1,n_trials);
    for i = 1:numel(trials.cooling) % n_trials
        if trials.cooling(i)~=2
        trial_start = animal.time(trials.start(i));
        trial_end = animal.time(trials.end(i));
        temperature_start = find(temperature.timestamps>trial_start,1);
%         temperature_end = temperature_start+temperature.sr*2;
        temperature_end = find(temperature.timestamps>trial_end,1);
%         duration = temperature_end-temperature_start;
%         temperature_end = min([temperature_end,temperature_start+temperature.sr*4]);
        
%         temp_trial_onset(1:(temperature_end-temperature_start+temperature.sr*2+1),i) = temperature.data(temperature_start-temperature.sr*2:temperature_end);
        temp_trial_onset(:,i) = temperature.data(temperature_start-temperature.sr*2:temperature_start+temperature.sr*4);
        behavior_trial_onset(:,i) = animal.pos(2,trials.start(i)-animal.sr*2:trials.start(i)+animal.sr*4);
%         pretimes = -(REMstate(:,2)-REMstate(i,1));
%         pretimes(pretimes<0)=inf;
%         
%         state_begin = min([min(pretimes),120]);
%         state_end = min([diff(REMstate(i,:)),180]);
%         
%         temp_trial_onset((121-state_begin):state_end+121,i) = temperature.data_slow(REMstate(i,1)-state_begin:REMstate(i,1)+state_end);
        end
    end
    
    temp_trial_onset = temp_trial_onset-temp_trial_onset(temperature.sr*2+1,:);
    figure, 
    subplot(2,1,1)
    plot(t_interval,temp_trial_onset), hold on
    plot(t_interval,nanmean(temp_trial_onset')','k','linewidth',2),title(['Behavior onset: ',basename],'interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), hold on, axis tight,xlim([-2,4])
    subplot(2,1,2)
    plot(t_interval2,behavior_trial_onset), hold on
    plot(t_interval2,nanmean(behavior_trial_onset')','k','linewidth',2), xlabel('Time(s)'),ylabel('Position (cm)'), hold on, axis tight,xlim([-2,4])
    
    if temperature.sr==1250
        temp_trial_onset_all = [temp_trial_onset_all,temp_trial_onset];
        behavior_trial_onset_all = [behavior_trial_onset_all,behavior_trial_onset];
    end
    drawnow
end
sr = 1250;
t_interval = (-sr*2:sr*4)/sr;
figure
subplot(3,1,1)
plot(t_interval,temp_trial_onset_all), hold on
plot(t_interval,nanmean(temp_trial_onset_all')','k','linewidth',2),title('Temperature autocorrelogram','interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), hold on, axis tight

y = nanmean(temp_trial_onset_all');
err = nanstd(temp_trial_onset_all');
subplot(3,1,2)
errorbarPatch(t_interval,y,err,'b'), xlabel('Time(s)'),title('','interpreter','none'), axis tight

subplot(3,1,3)
plot(t_interval2,behavior_trial_onset_all), hold on
plot(t_interval2,nanmean(behavior_trial_onset_all')','k','linewidth',2), xlabel('Time(s)'),ylabel('Position (cm)'), hold on, axis tight,xlim([-2,4])

%% Trial average of temperature with local heating and cooling

sessions_cooling = {'Temp_R05_20201219','Temp_R05_20201228','Temp_R05_20201229','Temp_R05_20210101','Temp_R05_20210102','Temp_R05_20210129','Temp_R05_20210130_overnight','Temp_R07_20210215','Temp_R07_20210219','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210307'};
sessions_heating = {'Temp_R05_20210130','Temp_R05_20210131','Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217','Temp_R07_20210219','Temp_R08_20210224','Temp_R08_20210304','Temp_R08_20210305','Temp_R08_20210306','Temp_R09_20210404','Temp_R09_20210407'};
sessions = sessions_cooling;
newField = 'Cooling3';
% newField = 'Heating3';
temp_trial_onset_all = [];
for k = 1:numel(sessions)

    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    
%     if ~isfield(temperature.states,newField)
        temperature = StateExplorer(temperature);
%         saveStruct(temperature,'timeseries','session',session);
%     end
    temp_intervals = temperature.states.(newField);
    temp_intervals(diff(temp_intervals')<30,:) = [];
    t_window = 300;
    temp_trial_onset = nan(temperature.sr*t_window*2+1,size(temp_intervals,1));
    t_interval = (-temperature.sr*t_window:temperature.sr*t_window)/temperature.sr;
    for i = 1:size(temp_intervals,1)
        pretimes = -(temp_intervals(:,2)-temp_intervals(i,1));
        pretimes(pretimes<0)=inf;
        
        state_begin = min([min(pretimes),t_window]);
        if temp_intervals(i,1)<t_window
            state_begin = state_begin-temp_intervals(i,1);
        end
        state_end = min([diff(temp_intervals(i,:)),t_window]);
        temperature_start = find(temperature.timestamps>temp_intervals(i,1)-state_begin,1);
        temperature_end = find(temperature.timestamps>temp_intervals(i,1)+state_end,1)-1;
        idx_onset = floor(1+temperature.sr*(t_window-state_begin));
        temp_trial_onset(idx_onset:idx_onset+temperature_end-temperature_start,i) = temperature.data(temperature_start:temperature_end);
    end
    figure
    plot(t_interval,temp_trial_onset), title(basename), xlabel('Time(s)'),ylabel('Temperature (C)'), axis tight, hold on
    plot(t_interval,nanmean(temp_trial_onset')','k','linewidth',2),title(['Behavior onset: ',basename],'interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), axis tight
    if temperature.sr==1250
        temp_trial_onset_all = [temp_trial_onset_all,temp_trial_onset];
    end
end

sr = 1250;
temp_trial_onset_all2 = [];
for j = 1:size(temp_trial_onset_all,2)
    for i = 1:floor(size(temp_trial_onset_all,1)/sr)
        temp_trial_onset_all2(i,j) = nanmean(temp_trial_onset_all(sr*(i-1)+1:i*sr,j));
    end
end

t_interval = (-sr*t_window:sr*t_window)/sr;
t_interval2 = (-t_window:t_window-1);
figure
subplot(3,1,1)
plot(t_interval2,temp_trial_onset_all2), hold on
plot(t_interval2,nanmean(temp_trial_onset_all2')','k','linewidth',2),title('Temperature autocorrelogram','interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), hold on, axis tight
subplot(3,1,2)
y = nanmean(temp_trial_onset_all2');
err = nanstd(temp_trial_onset_all2');
errorbarPatch(t_interval2,y,err,'b'), xlabel('Time(s)'),title('','interpreter','none'), axis tight
plot(t_interval2,nanmean(temp_trial_onset_all2')','k','linewidth',2),title('Temperature autocorrelogram','interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), hold on, axis tight
subplot(3,1,3)
temp_trial_onset_all3 = temp_trial_onset_all2-nanmean(temp_trial_onset_all2(200:295,:));
y = nanmean(temp_trial_onset_all3');
err = nanstd(temp_trial_onset_all3');
errorbarPatch(t_interval2,y,err,'b'), xlabel('Time(s)'),title('','interpreter','none'), axis tight
plot(t_interval2,nanmean(temp_trial_onset_all3')','k','linewidth',2),title('Temperature autocorrelogram','interpreter','none'), xlabel('Time(s)'),ylabel('Temperature (C)'), hold on, axis tight


%% Leave one out and single predictors with temporal offsets

% sessions = [sessions_control_peter,sessions_misi];
% sessions = sessions_misi;
% sessions = sessions_control_peter;

sessions = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', 'Peter_MS12_170717_111614_concat', ...
    'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180627_143449_concat', ...
    'Peter_MS21_180712_103200_concat', 'Peter_MS22_180628_120341_concat', 'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};

mld_results1 = [];
mld_results3 = [];
kk = 1;
step_size = 20; % in seconds
n_steps = 120;
n_steps_all = n_steps*2+1;
temporal_offsets = -step_size*n_steps:step_size:step_size*n_steps;
colors = autumn(length(temporal_offsets));
colors2 = autumn(5);
labels1 = {'temperature','power-spectrum-slope','states','rate','theta-delta-ratio'};
labels2 = {'temperature','power-spectrum-slope','states','frequency','theta-delta-ratio'};

for k = 1:numel(sessions)
    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature.states,'cooling') && isfield(temperature.states,'Cooling')
        temperature.states.cooling = temperature.states.Cooling;
    end
    
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    for i = 1:floor(numel(temperature.data)/(temperature.sr))
        temperature.data_slow(i) = nanmean(temperature.data(temperature.sr*(i-1)+1:i*temperature.sr));
        temperature.timestamps_slow(i) = i;
    end
    
    ripples_limited = loadStruct('ripples_limited','events','session',session);
    ripples = ripples_limited;
    
    % States
    SleepState = loadStruct('SleepState','states','session',session);
    
    % Ripple rate
    t_minutes = [0:5:ceil(max(ripples.peaks))+5];
    ripple_rate = histcounts(ripples.peaks,t_minutes)/5;
    
    % theta delta ratio
    lfp = bz_GetLFP(ripples.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
    [spectrogram,t,f] = MTSpectrogram([lfp.timestamps,double(lfp.data)]);
    bands = SpectrogramBands(spectrogram,f,'theta',[5,12]);
    
    % power_spectrum_slope
    f_span = 66:3277;
    % fooof_results = fooof(f(f_span), spectrogram(f_span,1), [5 300],[],true)
    x = f(f_span);
    power_spectrum_slope1 = [];
    power_spectrum_slope2 = [];
    for i = 1:size(spectrogram,2)
        y = spectrogram(f_span,i);
        p = polyfit(log(x),log(y),1);
        power_spectrum_slope2(i) = p(1);
    end
    
    for ii = 1:n_steps_all        
        offset = step_size*(ii-1-n_steps);
        if isfield(temperature.states,'Bad')
            idx = find(~InIntervals(ripples.peaks+offset,temperature.states.cooling) & ~InIntervals(ripples.peaks+offset,temperature.states.Bad));
        else
            idx = find(~InIntervals(ripples.peaks+offset,temperature.states.cooling));
        end
        
        ripples.temperature = interp1(temperature.timestamps_slow,temperature.data_slow, ripples.peaks+offset);
        ripples.states = interp1(SleepState.idx.timestamps,SleepState.idx.states, ripples.peaks+offset);
        ripples.rate = interp1(t_minutes(2:end),ripple_rate, ripples.peaks+offset);
        ripples.theta_delta_ratio = interp1(t,bands.ratio, ripples.peaks+offset);
        ripples.power_spectrum_slope2 = interp1(t,power_spectrum_slope2, ripples.peaks+offset);
        
        % Ripple frequency - leave one out
        X0 = [ripples.temperature,ripples.states,ripples.rate,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];
        X1 = [ripples.states,ripples.rate,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];          % out: temperature
        X2 = [ripples.temperature,ripples.states,ripples.rate,ripples.theta_delta_ratio];                % out: power_spectrum_slope
        X3 = [ripples.temperature,ripples.rate,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];     % out: states
        X4 = [ripples.temperature,ripples.states,ripples.theta_delta_ratio,ripples.power_spectrum_slope2];   % out: rate
        X5 = [ripples.temperature,ripples.states,ripples.rate,ripples.power_spectrum_slope2];                % out: theta_delta_ratio
        
        mdl0 = fitlm(X0(idx,:),ripples.peakFrequency(idx));
        mdl1 = fitlm(X1(idx,:),ripples.peakFrequency(idx));
        mdl2 = fitlm(X2(idx,:),ripples.peakFrequency(idx));
        mdl3 = fitlm(X3(idx,:),ripples.peakFrequency(idx));
        mdl4 = fitlm(X4(idx,:),ripples.peakFrequency(idx));
        mdl5 = fitlm(X5(idx,:),ripples.peakFrequency(idx));
        
%         mld_results1(:,ii,kk) = [mdl1.RMSE,mdl2.RMSE,mdl3.RMSE,mdl4.RMSE,mdl5.RMSE];
        mld_results1(:,ii,kk) =   [mdl1.RMSE,mdl2.RMSE,mdl3.RMSE,mdl4.RMSE,mdl5.RMSE]-mdl0.RMSE;
        
        % Error by single metric
        mdl1 = fitlm(ripples.temperature(idx),ripples.peakFrequency(idx));
        mdl2 = fitlm(ripples.power_spectrum_slope2(idx),ripples.peakFrequency(idx));
        mdl3 = fitlm(ripples.states(idx),ripples.peakFrequency(idx));
        mdl4 = fitlm(ripples.rate(idx),ripples.peakFrequency(idx));
        mdl5 = fitlm(ripples.theta_delta_ratio(idx),ripples.peakFrequency(idx));
        
        mld_results3(:,ii,kk) = [mdl1.RMSE,mdl2.RMSE,mdl3.RMSE,mdl4.RMSE,mdl5.RMSE];
%         mld_results3(:,ii,kk) = mld_results3(:,ii,kk)-mean(mld_results3(:,ii,kk));
    end
    
    figure('Position',[100 100 1200 800])
    subplot(3,2,1)
    for ii = 1:n_steps_all
        plot([1:5],mld_results1(:,ii,kk),'color',colors(ii,:)), hold on
    end
    plot([1:5],mld_results1(:,n_steps+1,kk),'color','k')
    title('Leave one out')
    subplot(3,2,3)
    imagesc([1:5],temporal_offsets,mld_results1(:,:,kk)')
    xticks([1:5]), xticklabels(labels1), hold on, title(basename)
    
    subplot(3,2,2)
    for ii = 1:n_steps_all
        plot([1:5],mld_results3(:,ii,kk),'color',colors(ii,:)), hold on
    end
    plot([1:5],mld_results3(:,n_steps+1,kk),'color','k')
    title('Single predictor')
    subplot(3,2,4)
    imagesc([1:5],temporal_offsets,mld_results3(:,:,kk)')
    xticks([1:5]), xticklabels(labels1), hold on, title(basename)
    subplot(3,2,5)
    for ii = 1:5
        plot(temporal_offsets,mld_results1(ii,:,kk),'color',colors2(ii,:)), hold on
    end
    subplot(3,2,6)
    for ii = 1:5
        plot(temporal_offsets,mld_results3(ii,:,kk),'color',colors2(ii,:)), hold on
    end
    kk = kk+1;
    drawnow
end
% colormap(autumn)

image1 = mean(mld_results1,3);
image31 = mean(mld_results3,3);
image1_std = std(mld_results1,0,3);
image31_std = std(mld_results3,0,3);
figure
subplot(3,2,1)
for ii = 1:n_steps_all
    plot([1:5],image1(:,ii),'color',colors(ii,:)), hold on
end
plot([1:5],image1(:,n_steps+1),'color','k')

xticks([1:5]), title('Ripple frequency - leave one out'), xlabel('Leave one out'), ylabel('RMSE'), xlim([0.5,5.5])
subplot(3,2,3)
imagesc([1:5],temporal_offsets,image1'), hold on
% xticks([1:5]), xticklabels(labels1), 
title('Ripple frequency - leave one out'), xlabel('Leave one out'), ylabel('RMSE'), xlim([0.5,5.5])
% xtickangle(45)
% Individual fits
subplot(3,2,2)
for ii = 1:n_steps_all
    plot([1:5],image31(:,ii),'color',colors(ii,:)), hold on
end
plot([1:5],image31(:,n_steps+1),'color','k')

xticks([1:5]),  title('Ripple frequency - Single predictor'), xlabel('Single predictor'), ylabel('RMSE'), xlim([0.5,5.5])
subplot(3,2,4)
imagesc([1:5],temporal_offsets,image31'), hold on
% xticks([1:5]), xticklabels(labels1), 
title('Ripple frequency, single predictor'), xlabel('Individual fits'), ylabel('RMSE'), xlim([0.5,5.5])
% xtickangle(45)
subplot(3,2,5)
for ii = 1:5
    plot(temporal_offsets,image1(ii,:)), hold on, grid on, xlabel('Time (sec)'), ylabel('RMSE')
end
subplot(3,2,6)
for ii = 1:5
    plot(temporal_offsets,image31(ii,:)), hold on, grid on, xlabel('Time (sec)'), ylabel('RMSE')
end
legend(labels1)


figure
subplot(2,2,1)
for ii = 1:n_steps_all
    plot([1:5],image1(:,ii),'color',colors(ii,:)), hold on
end
plot([1:5],image1(:,n_steps+1),'color','k')
errorbar([1:5],image1(:,n_steps+1),image1_std(:,n_steps+1))


xticks([1:5]), title('Ripple frequency - leave one out'), xlabel('Leave one out'), ylabel('RMSE'), xlim([0.5,5.5])

subplot(2,2,2)
for ii = 1:n_steps_all
    plot([1:5],image31(:,ii),'color',colors(ii,:)), hold on
end
plot([1:5],image31(:,n_steps+1),'color','k')
errorbar([1:5],image31(:,n_steps+1),image31_std(:,n_steps+1))
xticks([1:5]),  title('Ripple frequency - Single predictor'), xlabel('Single predictor'), ylabel('RMSE'), xlim([0.5,5.5])

subplot(2,2,3)
for ii = 1:5
    plot(temporal_offsets,image1(ii,:)), hold on, grid on, xlabel('Time (sec)'), ylabel('RMSE')
end

subplot(2,2,4)
for ii = 1:5
    plot(temporal_offsets,image31(ii,:)), hold on, grid on, xlabel('Time (sec)'), ylabel('RMSE')
end
legend(labels1)


% Statistics
y1 = squeeze(mld_results1(:,121,:));
y2 = squeeze(mld_results3(:,121,:));
y2 = y2-mean(y2);
[k,p] = kstest2(y1(1,:),y1(2,:))
[stats1] = groupStats(num2cell(y1,2),[],'repeatedMeasures',true)
[stats2] = groupStats(num2cell(y2,2),[],'repeatedMeasures',true)


%% Histogram of temperature across sessions and subjects
% Temperature offset: 'Peter_MS22_180628_120341_concat',
sessions = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', 'Peter_MS13_171130_121758_concat', ...
    'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180627_143449_concat', 'Peter_MS21_180712_103200_concat',  ...
    'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat',...
    'Temp_R05_20210101','Temp_R07_20210219','Temp_R05_20210131','Temp_R08_20210224','Temp_R08_20210306','Temp_R09_20210404'};


temp_all1 = [];
temp_awake1 = [];
temp_nonREM1 = [];
temp_REM1 = [];
tempmicroArousals1 = [];
temp_means = [];
for k = 1:numel(sessions)

    disp([num2str(k),'/',num2str(numel(sessions))])
    sessionID = k;
    basename = sessions{sessionID};
    
    animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
    if ismac
        basepath = fullfile('/Volumes/Samsung_T5/GlobusDataFolder',animal_subject,basename);
    elseif strcmp(user_name,'Peter')
%         basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename);
        basepath = fullfile('D:\',animal_subject,basename); % SSD
    else
        basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
    end
    cd(basepath)
    session = loadSession(basepath,basename); % Loading session info
    
    % Temperature
    temperature = loadStruct('temperature','timeseries','session',session);
    if ~isfield(temperature,'data')
        temperature.data = temperature.temp;
    end
    if ~isfield(temperature.states,'control')
        temperature = StateExplorer(temperature);
        saveStruct(temperature,'timeseries','session',session);
    end
    temperature.data_slow = [];
    temperature.timestamps_slow = [];
    for i = 1:floor(numel(temperature.data)/(temperature.sr))
        temperature.data_slow(i) = nanmean(temperature.data(temperature.sr*(i-1)+1:i*temperature.sr));
        temperature.timestamps_slow(i) = i;
    end

    idx1 = ~InIntervals(temperature.timestamps_slow,temperature.states.control);
    temperature.data_slow(idx1) = nan;
    temp_bins = 33:0.2:40;
%     temp_bins = temp_bins-20
    temp_bins2 = temp_bins(1:end-1)-0.1;
    figure
    norm1 = sum(temperature.data_slow~=nan);
    temp_all = histcounts(temperature.data_slow,temp_bins)/norm1;
    plot(temp_bins2,temp_all,'k'), hold on, xlabel('Temperature (°C)'), ylabel('Distribution'), title(basename)
    
    % States
    SleepState = loadStruct('SleepState','states','session',session);
    
    idx_awake = SleepState.idx.states==1; % Awake
    idx_nonREM = SleepState.idx.states==3; % nonREM
    idx_REM = SleepState.idx.states==5; % REM
    idx_microArousals = SleepState.idx.states==4; % Micro arousals
    temp_awake = histcounts(temperature.data_slow(idx_awake),temp_bins)/norm1;
    temp_nonREM = histcounts(temperature.data_slow(idx_nonREM),temp_bins)/norm1;
    temp_REM = histcounts(temperature.data_slow(idx_REM),temp_bins)/norm1;
    temp_microArousals = histcounts(temperature.data_slow(idx_microArousals),temp_bins)/norm1;

    plot(temp_bins2,temp_awake,'g'), hold on
    plot(temp_bins2,temp_nonREM,'b'), hold on
    plot(temp_bins2,temp_REM,'r'), hold on
    
    temp_all1(k,:) = temp_all;
    temp_awake1(k,:) = temp_awake;
    temp_nonREM1(k,:) = temp_nonREM;
    temp_REM1(k,:) = temp_REM;
    temp_microArousals1(k,:) = temp_microArousals;
    
    temp_means.awake(k) = mean(temperature.data_slow(idx_awake),'omitnan');
    temp_means.nonREM(k) = mean(temperature.data_slow(idx_nonREM),'omitnan');
    temp_means.REM(k) = mean(temperature.data_slow(idx_REM),'omitnan');
    temp_means.microArousals(k) = mean(temperature.data_slow(idx_microArousals),'omitnan');
    figure, plot(temperature.timestamps_slow,temperature.data_slow)
end

figure
y = nanmean(temp_all1);
plot(temp_bins2,y,'k')
xlabel('Temperature (°C)'), hold on

y = nanmean(temp_awake1);
plot(temp_bins2,y,'g')

y = nanmean(temp_nonREM1);
plot(temp_bins2,y,'b')

y = nanmean(temp_REM1);
plot(temp_bins2,y,'r')

y = nanmean(temp_microArousals1);
plot(temp_bins2,y,'m')
legend({'All','Awake','nonREM','REM','Micro arousal'})
xlim([33,38])
text(0.1,0.9,['Awake=' num2str(mean(temp_means.awake, 'omitnan'),3),' +-',num2str(std(temp_means.awake, 'omitnan'),3)],'Color','k','Units','normalized')
text(0.1,0.7,['nonREM=' num2str(mean(temp_means.nonREM, 'omitnan'),3),' +-',num2str(std(temp_means.nonREM, 'omitnan'),3)],'Color','k','Units','normalized')
text(0.1,0.5,['REM=' num2str(mean(temp_means.REM, 'omitnan'),3),' +-',num2str(std(temp_means.REM, 'omitnan'),3)],'Color','k','Units','normalized')
text(0.1,0.3,['microArousals=' num2str(mean(temp_means.microArousals, 'omitnan'),3),' +-',num2str(std(temp_means.microArousals, 'omitnan'),3)],'Color','k','Units','normalized')

