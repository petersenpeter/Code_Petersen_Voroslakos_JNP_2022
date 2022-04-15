% Figure 2 Example session

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

%% % Peter_MS13_171129_105507_concat
basename = 'Peter_MS13_171129_105507_concat';

gausswin_steps = 160;
animal_subject = sessions_metadata.Animal_subject{strcmp(sessions_metadata.Session_name,basename)};
basepath = fullfile(basepath_root,animal_subject,basename);
% if ismac
%     
% elseif strcmp(user_name,'Peter')
%     basepath = fullfile('D:\',animal_subject,basename); % SSD
%     basepath = fullfile('Z:\Buzsakilabspace\LabShare\PeterPetersen\',animal_subject,basename); % NYU share
% else
%     basepath = fullfile('Z:\Homes\voerom01\Temperature\',animal_subject,basename);
% end
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

idx2 = find(InIntervals(temperature.timestamps,temperature.states.cooling));
temperature.timestamps(idx2) = [];
temperature.data(idx2) = [];

disp('Getting ripples from raw dat file')
ripples_intervals = getIntervalsFromDat(ripples.peaks,session,'nPull',Inf,'showIntervals',false,'keepIntervals_filt', true,'wfWin_sec', 0.080,'wfWinKeep', 0.020,'filtFreq',[110,180]);
ripples_intervals_all = ripples_intervals.intervals.filt;
timestamps = ripples_intervals.timeInterval;
disp('Determining peak frequency')
peakFrequency = [];
for i = 1:size(ripples_intervals_all,3)
    peakFrequency(i) = getRipplePeakFrequency(ripples_intervals_all(ripples.detectorinfo.detectionchannel1,:,i),timestamps/1000);
end
figure, plot(peakFrequency,'.')
ripples.peakFrequency = peakFrequency';

%% non-REM
ripples_nonREM = ripples;
SleepState = loadStruct('SleepState','states','session',session);
idx = InIntervals(ripples_nonREM.peaks,SleepState.ints.NREMstate);
ripples_nonREM.peaks = ripples_nonREM.peaks(idx);
ripples_nonREM.timestamps = ripples_nonREM.timestamps(idx,:);
ripples_nonREM.peakNormedPower = ripples_nonREM.peakNormedPower(idx);
ripples_nonREM.peakFrequency = ripples_nonREM.peakFrequency(idx);

lfp = bz_GetLFP(ripples_nonREM.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
ripfiltlfp = bz_Filter(lfp.data,'passband',[110,180],'filter','fir1');
[maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples_nonREM);
% PlotRippleStats(ripples,maps,data,stats)
% ripples_nonREM.peakFrequency = data.peakFrequency;
ripples_nonREM.duration = data.duration*1000;
ripples_nonREM.peakAmplitude = data.peakAmplitude/1000;
ripples_nonREM.ISIs = diff(ripples_nonREM.peaks); ripples_nonREM.ISIs = log10([ripples_nonREM.ISIs;ripples_nonREM.ISIs(end)]);

t_minutes = [0:ceil(max(ripples_nonREM.peaks)/30)];
SleepState.idx.states(SleepState.idx.states ~= 3) = 0;
SleepState.idx.states(SleepState.idx.states == 3) = 1;
state_normalizer = zeros(1,size(t_minutes,2)-1);
for j = 1:numel(state_normalizer)-1
    state_normalizer(j) = sum(SleepState.idx.states((j-1)*30+1:j*30));
end
state_normalizer(state_normalizer==0)=1;

ripple_rate = histcounts(ripples_nonREM.peaks/30,t_minutes);
ripple_rate = ripple_rate./state_normalizer;
%ripple_rate = ripple_rate/30;
ripples_nonREM.rate = interp1(t_minutes(2:end),ripple_rate, ripples_nonREM.peaks/30);
ripples_nonREM.rate(ripples_nonREM.rate>2)= 1;

ripples_nonREM.temperature = interp1(temperature.timestamps,temperature.data, ripples_nonREM.peaks);

%%
% Getting raw data and filtering it
% data3 = LoadBinary([session.general.baseName,'.lfp'],'nChannels',session.extracellular.nChannels,'channels',ripples_nonREM.detectorinfo.detectionchannel,'frequency',session.extracellular.sr);
% lfp1 = {};
% lfp1.data = data3;
% lfp1.samplingRate = session.extracellular.sr;
% lfp1.timestamps = [1:numel(data3)]/session.extracellular.sr;
% ripfiltlfp = bz_Filter(lfp1,'passband',[100 220],'filter','fir1');
% [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples_awake);

%% Awake
ripples_awake = ripples;
SleepState = loadStruct('SleepState','states','session',session);
idx = InIntervals(ripples_awake.peaks,SleepState.ints.WAKEstate);
ripples_awake.peaks = ripples_awake.peaks(idx);
ripples_awake.timestamps = ripples_awake.timestamps(idx,:);
ripples_awake.peakNormedPower = ripples_awake.peakNormedPower(idx);
ripples_awake.peakFrequency = ripples_awake.peakFrequency(idx);

% lfp = bz_GetLFP(ripples_awake.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename);
% ripfiltlfp = bz_Filter(lfp.data,'passband',[100 220],'filter','fir1');
% [maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples_awake);
[maps,data1] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples_awake);

% PlotRippleStats(ripples,maps,data,stats)
% ripples_awake.peakFrequency = data1.peakFrequency;
ripples_awake.duration = data1.duration*1000;
ripples_awake.peakAmplitude = data1.peakAmplitude/1000;
ripples_awake.ISIs = diff(ripples_awake.peaks); ripples_awake.ISIs = log10([ripples_awake.ISIs;ripples_awake.ISIs(end)]);

t_minutes = [0:ceil(max(ripples_awake.peaks)/30)];
SleepState.idx.states(SleepState.idx.states ~= 1) = 0;
state_normalizer = zeros(1,size(t_minutes,2)-1);
for j = 1:numel(state_normalizer)-1
    state_normalizer(j) = sum(SleepState.idx.states((j-1)*30+1:j*30));
end
state_normalizer(state_normalizer==0)=1;

ripple_rate = histcounts(ripples_awake.peaks/30,t_minutes);
ripple_rate = ripple_rate./state_normalizer;
%ripple_rate = ripple_rate/30;
ripples_awake.rate = interp1(t_minutes(2:end),ripple_rate, ripples_awake.peaks/30);
ripples_awake.rate(ripples_awake.rate>2)= 1;

ripples_awake.temperature = interp1(temperature.timestamps,temperature.data, ripples_awake.peaks);


%% Figure
fig1 = figure('name',basename,'position',[50,50,1200,900]); % Figure 1
subplot(3,3,1:2)
plot(ripples_nonREM.peaks/3600, ripples_nonREM.peakFrequency,'.b'), 
hold on,
plot(ripples_awake.peaks/3600, ripples_awake.peakFrequency,'.g')


% 30 sec windows
% nonREM
step_length = 60;
n_steps = ceil(max(ripples_nonREM.peaks/step_length));
peakFrequency_nonREM_average = nan(n_steps,1);
t_peakFrequency = [0:n_steps-1]*step_length+step_length/2;
temp_nonREM = nan(n_steps,1);
for i = 1:n_steps
    idx = ripples_nonREM.peaks>=(i-1)*step_length & ripples_nonREM.peaks<(i)*step_length;
    peakFrequency_nonREM_average(i) = mean(ripples_nonREM.peakFrequency(idx));
    temp_nonREM(i) = mean(ripples_nonREM.temperature(idx));
end
% plot(t_peakFrequency/3600,peakFrequency_nonREM_average,'.-r')

% Awake
step_length = 60;
n_steps = ceil(max(ripples_awake.peaks/step_length));
peakFrequency_awake_average = nan(n_steps,1);
temp_awake = nan(n_steps,1);
t_peakFrequency = [0:n_steps-1]*step_length+step_length/2;
for i = 1:n_steps
    idx = ripples_awake.peaks>=(i-1)*step_length & ripples_awake.peaks<(i)*step_length;
    peakFrequency_awake_average(i) = mean(ripples_awake.peakFrequency(idx));
    temp_awake(i) = mean(ripples_awake.temperature(idx));
end
% plot(t_peakFrequency/3600,peakFrequency_awake_average,'.-m')

text(0.05,1.2,['nonREM peakFrequency = ' num2str(mean(ripples_nonREM.peakFrequency),4),' Hz'],'Color','k','Units','normalized')
text(0.05,1.1,['Awake peakFrequency = ' num2str(mean(ripples_awake.peakFrequency),4),' Hz'],'Color','k','Units','normalized')

%legend({'Ripples','Temperature (zscored & aligned)','Ripple freq running average'},'AutoUpdate','off')
axis tight, %gridxy(cumsum(session.epochs.duration))
title('Ripple frequency'), xlabel('Time (hours)'), ylabel('Frequency (Hz)'),
ylim([110,180])

yyaxis right
plot(temperature.timestamps/3600,temperature.data), ylim([31,39]+0.5)

subplot(3,3,3)
plot(ripples_nonREM.temperature,ripples_nonREM.peakFrequency,'.b'), 
hold on
plot(ripples_awake.temperature,ripples_awake.peakFrequency,'.g')
title('Ripple rate vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Ripple rate (Hz)')

% plot(temp_nonREM,peakFrequency_nonREM_average,'.r')
% plot(temp_awake,peakFrequency_awake_average,'.m')

% nonREM
x = ripples_nonREM.temperature;
y1 = ripples_nonREM.peakFrequency;
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(0.05,1.1,['nonREM Slope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
[R,P] = corrcoef(x,y1);
text(0.05,1.15,['nonREM R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
y1 = nanconv(ripples_nonREM.peakFrequency,gausswin(gausswin_steps),'edge');
[R,P] = corrcoef(x,y1);
text(0.05,1.2,['nonREM R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')

% Awake
x = ripples_awake.temperature;
y1 = ripples_awake.peakFrequency;
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(0.95,1.1,['AwakeSlope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
[R,P] = corrcoef(x,y1);
text(0.95,1.15,['Awake R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
x = ripples_awake.temperature;
y1 = nanconv(ripples_awake.peakFrequency,gausswin(gausswin_steps),'edge');
[R,P] = corrcoef(x,y1);
text(0.95,1.2,['Awake R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
ylim([110,180])

subplot(3,3,4:5)
plot(ripples_nonREM.peaks/3600, ripples_nonREM.rate,'.b'), hold on,
plot(ripples_awake.peaks/3600, ripples_awake.rate,'.g')

axis tight,
title('Ripple rate (Hz)'), xlabel('Time (hours)'), ylabel('Frequency (Hz)'),
yyaxis right
plot(temperature.timestamps/3600,temperature.data), ylim([33,38])
text(0.05,1.2,['nonREM rate = ' num2str(mean(ripples_nonREM.rate),4),' Hz'],'Color','k','Units','normalized')
text(0.05,1.1,['Awake rate = ' num2str(mean(ripples_awake.rate),4),' Hz'],'Color','k','Units','normalized')


subplot(3,3,6)
plot(ripples_nonREM.temperature,ripples_nonREM.rate,'.b'), hold on
plot(ripples_awake.temperature,ripples_awake.rate,'.g')
title('Ripple rate vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Ripple rate (Hz)')
% nonREM
x = ripples_nonREM.temperature(~isnan(ripples_nonREM.rate));
y1 = ripples_nonREM.rate(~isnan(ripples_nonREM.rate));
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(0.05,1.15,['nonREM Slope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
[R,P] = corrcoef(x,y1);
text(0.05,1.1,['nonREMR = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
% Awake
x = ripples_awake.temperature(~isnan(ripples_awake.rate));
y1 = ripples_awake.rate(~isnan(ripples_awake.rate));
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(0.95,1.15,['Awake Slope: ' num2str(P(1),3),' Hz/degree'],'Color','k','Units','normalized')
[R,P] = corrcoef(x,y1);
text(0.95,1.1,['Awake R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')


subplot(3,3,[7,8])

subplot(3,3,[7,8])
plot(ripples_nonREM.peaks/3600, ripples_nonREM.duration,'.b'), hold on,
% plot(ripples_nonREM.peaks/3600, nanconv(ripples_nonREM.duration,gausswin(gausswin_steps),'edge'),'.m'), hold on,
plot(ripples_awake.peaks/3600, ripples_awake.duration,'.g'), hold on,
% plot(ripples_awake.peaks/3600, nanconv(ripples_awake.duration,gausswin(gausswin_steps),'edge'),'.m'), hold on,
plot(temperature.timestamps/3600,zscore(temperature.data)*10+60,'r')
axis tight
title('Ripple duration'), xlabel('Time (hours)'), ylabel('Duration (ms)'),

text(0.05,1.2,['nonREM duration = ' num2str(mean(ripples_nonREM.duration),4),' ms'],'Color','k','Units','normalized')
text(0.05,1.1,['Awake duration = ' num2str(mean(ripples_awake.duration),4),' ms'],'Color','k','Units','normalized')


subplot(3,3,9)
plot(ripples_nonREM.temperature,ripples_nonREM.duration,'.b'), hold on
% plot(ripples_nonREM.temperature, nanconv(ripples_nonREM.duration,gausswin(gausswin_steps),'edge'),'.m'), hold on,
plot(ripples_awake.temperature,ripples_awake.duration,'.g'), hold on
% plot(ripples_awake.temperature, nanconv(ripples_awake.duration,gausswin(gausswin_steps),'edge'),'.m'), hold on,
title('Ripple duration vs Temperature'), xlabel('Temperature (degree C)'), ylabel('Duration (ms, log10)') % nonREM
x = ripples_nonREM.temperature;
y1 = ripples_nonREM.duration;
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(0.05,1.1,['nonREM Slope: ' num2str(P(1),3),' ms/degree'],'Color','k','Units','normalized')
[R,P] = corrcoef(x,y1);
text(0.05,1.15,['nonREM R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
x = ripples_nonREM.temperature;
y1 = nanconv(ripples_nonREM.duration,gausswin(gausswin_steps),'edge');
[R,P] = corrcoef(x,y1);
text(0.05,1.2,['nonREM R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')

% Awake
x = ripples_awake.temperature;
y1 = ripples_awake.duration;
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'-k');
text(0.95,1.1,['Awake Slope: ' num2str(P(1),3),' ms/degree'],'Color','k','Units','normalized')
[R,P] = corrcoef(x,y1);
text(0.95,1.15,['Awake R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')
x = ripples_awake.temperature;
y1 = nanconv(ripples_awake.duration,gausswin(gausswin_steps),'edge');
[R,P] = corrcoef(x,y1);
text(0.95,1.2,['Awake R-smooth = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k','Units','normalized')


%% Plotting states
statesNames = fieldnames(SleepState.ints);
clr_states = jet(length(statesNames));
figure
for jj = 1:length(statesNames)
    statesData = SleepState.ints.(statesNames{jj});
    p1 = patch(double([statesData,flip(statesData,2)])',[0;0;1;1]*ones(1,size(statesData,1)),clr_states(jj,:),'EdgeColor',clr_states(jj,:),'HitTest','off'); hold on
end
legend(statesNames), axis tight

%% Plotting smoothed measures
window_sizes = [2,5,10:20:2000]; % in seconds
R_peak = [];
parfor kk = 1:length(window_sizes)
    kk
    x_bins = [0:window_sizes(kk):ceil(ripples.peaks(end)+window_sizes(kk))];
    
    peakFrequency_slow = [];
    temperature_slow = [];
    for i = 1:length(x_bins)-1
        peakFrequency_slow(i) = mean(ripples.peakFrequency(ripples.peaks > x_bins(i) & ripples.peaks < x_bins(i+1)));
        temperature_slow(i) = mean(temperature.data(temperature.timestamps > x_bins(i) & temperature.timestamps < x_bins(i+1)));
    end
    [R,P] = corrcoef(peakFrequency_slow(~isnan(peakFrequency_slow)),temperature_slow(~isnan(peakFrequency_slow)));
    R_peak(kk) = R(1,2);
    R_peak(kk)
end

R_peak_gausswin = [];
for kk = 1:length(window_sizes)
    [R,P] = corrcoef(ripples.temperature,nanconv(ripples.peakFrequency,gausswin(window_sizes(kk)),'edge'));
    R_peak_gausswin(kk) = R(1,2);
end
[R_min,idx] = max(abs(R_peak));


x_bins = [0:500:ceil(ripples.peaks(end)+window_sizes(kk))];
peakFrequency_slow = [];
temperature_slow = [];
for i = 1:length(x_bins)-1
    peakFrequency_slow(i) = mean(ripples.peakFrequency(ripples.peaks > x_bins(i) & ripples.peaks < x_bins(i+1)));
    temperature_slow(i) = mean(temperature.data(temperature.timestamps > x_bins(i) & temperature.timestamps < x_bins(i+1)));
end

figure
subplot(2,2,1)
% plot(window_sizes,R_peak), hold on
plot(window_sizes,R_peak_gausswin)

subplot(2,2,2)
plot(ripples.temperature,ripples.peakFrequency,'.b'), hold on
plot(ripples.temperature,nanconv(ripples.peakFrequency,gausswin(window_sizes(idx)),'edge'),'.g')
plot(temperature_slow,peakFrequency_slow,'.g')

subplot(2,3,4)
plot(ripples.peaks,ripples.peakFrequency,'.g')
title('Ripple frequency'), xlabel('Time (s)'), ylabel('Frequency (Hz)'),
ylim([100,200]+3)
yyaxis right
plot(temperature.timestamps,temperature.data), 
ylim([33,38])

subplot(2,3,5)
plot(x_bins(1:end-1),peakFrequency_slow,'.-g')
ylim([135,145]+3)
title('Ripple frequency bins'), xlabel('Time (s)'), ylabel('Frequency (Hz)'),
yyaxis right
plot(x_bins(1:end-1),temperature_slow), 
ylim([33,38])

subplot(2,3,6)
% plot(ripples.timestamps,nanconv(ripples.peakFrequency,gausswin(window_sizes(idx)),'edge'),'.g'), hold on
plot(ripples.timestamps,nanconv(ripples.peakFrequency,gausswin(400),'edge'),'.b')
title('Ripple frequency gausswin'), xlabel('Time (s)'), ylabel('Frequency (Hz)'),
ylim([135,145]+3)
yyaxis right
plot(temperature.timestamps,temperature.data),
ylim([33,38])

%% 
ripples_intervals = getIntervalsFromDat(ripples.peaks,session,'nPull',Inf,'showIntervals',false,'keepIntervals_filt', true);
ripples_intervals_all = ripples_intervals.intervals.filt;
peakFrequency = [];
timestamps = ripples_intervals.timeInterval;
disp('Determining peak frequency')
for i = 1:size(ripples_intervals_all,3)
    peakFrequency(i) = getRipplePeakFrequency(ripples_intervals_all(ripples.detectorinfo.detectionchannel1,:,i),timestamps/1000);
end
