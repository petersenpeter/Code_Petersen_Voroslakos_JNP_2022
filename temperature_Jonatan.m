clear all, close all

% Defining local paths
user_name=getenv('USERNAME');
computer_name = getenv('COMPUTERNAME');

if ismac
    disp('Mac computer detected')
    basepath_root = '/Volumes/Peter_SSD_4/';
    local_path = '/Volumes/Samsung_T5/Dropbox/Buzsakilab Postdoc/Matlab';
elseif strcmp(computer_name,'PETER')
    disp('Computer Peter detected')
    basepath_root = 'Z:\SUN-IN-Petersen-lab\EphysData\';
    local_path = 'K:\Dropbox\Buzsakilab Postdoc\Matlab';
elseif strcmp(computer_name,'ANASTASIA')
    disp('Computer Anastasia detected')
    basepath_root = 'Z:\SUN-IN-Petersen-lab\EphysData\';
    local_path = 'C:\Users\peter\Dropbox\Buzsakilab Postdoc\Matlab';
else
    disp('No computer detected')
    basepath_root = 'D:\';
    local_path = 'K:\Dropbox\Buzsakilab Postdoc\Matlab\';
end

if exist([local_path 'sessions_metadata_sleep_scoring.mat'])
    load([local_path 'sessions_metadata_sleep_scoring.mat'],'sessions_metadata_sleep_scoring');
else
    DOCID = '13HGeRteVxpiTYRpX6Cavq3caMUqFWIhZTGOCtqcLQ4s';
    session_temp = GetGoogleSpreadsheet(DOCID)';
    sessions_metadata_sleep_scoring = session_temp(:,2:end)';
    sessions_metadata_sleep_scoring = cell2table(sessions_metadata_sleep_scoring);
    sessions_metadata_sleep_scoring.Properties.VariableNames = session_temp(:,1);
    save([local_path 'sessions_metadata_sleep_scoring.mat'],'sessions_metadata_sleep_scoring');
end

%% Ripples analysis nonREM 
sessions_control_peter = {'Peter_MS12_170714_122034_concat', 'Peter_MS12_170715_111545_concat', ...
    'Peter_MS12_170716_172307_concat', 'Peter_MS12_170717_111614_concat', 'Peter_MS12_170719_095305_concat', 'Peter_MS13_171129_105507_concat', ...
    'Peter_MS13_171130_121758_concat', 'Peter_MS13_171128_113924_concat', 'Peter_MS13_171201_130527_concat', 'Peter_MS21_180629_110332_concat', ...
    'Peter_MS21_180627_143449_concat', 'Peter_MS21_180712_103200_concat', 'Peter_MS21_180628_155921_concat', 'Peter_MS22_180628_120341_concat', ...
    'Peter_MS22_180629_110319_concat', 'Peter_MS22_180720_110055_concat', 'Peter_MS22_180711_112912_concat'};

sessions_control_misi = {'Temp_R04_20201021','Temp_R04_20201023','Temp_R04_20201024','Temp_R04_20201027',...
    'Temp_R04_20201112','Temp_R04_20201113','Temp_R04_20201114','Temp_R05_20201219','Temp_R05_20201228',...
    'Temp_R05_20201229','Temp_R05_20210101','Temp_R05_20210102','Temp_R05_20210129','Temp_R05_20210130',...
    'Temp_R05_20210130_overnight','Temp_R05_20210131'};

sessions_control2 = {'Peter_MS10_170317_153237_concat','Peter_MS10_170314_163038','Peter_MS10_170315_123936',...
    'Peter_MS10_170307_154746_concat','Peter_MS13_171130_121758_concat','Peter_MS13_171110_163224_concat',...
    'Peter_MS21_180719_155941_concat','Peter_MS21_180625_153927_concat','Peter_MS21_180807_122213_concat',...
    'Peter_MS22_180719_122813_concat','Temp_R07_20210215','Temp_R07_20210216','Temp_R07_20210217',...
    'Temp_R07_20210219','Temp_R08_20210223','Temp_R08_20210224','Temp_R08_20210227','Temp_R08_20210304',...
    'Temp_R08_20210305','Temp_R08_20210306','Temp_R08_20210307','Temp_R09_20210404','Temp_R09_20210407','R2W3_10A2_20191014'};

sessions_all = sessions_control_misi;
sessionID = 10;
basename = sessions_all{sessionID};
idx = strcmp(sessions_metadata_sleep_scoring.Dataset,basename);
investigator = sessions_metadata_sleep_scoring.Repository{idx};
subject = sessions_metadata_sleep_scoring.Subject{idx};
basepath = fullfile(basepath_root,investigator,subject,basename);
cd(basepath)
session = loadSession(basepath,basename); % Loading session info

% Inspecting with a GUI
% session = gui_session(session);

% Visualizing the session in NeuroScope2:
% NeuroScope2

temperature = loadStruct('temperature','timeseries','session',session);

figure, plot(temperature.timestamps,temperature.data)

% figure, plot(temperature.temp)

%% Adding missing field
if ~isfield(session.inputs,'temperature') && isfield(session.inputs,'Temperature')
    session.inputs = renameStructField(session.inputs, 'Temperature', 'temperature');
end
if ~isfield(session.inputs,'AccelerometerX')
    session.inputs.AccelerometerX.channels = 1;
    session.inputs.AccelerometerX.inputType = 'aux';
end
if ~isfield(session.inputs,'AccelerometerY')
    session.inputs.AccelerometerY.channels = 2;
    session.inputs.AccelerometerY.inputType = 'aux';
end 
if ~isfield(session.inputs,'AccelerometerZ')
    session.inputs.AccelerometerZ.channels = 3;
    session.inputs.AccelerometerZ.inputType = 'aux';
end
session = gui_session(session);

%% Temperature
% Importing temperature data from thermocouple
%
% Here we are also using a struct-field in session.inputs.temperature with details (example values):
% session.inputs.temperature.channels = 1; % Channel number
% session.inputs.temperature.inputType = 'aux'; % data source
%
% aux data source (example values):
% session.timeSeries.aux
% session.timeSeries.aux.fileName = 'auxiliary.dat';
% session.timeSeries.aux.precision = 'uint16';
% session.timeSeries.aux.nChannels = 4;
% session.timeSeries.aux.sr = 5000;
% session.timeSeries.aux.leastSignificantBit = 37.399999999999999;

idx = strcmp(sessions_metadata_sleep_scoring.Dataset,basename);
downsample_samples = 100;
Temperature_sensor_type = sessions_metadata_sleep_scoring.Temperature_sensor_type{idx};
session = loadSession; % Loading session info

if ~isfield(session.inputs,'temperature')
    session.inputs = renameStructField(session.inputs, 'Temperature', 'temperature');
    saveStruct(session);
end

Temperature_sensor_type = 'thermistor_20210';
temperature_pre = loadStruct('temperature','timeseries','session',session);
temperature = loadIntanAnalog('session',session,'dataName','temperature','container_type','timeseries','processing',Temperature_sensor_type,'down_sample',true,'downsample_samples',downsample_samples);
if isfield(temperature_pre,'states')
    temperature.states = temperature_pre.states;
    saveStruct(temperature,'timeseries','session',session);
end

%%
% Loading an existing Temperature struct
temperature = loadStruct('temperature','timeseries','session',session);

% Plotting temperature timeseries
figure, plot(temperature.timestamps,temperature.data)

% GUI for curation of temperature data (e.g. adding states)
temperature = StateExplorer(temperature);
saveStruct(temperature,'timeseries','session',session);

% states: control, cooling, heating 
% temperature.states.cooling
% temperature.states.control

%% States
% Perform state scoring (function from Buzcode)
if ~exist(fullfile(basepath,[basename,'.SleepState.states.mat']),'file')
    disp('Performing sleep scoring')
    SleepScoreMaster(basepath);
end
% Manual curation of states in TheStateEditor (function from Buzcode)
TheStateEditor(basename);

% Loading an existing SleepState struct
SleepState = loadStruct('SleepState','states','session',session);

% Plotting states
ylim1 = [0 1];
t1 = 0;
t2 = inf;

figure
plot_states(SleepState.ints,t1,t2,ylim1)


%% Accelerometer data

channels = [session.inputs.AccelerometerX.channels,session.inputs.AccelerometerY.channels,session.inputs.AccelerometerZ.channels];
% channels = session.inputs.Accelerometer.channels;

accelerometer3D = loadIntanAnalog('session',session,'dataName','accelerometer3D','data_source_type','aux','container_type','timeseries','channels',channels,'processing','accelerometer','saveMat', false);

% 1D estimate of locomotion
accelerometer1D = accelerometer3D;
accelerometer1D.data = sum(accelerometer1D.data.^2,2);
saveStruct(accelerometer1D,'timeseries','session',session);
figure, plot(accelerometer1D.timestamps,accelerometer1D.data)
ylabel('Voltage'), xlabel('Time (s)'), axis tight

% Loading the data again
% accelerometer1D = loadStruct('accelerometer1D','timeseries','session',session);

%% Ripples
% Detecting ripples
ripples = ce_FindRipples(session,'thresholds', [18 48]/session.extracellular.leastSignificantBit, ...
    'passband', [80 240], 'EMGThresh', 0.8, 'durations', [20 150],'saveMat',false,'absoluteThresholds',true,'show','off');

% Loading an existing ripples struct
ripples = loadStruct('ripples','events','session',session);

% Manually curating ripples
NeuroScope2('session',session,'events','ripples')

% Manually flagged ripples (done via CellExplorer)
ripples.flagged; % Flags are stored as indexes

% Removing flagged ripples
ripples2 = ripples; % Renaming the ripples struct to avoid overwriting the original data
ripples2.peaks(ripples.flagged) = [];
ripples2.timestamps(ripples.flagged) = [];
ripples2.peakNormedPower(ripples.flagged) = [];
ripples2.peakFrequency(ripples.flagged) = [];

% filtering ripple events by the brainstate NREMstate
idx = InIntervals(ripples.peaks,SleepState.ints.NREMstate);
ripples.peaks = ripples.peaks(idx);
ripples.timestamps = ripples.timestamps(idx,:);
ripples.peakNormedPower = ripples.peakNormedPower(idx);
% ripples.peakFrequency = ripples.peakFrequency(idx);

% Getting ripples from raw dat file
ripples_intervals = getIntervalsFromDat(ripples.peaks,session,'nPull',1000);

figure, plot(ripples_intervals.timeInterval_all,ripples_intervals.filtIntervals_all(ripples.detectorinfo.detectionchannel1,:));

% Getting LFP data for the ripple analysis
lfp = bz_GetLFP(ripples.detectorinfo.detectionchannel,'basepath',basepath,'basename',basename); % function is from buzcode
ripfiltlfp = bz_Filter(lfp.data,'passband',[110,180],'filter','fir1'); % function is from buzcode
[maps,data] = bz_RippleStats(ripfiltlfp,lfp.timestamps,ripples); % function is from buzcode
    

%% Intan load digital series

% Digital data (digitalseries)
intanDig = loadIntanDigital(session);

% After this you can load the generated file:
intanDig = loadStruct('intanDig','digitalseries','session',session);


%% Raw data
% Getting an interval from 100sec to 300sec for channel 2
start = 100; % Can be multiple start times as well
duration = 200; % Duration of traces to load
channels = 2; % channels to load - can be a list as well
data_out = loadBinaryData('session',session,'channels',channels,'start',start,'duration',duration);
traces = session.extracellular.leastSignificantBit * double(data_out);

% Plotting the second channel of the first interval
figure, plot(traces)


%% Loading spikes struct
spikes = loadSpikes('session',session);


%% Generating a session summary figure (brain temperature, brain states, accelerometer)

% Loading structures
temperature = loadStruct('temperature','timeseries','session',session);
SleepState = loadStruct('SleepState','states','session',session);
accelerometer1D = loadStruct('accelerometer1D','timeseries','session',session);

figure,
subplot(5,1,[1,2])
plot(temperature.timestamps,temperature.data)
ylabel('Temperature (C)'), xlabel('Time (s)'), axis tight

subplot(5,1,3)
ylim1 = [0 1];
t1 = 0;
t2 = inf;
plot_states(SleepState.ints,t1,t2,ylim1), 
ylabel('States'), xlabel('Time (s)'), axis tight

subplot(5,1,4)
plot(accelerometer1D.timestamps,accelerometer1D.data)
ylabel('V^2'), xlabel('Time (s)'), axis tight

subplot(5,1,5)
if isfield(temperature,'states')
    ylim1 = [0 1];
    t1 = 0;
    t2 = inf;
    plot_states(temperature.states,t1,t2,ylim1),
    ylabel('States'), xlabel('Time (s)'), axis tight
end
