% Sleep scoring from temperature data

% Gode sessioner at starte med:
% R2W3_10A2_20191014 : 24 hours session shown in figure 1 in paper (under MisiVoroslakos).
% Peter_MS13_171129_105507_concat : Shown in figure 2 in paper (under PeterPetersen).
% Peter_MS23_190826_100705 : 3 temperature probes - freely moving – no ephys (under PeterPetersen)..

clear all
basenames = {...
    'R2W3_10A2_20191014',...
    'Peter_MS13_171129_105507_concat',...
    'Peter_MS23_190826_100705'...
    };
basepaths = {...
    'Z:\SUN-IN-Petersen-lab\EphysData\MisiVoroslakos\R2W3\R2W3_10A2_20191014',...
    'Z:\SUN-IN-Petersen-lab\EphysData\PeterPetersen\MS13\Peter_MS13_171129_105507_concat',...
    ''...
    };

% Defining the session
session_id = 2;
basename = basenames{session_id};
basepath = basepaths{session_id};

cd(basepath)

% Visualizing the session in NeuroScope2:
% NeuroScope2

%% Loading metadata for session
session = loadSession(basepath,basename); % Loading session info

% Inspecting with a GUI
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

temperature = loadIntanAnalog('session',session,'dataName','temperature','container_type','timeseries','processing','thermocouple');

% Importing temperature data from thermistor
temperature = loadIntanAnalog('session',session,'dataName','temperature','data_source_type','adc','container_type','timeseries','processing','thermistor_10000','down_sample',true,'downsample_samples',20000);

% Loading an existing Temperature struct
temperature = loadStruct('temperature','timeseries','session',session);

% Plotting temperature timeseries
figure, plot(temperature.timestamps,temperature.data)

% Filtering temperature
[b1, a1] = butter(3, 2/temperature.sr*2, 'low');
temperature.filter = filtfilt(b1, a1, temperature.data);
figure, plot(temperature.timestamps,temperature.filter)

% GUI for curation of temperature data (e.g. adding states)
temperature = StateExplorer(temperature);
saveStruct(temperature,'timeseries','session',session);

% states: cooling/Cooling and heating/Heating
% temperature.states

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
accelerometer3D = loadIntanAnalog('session',session,'dataName','accelerometer3D','data_source_type','aux','container_type','timeseries','channels',channels);
figure, plot3(accelerometer3D.data(1,:),accelerometer3D.data(2,:),accelerometer3D.data(3,:))
ylabel('Voltage'), xlabel('Time (s)'), axis tight

% 1D estimate of locomotion
accelerometer1D = accelerometer3D;
accelerometer1D.data = sum(abs(accelerometer1D.data-mean(accelerometer1D.data,2)).^2);
saveStruct(accelerometer1D,'timeseries','session',session);
figure, plot(accelerometer1D.timestamps,accelerometer_data)
ylabel('Voltage'), xlabel('Time (s)'), axis tight

% Loading the data again
accelerometer1D = loadStruct('accelerometer1D','timeseries','session',session);

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
