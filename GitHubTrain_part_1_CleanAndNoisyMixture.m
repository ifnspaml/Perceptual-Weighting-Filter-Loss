%--------------------------------------------------------------------------
% GitHubTrain_part_1_CleanAndNoisyData - Loading clean speech and
% noise, generating mixture signal (noisy) with 6 SNRs, generating
% frame-wise frequency amplitudes for clean and noisy speech. 
% Note that the clean speech signals are from Grid corpous (downsampled to 
% 16 kHz) dataset and noise signals are from ChiMe-3 dataset. 
%
% Given data:
%             Grid corpous (clean speech) and ChiMe-3 (noise) datasets.
%         
% Output data:
%             s_speech             : whole clean speech signal 
%                                    (for part 2 usage) 
%             speech_fft_abs_clean : frequency amplitudes for clean speech
%                                    (for part 3 usage)
%             mixture_fft_abs      : frequency amplitudes for noisy speech
%                                    (for part 3 usage)
%
% 
% Technische Universität Braunschweig
% Institute for Communications Technology (IfN)
% Schleinitzstrasse 22
% 38106 Braunschweig
% Germany
% 2019 - 05 - 23 
% (c) Ziyue Zhao
%
% Use is permitted for any scientific purpose when citing the paper:
% Z. Zhao, S. Elshamy, and T. Fingscheidt, "A Perceptual Weighting Filter 
% Loss for DNN Training in Speech Enhancement", arXiv preprint arXiv: 
% 1905.09754.
%
%--------------------------------------------------------------------------
clear;
addpath(genpath(pwd));
% --- Settings 
num_snr_mix = 6; % Number of the mixed SNRs 
num_file = 100;  % number of files per speaker
speaker_num = 16;
Fs = 16000;
duration_per_file = 3;
% -- Set the noise levels:
% -21 for -5 dB SNR, -26 for 0 dB SNR, -31 for 5dB SNR, -36 for 10dB SNR, 
% -41 for 15dB SNR, -46 for 20dB SNR
noi_lev_vec = -21:-5:-46; 
% -- Frequency domain parameters
fram_leng = 256; % window length
fram_shift = fram_leng/2; % frame shift
freq_coeff_leng = fram_shift + 1; % half-plus-one frequency coefficients

% --- Input directories
database_dir = '.\Audio Data\grid corpus 16khz\';
noise_dir_1 = '.\Audio Data\16khz noise\ped\BGD_150211_040_PED.CH2.wav'; % 32 mins
noise_dir_2 = '.\Audio Data\16khz noise\street\BGD_150211_030_STR.CH2.wav';% 26 mins
noise_dir_3 = '.\Audio Data\16khz noise\cafe\cafe1\BGD_150204_030_CAF.CH5.wav';% 30 mins
subdirs = cell(1,1);
subdirs{01} = 's1\';
subdirs{02} = 's2\';
subdirs{03} = 's3\';
subdirs{04} = 's4\';
subdirs{05} = 's5\';
subdirs{06} = 's6\';  % s7 in original dataset
subdirs{07} = 's7\';  % s11 in original dataset
subdirs{08} = 's8\';  % s15 in original dataset
subdirs{09} = 's9\';  % s6 in original dataset
subdirs{10} = 's10\'; % s8 in original dataset
subdirs{11} = 's11\'; % s9 in original dataset
subdirs{12} = 's12\'; % s10 in original dataset
subdirs{13} = 's13\'; % s16 in original dataset
subdirs{14} = 's14\'; % s18 in original dataset
subdirs{15} = 's15\'; % s20 in original dataset
subdirs{16} = 's16\'; % s21 in original dataset

% --- Output directories
train_sspeech_dir = '.\train\speech_clean_s_speech.mat'; 
train_clean_dir = '.\train\speech_fft_abs_clean_6snrs.mat'; 
train_mixture_dir = '.\train\mixture_fft_abs_6snrs.mat';

%% Read clean speech and produce frequency amplitudes
% --- Loop for loading clean speech 
s1 = cell(1,1);
num1 = 0;
for subdir_index = 1:speaker_num
    database_file = dir([database_dir subdirs{subdir_index}]);
    for ff = 1:length(database_file)
        if ~strcmp(database_file(ff).name(1), '.')
            if database_file(ff).isdir
                database_file_sub = dir([database_dir subdirs{subdir_index} database_file(ff).name '\*.wav']);  
                for kk = 1:num_file % Num of files per language folder. 
                     in_file = [database_dir subdirs{subdir_index} database_file(ff).name '\' database_file_sub(kk).name];
                     fprintf('  %s --> \n', in_file); 

                     % -- read as .raw file 
                     [speech_file_wav,fs] = audioread(in_file);  
                     speech_file = speech_file_wav.*(2^15);   
                     speech_int16 = int16(speech_file);

                     % -- normalize to -26 dBoV
                     [act_lev_speech, rms_lev_speech, gain_speech] = actlev('-sf 16000 -lev -26', speech_int16);
                     speech_scaled_int16 = speech_int16 * gain_speech;
                     speech_scaled = double(speech_scaled_int16);

                     % -- save the processed data to different cells
                     num1 = num1+1;
                     s1{num1} = speech_scaled;
 
                end
            end            
        end
    end
end

% --- Document the length of each speech file and save to s1_speech
num_element1 = 0;
for nn=1:num1
    num_element1 = num_element1 + length(s1{1,nn});
end
s1_speech = zeros(num_element1,1);
 
% --- Concatenate all files to one vector
num_cal1 = 0;
for mm = 1:num1
    num_cal1 = num_cal1+length(s1{1,mm});
    s1_speech(num_cal1-length(s1{1,mm})+1:num_cal1,1) = s1{1,mm};
end
   
% --- Copy 6 times for 6 SNRs 
s_speech=[s1_speech;s1_speech;s1_speech;s1_speech;s1_speech;s1_speech];

% --- frame-wise FFT processing 
wd = hanning(fram_leng,'periodic');
num_frame = (floor(length(s1_speech)*num_snr_mix/fram_shift)-1);
speech_fft_abs_clean = zeros(freq_coeff_leng,num_frame);
clear s1 speech_file_wav speech_file speech_file speech_scaled_int16 speech_int16 speech_scaled
for jj=1:num_frame
    % -- Get frequency amplitude
    speech_wd = s_speech(1+fram_shift*(jj-1):fram_leng+fram_shift*(jj-1),1).*wd;  
    speech_fft = fft(speech_wd); % FFT for the clear speech
    fft_abs = abs(speech_fft); % get the amplitude spectrogram
    speech_fft_abs_clean(:,jj) = fft_abs(1:freq_coeff_leng);
    % -- Display progress
    if mod(jj,10000) == 0,
        disp(['Percentage of frames finished (FFT): ' num2str( (jj/num_frame)* 100) '%']);
    end
end

% --- Save the clean speech frequency amplitude (129 coeff. from 256 FFT points)
save(train_clean_dir,'speech_fft_abs_clean','-v7.3')
save(train_sspeech_dir,'s_speech','-v7.3');
clear s_speech

%% Read noise and produce frequency amplitudes for mixture
% --- read noise
[noise_wav1,~]=audioread(noise_dir_1); 
[noise_wav2,~]=audioread(noise_dir_2); 
[noise_wav3,~]=audioread(noise_dir_3); 
noise_raw1=noise_wav1.*(2^15); % transfer to raw file
noise_raw2=noise_wav2.*(2^15); 
noise_raw3=noise_wav3.*(2^15); 

% --- Concatenate all 6 noise files to one vector and trim 
noise_raw_all = [noise_raw1;noise_raw2;noise_raw3]; % 88 mins (enough for 80 mins, i.e., clean speech duration )
noise_raw = noise_raw_all(1:speaker_num*num_file*duration_per_file*Fs,1); % 16 speakers, 100 files, 3 sec.
noise_int16 = int16(noise_raw);
clear noise_wav1 noise_wav2 noise_wav3 noise_raw_all
clear noise_raw1 noise_raw2 noise_raw3 

% --- Adjust the noise level according to the set SNR
noise = cell(1,1);
num_n = 0;
for act_n = noi_lev_vec 
    num_n = num_n+1;
    noise_contr = ['-sf 16000 -lev ' num2str(act_n) ' -rms'];
    [~, ~, gain_noise] = actlev(noise_contr, noise_int16);
    noise_int16_scale = noise_int16.*gain_noise;
    noise_scale = double(noise_int16_scale);
    % [act_lev1, rms_lev1, gain1] = actlev('-sf 16000 -lev -26',int16(noise_scale));
    noise{num_n} = noise_scale;
end
clear noise_raw noise_int16 speech_scaled noise_int16_scale

% --- mix the speech with SNRs
mixed_speech_cell = cell(1,1);
l_mix = min(num_element1,length(noise_scale));% minimum length of s1_speech and noise_scale
for cc = 1:num_n
    mixed_speech_raw = noise{cc}(1:l_mix,1)+s1_speech(1:l_mix,1);
    mixed_speech_cell{cc} = mixed_speech_raw;
end
clear s1_speech noise mixed_speech_raw noise_scale

% --- Save to one matrix: mixed_speech
num_element2 = 0;
for nn = 1:num_n
    num_element2=num_element2+length(mixed_speech_cell{1,nn});
end
mixed_speech=zeros(num_element2,1);

num_cal2 = 0;
for mm = 1:num_n
    num_cal2 = num_cal2+length(mixed_speech_cell{1,mm});
    mixed_speech(num_cal2-length(mixed_speech_cell{1,mm})+1:num_cal2,1) = mixed_speech_cell{1,mm};
end
l_mix=num_element2;
clear mixed_speech_cell 

% --- FFT processing
wd = hanning(fram_leng,'periodic');
l_process = floor(l_mix/fram_shift)-1;
mixture_fft_abs = zeros(freq_coeff_leng,l_process);
for jj = 1:l_process
    speech_wd = mixed_speech(1+fram_shift*(jj-1):fram_leng+fram_shift*(jj-1),1).*wd;  %segment the clear speech using hanning window
    speech_fft = fft(speech_wd); % FFT for the noisy speech
    fft_abs = abs(speech_fft); % get the amplitude spectrogram
    mixture_fft_abs(:,jj) = fft_abs(1:freq_coeff_leng);
    % -- Display progress
    if mod(jj,10000) == 0,
        disp(['Percentage of frames finished: ' num2str( (jj/l_process)* 100) '%']);
    end
end

% --- Save mixture
save(train_mixture_dir,'mixture_fft_abs','-v7.3')






