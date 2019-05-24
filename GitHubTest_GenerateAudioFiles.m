%--------------------------------------------------------------------------
% GitHubTest_GenerateAudioFiles - Loading DNN inferenced data and   
% reconstruct to the waveform of speech signals for white- and black-box 
% measurement.
% Note that the clean speech signals are from Grid corpous (downsampled to 
% 16 kHz) dataset and noise signals are from ChiMe-3 dataset. Signals in 
% both datasets are selected differently compared to training stage.
% Test files number: 20(files per speaker) * 4(speakers) * 3 sec. 
% * 4(noise type) = 960 sec. = 160 generated files.
% 
% Given data:
%             Grid corpous (clean speech) and ChiMe-3 (noise) datasets.
%             test_s_hat                : masked noisy speech
%             test_s_tilt               : masked clean speech
%             test_n_tilt               : masked noise
%             y_phase, s_phase, n_phase : phase information 
%         
% Output data:
%             All speech waveforms can be choosen to be saved or not.
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
% --- Set the noise levels:
% -21 for -5 dB SNR, -26 for 0 dB SNR, -31 for 5dB SNR, -36 for 10dB SNR, 
% -41 for 15dB SNR, -46 for 20dB SNR
noi_lev = -21; % Change "noi_lev" for various SNRs
save_files_flag = 0; % 1- Save all generated files; 0- Not save 
modle_type_str_vec = {'weight_filter_AMR_direct_freqz', 'baseline'}; % run both models to compare
noi_situ_model_str = '6snrs'; 
speaker_num_test = 4;
num_file_test = 20;  % number of files per speaker
file_sec = 6; % Generated files have 6 seconds duration
Fs = 16000;
% -- Frequency domain parameters
fram_leng = 256; % window length
fram_shift = fram_leng/2; % frame shift
freq_coeff_leng = fram_shift + 1; % half-plus-one frequency coefficients

% --- Deirctories 
database_dir = '.\Audio Data\grid coupus 16khz\';
database_noi_dir = '.\Audio Data\16khz noise\';
subdirs{1} = 's17\';
subdirs{2} = 's18\';
subdirs{3} = 's19\';
subdirs{4} = 's20\';

%% Generate clean, noise, and noisy speech
% -- Use all noise types per SNR
noi_type_str_vec = {'PED', 'CAF', 'STR', 'BUS'};
for k_noi_type = 1 : length(noi_type_str_vec)
    noi_type = noi_type_str_vec{k_noi_type};
    if strcmp(noi_type, 'PED')
        noi_file_name = [database_noi_dir 'ped\BGD_150203_020_' noi_type '.CH1.wav']; 
    elseif strcmp(noi_type, 'CAF')
        noi_file_name = [database_noi_dir 'cafe\BGD_150203_010_' noi_type '.CH1.wav']; 
    elseif strcmp(noi_type, 'STR')
        noi_file_name = [database_noi_dir 'street\BGD_150203_010_' noi_type '.CH1.wav']; 
    elseif strcmp(noi_type, 'BUS')
        noi_file_name = [database_noi_dir 'bus\BGD_150204_010_' noi_type '.CH1.wav']; 
    end


    % --- Generate s, n, y with set SNR
    num_file = 0;
    % --- Load test speech files 
    for subdir_index= 1:speaker_num_test
    database_file = dir([database_dir subdirs{subdir_index}]);
    for ff=1:length(database_file)
        if ~strcmp(database_file(ff).name(1), '.')
            if database_file(ff).isdir
                database_file_sub = dir([database_dir subdirs{subdir_index} database_file(ff).name '\*.wav']);  

                for kk = 1:num_file_test % Num of files per folder
                 in_file = [database_dir subdirs{subdir_index} database_file(ff).name '\' database_file_sub(kk).name];
                 fprintf('  %s --> \n', in_file); 
                 num_file = num_file + 1;

                  %--- read .wav file by loadshort function
                  [speech_file_wav,fs] = audioread(in_file);  
                  speech_file=speech_file_wav.*(2^15);   
                  speech_int16= int16(speech_file);

                  %--- normalize  to -26 dB 
                  [act_lev_speech, rms_lev_speech, gain_speech] = actlev('-sf 16000 -lev -26', speech_int16);
                  speech_scaled_int16 = speech_int16 * gain_speech;
                  speech_scaled=double(speech_scaled_int16);

                  %--- save to a matrix
                  s_mat(:,num_file) = speech_scaled;

                end
            end
        end
    end
    end
    s_vec = s_mat(:);
    s_vec_leng = length(s_vec);
    clear s_mat;

    % --- Load noise files
    [noi_test_wav,~] = audioread(noi_file_name);
    noi_test_wav = noi_test_wav .* 2^15;
    % --- Trim to same length as s_vec: n_vec
    n_vec = noi_test_wav(1:s_vec_leng);
    n_vec = int16(n_vec);
    % --- Make the noise level according to the set SNR
    noise_contr = ['-sf 16000 -lev ' num2str(noi_lev) ' -rms'];
    [~, ~, gain_noise] = actlev(noise_contr, n_vec);
    n_vec_scale = n_vec .* gain_noise;
    n_vec_scale = double(n_vec_scale);
    % --- Mix to generate noisy speech: y_vec
    y_vec_per_noitype(:,k_noi_type) = s_vec + n_vec_scale;
    % --- Document for each noise type
    n_vec_per_noitype(:,k_noi_type) = n_vec_scale;
    s_vec_per_noitype(:,k_noi_type) = s_vec;
end
y_vec_all = y_vec_per_noitype(:);
n_vec_all = n_vec_per_noitype(:);
s_vec_all = s_vec_per_noitype(:);
s_vec_all_leng = length(s_vec_all);

y_vec_all = y_vec_all.';
n_vec_all = n_vec_all.';
s_vec_all = s_vec_all.';
    
%% Generate s_tilde, n_tilde, and s_hat speech
% --- Run for all modle_type_str
for k_model_type = 1 : length(modle_type_str_vec)
    modle_type_str = modle_type_str_vec{k_model_type};
    % --- Load Python output & load phase matrix
    load(['./test results/mask_dnn_' modle_type_str '_s_hat_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);
    load(['./test results/mask_dnn_' modle_type_str '_s_tilt_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);
    load(['./test results/mask_dnn_' modle_type_str '_n_tilt_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);
    load(['./test data/test_phase_mats_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);
    
    % --- Generate long vectors from frames for 3 signals: s_hat, s_tilde, n_tilde
    num_fram = size(test_s_hat,1);
    s_hat_vec = zeros(1,(num_fram+1)*fram_shift);
    s_tilt_vec = zeros(1,(num_fram+1)*fram_shift);
    n_tilt_vec = zeros(1,(num_fram+1)*fram_shift);
    y_phase = y_phase.';
    s_phase = s_phase.';
    n_phase = n_phase.';
    s_hat_mat = zeros(num_fram,fram_leng);
    s_tilt_mat = zeros(num_fram,fram_leng);
    n_tilt_mat = zeros(num_fram,fram_leng);
    
    for k = 1 : num_fram
        fft_s_hat_half = test_s_hat(k,:);
        fft_s_hat = [fft_s_hat_half, fliplr(fft_s_hat_half(2:fram_shift))];
        fft_s_hat_cmpx = fft_s_hat .* exp(1j .* y_phase(k,:));
        s_hat_temp = real(ifft(fft_s_hat_cmpx,fram_leng));
        s_hat_mat(k,:) = s_hat_temp;
        
        fft_s_tilt_half = test_s_tilt(k,:);
        fft_s_tilt = [fft_s_tilt_half, fliplr(fft_s_tilt_half(2:fram_shift))];
        fft_s_tiltt_cmpx = fft_s_tilt .* exp(1j .* s_phase(k,:));
        s_tilt_temp = real(ifft(fft_s_tiltt_cmpx,fram_leng));
        s_tilt_mat(k,:) = s_tilt_temp;
        
        fft_n_tilt_half = test_n_tilt(k,:);
        fft_n_tilt = [fft_n_tilt_half, fliplr(fft_n_tilt_half(2:fram_shift))];
        fft_n_tiltt_cmpx = fft_n_tilt .* exp(1j .* n_phase(k,:));
        n_tilt_temp = real(ifft(fft_n_tiltt_cmpx,fram_leng));
        n_tilt_mat(k,:) = n_tilt_temp;
        
        % -- Form long vector with overlap-add
        if k == 1
            s_hat_vec(1:fram_shift) = s_hat_mat(1,1:fram_shift);
            s_tilt_vec(1:fram_shift) = s_tilt_mat(1,1:fram_shift);
            n_tilt_vec(1:fram_shift) = n_tilt_mat(1,1:fram_shift);
        elseif k > 1
            s_hat_nach = s_hat_mat(k-1,freq_coeff_leng:fram_leng);
            s_hat_vor  = s_hat_mat(k,1:fram_shift);
            s_hat_vec(1+(k-1)*fram_shift : k*fram_shift) = s_hat_nach + s_hat_vor;
            
            s_tilt_nach = s_tilt_mat(k-1,freq_coeff_leng:fram_leng);
            s_tilt_vor  = s_tilt_mat(k,1:fram_shift);
            s_tilt_vec(1+(k-1)*fram_shift : k*fram_shift) = s_tilt_nach + s_tilt_vor;
            
            n_tilt_nach = n_tilt_mat(k-1,freq_coeff_leng:fram_leng);
            n_tilt_vor  = n_tilt_mat(k,1:fram_shift);
            n_tilt_vec(1+(k-1)*fram_shift : k*fram_shift) = n_tilt_nach + n_tilt_vor;
        end
        
        % -- Display progress
         if mod(k,12000) == 0,
            disp(['Percentage of frames formed: ' num2str( (k/num_fram)* 100) '%']);
        end
    end
    
    % --- Seperate to two-sentence files for measurements
    file_leng = file_sec * Fs;
    file_num  = (num_fram+1)*fram_shift/file_leng;
    for k = 1 : file_num
        ind_vor  = 1 + (k-1) * file_leng;
        ind_nach = k * file_leng;
        
        % -- Form the files
        s_hat_temp = s_hat_vec(ind_vor : ind_nach);
        s_tilt_temp = s_tilt_vec(ind_vor : ind_nach);
        n_tilt_temp = n_tilt_vec(ind_vor : ind_nach);
        y_vec_temp = y_vec_all(ind_vor : ind_nach);
        s_vec_temp = s_vec_all(ind_vor : ind_nach);
        n_vec_temp = n_vec_all(ind_vor : ind_nach);
        
        % -- Save files or not
        if save_files_flag == 1
            saveshort(s_hat_temp,['./generated_files/test_data/s_hat_test_data_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_' modle_type_str '_' num2str(k) '.raw']);
            saveshort(s_tilt_temp,['./generated_files/test_data/s_tilde_test_data_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_' modle_type_str '_' num2str(k) '.raw']);
            saveshort(n_tilt_temp,['./generated_files/test_data/n_tilde_test_data_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_' modle_type_str '_' num2str(k) '.raw']);
            saveshort(y_vec_temp,['./generated_files/test_data/y_test_data_snr_' num2str(noi_lev) '_' num2str(k) '.raw']);
            saveshort(s_vec_temp,['./generated_files/test_data/s_' num2str(k) '.raw']);
            saveshort(n_vec_temp,['./generated_files/test_data/n_test_data_snr_' num2str(noi_lev) '_' num2str(k) '.raw']);
        end
        
        % -- Possible white- and black-box measurements here ...
        
        % -- Display percentage
         if mod(k,32) == 0,
            disp(['Percentage of files generated: ' num2str( (k/file_num)* 100) '%']);
        end
    end

end
