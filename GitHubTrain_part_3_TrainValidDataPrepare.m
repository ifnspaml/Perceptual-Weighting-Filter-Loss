%--------------------------------------------------------------------------
% GitHubTrain_part_3_TrainValidDataPrepare - Loading data from part 1 and 2
% and generate the final training/validation data. 
% Note that the clean speech signals are from Grid corpous (downsampled to 
% 16 kHz) dataset and noise signals are from ChiMe-3 dataset. 
%
% Given data:
%             h_fft_abs_half_mat   : the matrix contains frame-wise 
%                                    weighting filter amplitude response, 
%                                    with the dimension as: (half-plus-one  
%                                    frequency bins, num. of frames)
%                                    (from part 2) 
%             speech_fft_abs_clean : frequency amplitudes for clean speech
%                                    (from part 1)
%             mixture_fft_abs      : frequency amplitudes for noisy speech
%                                    (from part 1)
%         
% Output data:
%             training_input_6snrs          : training input
%             validation_input_6snrs        : validation input
%             training_input_unnorm_6snrs   : training auxiliary input
%             validation_input_unnorm_6snrs : validation auxiliary input
%             training_target_6snrs         : training target
%             validation_target_6snrs       : validation target
%             ( Note that the above data dimension is num_frame X 129 )
%             mean_training_6snrs, std_training_6snrs : training data
%             statistics
%
% 
% Technische Universit�t Braunschweig
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
Fs = 16000;
num_snr_mix = 6; % Number of the mixed SNRs
speaker_num = 16;
duration_per_file = 3;
num_file = 100;
valid_rate = 0.2; % the rate of: validation data / whole data
% -- Frequency domain parameters
fram_leng = 256; % window length
fram_shift = fram_leng/2; % frame shift
freq_coeff_leng = fram_shift + 1; % half-plus-one frequency coefficients

% --- Input directories
train_clean_dir = '.\train\speech_fft_abs_clean_6snrs.mat'; 
train_wgh_filter_dir = '.\train\h_fft_abs_half_mat_AMR_direct_freqz_6snrs.mat';
train_mixture_dir = '.\train\mixture_fft_abs_6snrs.mat';

% --- Output directories
train_data_input_dir = '.\training data\training_input_6snrs.mat';
valid_data_input_dir = '.\training data\validation_input_6snrs.mat';
train_data_unnorm_input_dir = '.\training data\training_input_unnorm_6snrs.mat';
valid_data_unnorm_input_dir = '.\training data\validation_input_unnorm_6snrs.mat';
train_data_mean_dir = '.\training data\mean_training_6snrs.mat';
train_data_std_dir = '.\training data\std_training_6snrs.mat';
train_data_target_dir = '.\training data\training_target_6snrs.mat';
valid_data_target_dir = '.\training data\validation_target_6snrs.mat';
train_data_wgh_filter_dir = '.\training data\h_fft_abs_half_mat_AMR_direct_freqz_train_6snrs.mat';
valid_data_wgh_filter_dir = '.\training data\h_fft_abs_half_mat_AMR_direct_freqz_validation_6snrs.mat';


% --- Data processing for both Input and Target Data
data_type_str_vec = {'input_data', 'target_data'};
for data_type_k = 1 : length(data_type_str_vec)
    data_type_str = data_type_str_vec{data_type_k};
    if strcmp(data_type_str,'input_data')
        load(train_mixture_dir) % from part 1

        % --- Arrange to cells per SNR
        target_length = speaker_num*num_file*duration_per_file*Fs/fram_shift; % number of frames per SNR level
        SNR_mixture = cell(1,1);
        for snr = 1:num_snr_mix
            if snr==6
                SNR_mixture{snr}=mixture_fft_abs(:,1+(snr-1)*target_length:end);
            else 
                SNR_mixture{snr}=mixture_fft_abs(:,1+(snr-1)*target_length:snr*target_length);
            end
        end
        clear mixture_fft_abs

        % --- Arrange to cells per speaker per SNR
        target_length = num_file*3*Fs/fram_shift; % number of frames per speaker per SNR
        mixture_speaker = cell(1,1);
        for snr = 1:num_snr_mix
            for speaker = 1:speaker_num
                if speaker==16
                    mixture_speaker{snr,speaker} = SNR_mixture{snr}(:,1+(speaker-1)*target_length:end);
                else
                    mixture_speaker{snr,speaker} = SNR_mixture{snr}(:,1+(speaker-1)*target_length:speaker*target_length);
                end
            end
        end
        clear SNR_mixture

        % --- Arrange to cells for training/validation per speaker per SNR
        vali_input_cell = cell(1,1);
        training_input_cell = cell(1,1);
        target_length = floor(valid_rate*num_file*duration_per_file*Fs/fram_shift); % number of frames for validation per speaker per SNR: 1/5
        for snr = 1:num_snr_mix
            for speaker = 1:speaker_num
                vali_input_cell{snr,speaker} = 1.* mixture_speaker{snr,speaker}(:,1:target_length);
                training_input_cell{snr,speaker} = 1.* mixture_speaker{snr,speaker}(:,target_length+1:end);
            end
        end
        clear mixture_speaker

        % --- Cells to matrix
        num_vali = 0;
        num_train = 0;
        for snr = 1:num_snr_mix
            for speaker = 1:speaker_num
                num_vali = num_vali+size(vali_input_cell{snr,speaker},2);
                num_train = num_train+size(training_input_cell{snr,speaker},2);
            end
        end
        training_input_raw = zeros(freq_coeff_leng,num_train);
        validation_input_raw = zeros(freq_coeff_leng,num_vali);

        num_vali = 0;
        num_train = 0;
        for snr=1:num_snr_mix
            for speaker = 1:speaker_num
                num_train = num_train+size(training_input_cell{snr,speaker},2);
                training_input_raw(:,num_train-size(training_input_cell{snr,speaker},2)+1:num_train) = ...
                    training_input_cell{snr,speaker};

                num_vali = num_vali+size(vali_input_cell{snr,speaker},2);
                validation_input_raw(:,num_vali-size(vali_input_cell{snr,speaker},2)+1:num_vali) = ...
                    vali_input_cell{snr,speaker};
            end
        end

        clear vali_input_cell training_input_cell
        training_input = training_input_raw';
        validation_input = validation_input_raw';
        clear validation_input_raw training_input_raw

        % --- save auxiliary input
        save(train_data_unnorm_input_dir,'training_input');
        save(valid_data_unnorm_input_dir,'validation_input');

        % --- Normalization of the input data
        mean_training = mean(training_input,1);
        std_training = std(training_input,1,1);
        for j=1:freq_coeff_leng
            training_input(:,j)=training_input(:,j)-mean_training(:,j);
        end
        for j=1:freq_coeff_leng
            training_input(:,j)=training_input(:,j)./std_training(:,j);
        end

        for j=1:freq_coeff_leng
            validation_input(:,j)=validation_input(:,j)-mean_training(:,j);
        end
        for j=1:freq_coeff_leng
            validation_input(:,j)=validation_input(:,j)./std_training(:,j);
        end

        % --- save main input
        save(train_data_input_dir,'training_input');
        save(valid_data_input_dir,'validation_input');
        save(train_data_mean_dir,'mean_training');
        save(train_data_std_dir,'std_training');

    elseif strcmp(data_type_str, 'target_data')

        % --- For target (clean data) and weighting filter coeff. 
        % --- Arrange to cells per SNR
        load(train_clean_dir); % from part 1
        load(train_wgh_filter_dir); % from part 2
        target_length = speaker_num*num_file*duration_per_file*Fs/fram_shift; %number of frames per SNR level
        SNR_mask = cell(1,1);
        SNR_mask_h = cell(1,1);
        for snr = 1:num_snr_mix
            if snr==6
                SNR_mask{snr} = speech_fft_abs_clean(:,1+(snr-1)*target_length:end);
                SNR_mask_h{snr} = h_fft_abs_half_mat(:,1+(snr-1)*target_length:end);
            else 
                SNR_mask{snr} = speech_fft_abs_clean(:,1+(snr-1)*target_length:snr*target_length);
                SNR_mask_h{snr} = h_fft_abs_half_mat(:,1+(snr-1)*target_length:snr*target_length);
            end
        end
        clear speech_fft_abs_clean h_fft_abs_half_mat

        % --- Arrange to cells per speaker per SNR
        target_length = num_file*duration_per_file*Fs/fram_shift; % number of samples per speaker per SNR
        mask_speaker = cell(1,1);
        mask_speaker_h = cell(1,1);
        for snr = 1:num_snr_mix
            for speaker = 1:speaker_num
                if speaker==16
                    mask_speaker{snr,speaker}=SNR_mask{snr}(:,1+(speaker-1)*target_length:end);
                    mask_speaker_h{snr,speaker}=SNR_mask_h{snr}(:,1+(speaker-1)*target_length:end);
                else
                    mask_speaker{snr,speaker}=SNR_mask{snr}(:,1+(speaker-1)*target_length:speaker*target_length);
                    mask_speaker_h{snr,speaker}=SNR_mask_h{snr}(:,1+(speaker-1)*target_length:speaker*target_length);
                end
            end
        end
        clear SNR_mask SNR_mask_h

        % --- Arrange to cells for training/validation per speaker per SNR
        vali_target_cell = cell(1,1);
        training_target_cell = cell(1,1);
        vali_target_cell_h = cell(1,1);
        training_target_cell_h = cell(1,1);
        target_length = floor(valid_rate*num_file*duration_per_file*Fs/fram_shift); % number of files for validation per speaker per SNR: 1/5
        for snr=1:num_snr_mix
            for speaker=1:speaker_num
                vali_target_cell{snr,speaker}=1.* mask_speaker{snr,speaker}(:,1:target_length);
                training_target_cell{snr,speaker}=1.* mask_speaker{snr,speaker}(:,target_length+1:end);

                vali_target_cell_h{snr,speaker}=1.* mask_speaker_h{snr,speaker}(:,1:target_length);
                training_target_cell_h{snr,speaker}=1.* mask_speaker_h{snr,speaker}(:,target_length+1:end);
            end
        end
        clear mask_speaker mask_speaker_h

        % --- Cells to matrix
        num_vali=0;
        num_train=0;
        for snr=1:num_snr_mix
            for speaker=1:16
                num_vali=num_vali+size(vali_target_cell{snr,speaker},2);
                num_train=num_train+size(training_target_cell{snr,speaker},2);
            end
        end
        training_target_raw=zeros(freq_coeff_leng,num_train);
        validation_target_raw=zeros(freq_coeff_leng,num_vali);

        num_vali = 0;
        num_train = 0;
        for snr = 1:num_snr_mix
            for speaker = 1:16
                num_train = num_train+size(training_target_cell{snr,speaker},2);
                training_target_raw(:,num_train-size(training_target_cell{snr,speaker},2)+1:num_train) = ...
                    training_target_cell{snr,speaker};

                training_target_raw_h(:,num_train-size(training_target_cell_h{snr,speaker},2)+1:num_train) = ...
                    training_target_cell_h{snr,speaker};

                num_vali = num_vali+size(vali_target_cell{snr,speaker},2);
                validation_target_raw(:,num_vali-size(vali_target_cell{snr,speaker},2)+1:num_vali) = ...
                    vali_target_cell{snr,speaker};

                validation_target_raw_h(:,num_vali-size(vali_target_cell_h{snr,speaker},2)+1:num_vali) = ...
                    vali_target_cell_h{snr,speaker};
            end
        end

        clear vali_target_cell vali_input_cell training_target_cell training_input_cell vali_target_cell_h training_target_cell_h
        training_target = training_target_raw';
        validation_target = validation_target_raw';
        h_filt_input = training_target_raw_h'; % weighting filter coeff.
        h_filt_vali_input = validation_target_raw_h'; % weighting filter coeff.
        clear validation_target_raw training_target_raw

        % --- Save all data
        save(train_data_target_dir,'training_target');
        save(valid_data_target_dir,'validation_target');

        save(train_data_wgh_filter_dir,'h_filt_input');
        save(valid_data_wgh_filter_dir,'h_filt_vali_input');

    end
end

