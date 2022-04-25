%--------------------------------------------------------------------------
% GitHubTrain_part_2_WghFilterResponse - Generating frequency amplitude  
% response for the perceptual weighting filter in CELP speech codec (e.g., 
% AMR). Given the (clean) speech signal, frame-wise amplitude response is
% computed and saved. 
% Note that the clean speech signals are from Grid corpous (downsampled to 
% 16 kHz) dataset and noise signals are from ChiMe-3 dataset. 
%
%
% Given data:
%             s_speech : the given input (clean) speech signal 
%         
% Output data:
%             h_fft_abs_half_mat : the matrix contains frame-wise weighting
%                                  filter amplitude response, with the
%                                  dimension as: (half-plus-one frequency 
%                                  bins, num. of frames)
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
gamma2 = 0.6;  % weighting filter factors: gamma1 and gamma2
gamma1 = 0.92; % possible gamma1, e.g., 0.9, 0.92, 0.94, 0.96, etc.
Np = 16;
fram_leng = 256;
fram_shift = fram_leng/2; % frame shift
Lplus = 0;
freq_coeff_leng = fram_shift + 1; % half-plus-one frequency coefficients

% --- Define directories  
train_sspeech_dir = '.\train\speech_clean_s_speech.mat'; 
train_wgh_filter_dir = '.\train\h_fft_abs_half_mat_AMR_direct_freqz_6snrs.mat';

% --- Load clean speech from part 1 
load(train_sspeech_dir);

% --- Get LPC coefficients
wd = hanning(fram_leng,'periodic');
[a_lpc, ~] = lpc_analysis_for_weight_filt( s_speech, fram_shift, Lplus, wd, Np );

% --- frame-wise forming weighting filter using LPC coefficients
h_fft_abs_half_mat = zeros(freq_coeff_leng,size(a_lpc,1));
for k = 1 : size(a_lpc,1)
    % -- Get filter z-domain coefficients
    a_temp = a_lpc(k, 2:Np+1); % Np coefficients (exclude the "a(0)=1")
    a_gamma1 = zeros(1,Np+1); % initialization of the numerator coefficients for the weighting filter 
    a_gamma1(1,1) = 1; % see (2) in the paper
    a_gamma2 = zeros(1,Np+1); % initialization of the denominator coefficients for the weighting filter
    a_gamma2(1,1) = 1; % see (2) in the paper
    for k1 = 1 : Np, % compute numerator and denominator coefficients
        a_gamma1(k1 + 1) = a_temp(k1) * gamma1 ^ k1; 
        a_gamma2(k1 + 1) = a_temp(k1) * gamma2 ^ k1; 
    end

    % -- Get frequency response 
    % -- Directly generate frequency response from z-domain coefficients
    [h_iir_temp,~] = freqz(a_gamma1, a_gamma2, 'whole', fram_leng);
    h_fft_abs_half = abs(h_iir_temp(1:freq_coeff_leng));

    % -- Document the frequency response to the matrix
    h_fft_abs_half_mat(:,k) = h_fft_abs_half;

    % -- Display progress
    if mod(k,50000) == 0,
        disp(['Percentage of frames finished: ' num2str( (k/size(a_lpc,1))* 100) '%']);
    end

end

% --- Save weighting filter coeff.
save(train_wgh_filter_dir,'h_fft_abs_half_mat')
