function [a, r] = lpc_analysis_for_weight_filt( speech, L, Lplus, window, Np )
% LPC_ANALYSIS_FOR_WEIGHT_FILT - Comfortable LPC analysis with arbitrary
% window functions.
%
% Usage:  [a, r] = lpc_analysis_for_weight_filt( s, L, Lplus, window, Np );
%
%         An LPC analysis of the input s is performed.
%
% Input:
%         s      : input speech
%         L      : frame shift length in samples
%         Lplus  : offset of window function in samples (+/- allowed)
%         window : array containing window function
%         Np     : LPC filter order
%
%                       frame shift (length L)
%         ___________|__________________________|___________ speech frames
%                                                <- Lplus ->
%                 |_________________________________________|  window array
%                               length (window)    
%
% Output:
%         a      : array containing the LPC coefficients for each frame
%                  a(1:num_of_frames,1:Np+1) with a(:,1) = ones(num_of_frames,1)
%         r      : array containing the (PARCOR) reflection coefficients  
%                : for each frame:  r(1:num_of_frames,1:Np)
%
%         At the beginning of the analysis, the end of the window array
%         is fixed Lplus samples above the end of the first frame of
%         the input speech signal. 
%         The analysis procedure is aborted, once the window array
%         does not fit any more completely to the speech signal.
%
%
% Technische Universität Braunschweig
% Institute for Communications Technology (IfN)
% Schleinitzstrasse 22
% 38106 Braunschweig
% Germany
% 2006 - 09 - 08
% (c) Prof. Dr.-Ing. Tim Fingscheidt
%
% Modified by Ziyue Zhao, Technische Universität Braunschweig, IfN 
% Only reserve the functionality of getting LP coefficients.
% 2019 - 05 - 22 
% 
% Use is permitted for any scientific purpose when citing the paper:
% Z. Zhao, S. Elshamy, and T. Fingscheidt, "A Perceptual Weighting Filter 
% Loss for DNN Training in Speech Enhancement", arXiv preprint arXiv: 
% 1905.09754.
%
%--------------------------------------------------------------------------

%=== Initializations ======================================================
 speech = speech(:);
 window = window(:);
 Nw     = length(window);

%--- Compute LPC filter coefficients --------------------------------------
%--- Prepare speech file for analysis purposes 
 disp('LPC Analysis: Start... (please wait)');
 num_frames = floor( (length(speech) - Lplus) / L ) - 1; % number of frames 
 speech_ana = [ speech(1:(num_frames+1)*L + Lplus)' ]';  % speech signal used for analysis (no zero-padding before speech)

%--- Allocate memories ----------------------------------------------------
 
%--- Shorten speech file for filter purposes
 autocorr_coeff = zeros(Np+1,num_frames);
 a              = zeros(num_frames,Np+1);
 r              = zeros(num_frames,Np);
 
%=== Compute LPC coefficients frame-wise ==================================
%--- Compute autocorrelation coefficients
 for i = 1:num_frames,
     s = speech_ana(1+L*(i-1) : Nw+L*(i-1),1).*window; % current speech frame and windowing
     % autocorr_coeff(:,i) = acf(s, Np+1);
     tmp                 = xcorr(s, s, Np);
     autocorr_coeff(:,i) = tmp(Np+1:2*Np+1);
     if(mod(i,500000)==0)
         disp(sprintf('LPC Analysis: %d/%d frames of autocorrelations computed...',i,num_frames));
     end;
 end; %for         
 disp(sprintf('LPC Analysis: %d/%d frames of autocorrelations computed...',num_frames,num_frames));

%--- Compute LPC filter coefficients
 for i = 1:num_frames,
     if sum(autocorr_coeff(:,i)) == 0,
       a(i,:) = [ 1 zeros(1,Np) ];
       r(i,:) = [ zeros(1,Np) ];
     else
       [a(i,:),~,r(i,:)] = levinson( autocorr_coeff(:,i), Np );
     end;
     if(mod(i,500)==0)
         disp(sprintf('LPC Analysis: %d/%d LPC filter coefficient sets computed...',i,num_frames));
     end;
 end; %for         
 disp(sprintf('LPC Analysis: %d/%d LPC filter coefficient sets computed...',num_frames,num_frames));

 % --- End of getting LPC coefficients: a
 
 disp('LPC Analysis: Done.');
