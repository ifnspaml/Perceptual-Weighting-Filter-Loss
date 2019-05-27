% -------------------------------------------------------------------------
% actlev mex file for 64 bit systems, direct passthrough of audio samples
% Compiled from G.191 (03/10)
% https://www.itu.int/rec/T-REC-G.191-201003-I/en
% 
% Usage:    [act_lev, rms_lev, gain] = actlev(params, s)
%
% Input:
%				params	:	string with parameters '-sf 8000 -lev -26 -rms', 
%							'-sf 8000 -lev -26', '-sf 8000'
%							not supported: -start -end -n -bits (assumes 16bit)
%				s		:	reference array expected in 16bit, use int16(s) to 
% 							convert and simulate saveshort
%
% Output:
%				act_lev	:	Active speech level [dBov]
%				rms_lev	:	RMS level [dBov]
%				gain	:	Gain factor to normalize to given dBov by -lev
%							parameter
%
% Technische Universität Braunschweig
% Institute for Communications Technology (IfN)
% Schleinitzstrasse 22
% 38106 Braunschweig
% Germany
% 2015 - 11 - 12
% Samy Elshamy, M.Sc.
%
% This file is part of the SV Matlab Toolbox!
% 
% Use is permitted for any scientific purpose when citing the paper:
% Z. Zhao, S. Elshamy, and T. Fingscheidt, "A Perceptual Weighting Filter 
% Loss for DNN Training in Speech Enhancement", arXiv preprint arXiv: 
% 1905.09754.
%
%--------------------------------------------------------------------------
