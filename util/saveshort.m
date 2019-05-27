function count = saveshort( A, filename )

% Writes 16 bit short data to file on the disk
%
% Usage:    count = saveshort( A, filename )
%            
%           A        : MATLAB matrix or vector with data
%           filename : Name of file data will be written to
%           count    : Number elements that have been written to file
%
% Technische Universität Braunschweig
% Institute for Communications Technology (IfN)
% Schleinitzstrasse 22
% 38106 Braunschweig
% Germany
% 2006 - 09 - 26
% (c) Prof. Dr.-Ing. Tim Fingscheidt
% 
% Use is permitted for any scientific purpose when citing the paper:
% Z. Zhao, S. Elshamy, and T. Fingscheidt, "A Perceptual Weighting Filter 
% Loss for DNN Training in Speech Enhancement", arXiv preprint arXiv: 
% 1905.09754.
%
%--------------------------------------------------------------------------

infid = fopen(filename,'wb');

if infid==-1,
   error(['SAVESHORT: File ', filename , ' could not be opened!']);
   return;
end;

count = fwrite(infid,A,'short');

if fclose(infid)~=0,
   error(['SAVESHORT: File ', filename , ' was not closed properly!']);
end;

