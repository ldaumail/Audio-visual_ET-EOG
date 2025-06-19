function [pinnaReflections] = pinna(theta, phi, Fs, input)
 
A = [1, 5, 5, 5, 5];
B = [2, 4, 7, 11, 13];
D = [1, 0.5, 0.5, 0.5, 0.5];
gain = [0.5, -1, 0.5, -0.25, 0.25];
pinnaDelays = zeros(1,5);
 
theta = theta*pi/180;
phi = phi*pi/180;
 
for (i=1:5)
    pinnaDelays(i) = A(i)*cos(theta/2)*sin(D(i)*(pi/2-phi)) + B(i);
    if (Fs ~= 44100)
        pinnaDelays(i) = Fs/1000*pinnaDelays(i);
    end
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pad the input so that the delayed   %
% samples are not cut-off at the end. %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp = [input' zeros(1, ceil(max(pinnaDelays))+2)]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copy the non-delayed samples to the output so %
% that they are not cut-off at the beginning.   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pinnaReflections = temp;
clear input;
 
for (i=1:5)
    M = floor(pinnaDelays(i));
    frac = pinnaDelays(i) - M;
    for(j=M+2:length(temp)-1);
        linIntpl = temp(j-M-1)*(1-frac) + temp(j-M)*frac;
        pinnaReflections(j) = pinnaReflections(j) + gain(i)*linIntpl;
    end
end
 

