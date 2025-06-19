function [xx, yy] = MakeTrajectory(p)

% p.startxy: [x y] in cartesian coordinates (where the stimulus should begin)
% p.endxy: [x y] in cartesian coordinates (where the stimulus should stop)
 
% z (elevation) is 0 for now 

% by Woon Ju Park January 2019

nsamples = p.Fs * p.noiseDur; 

if ~p.orbital 
    
    xx = linspace(p.startxy(1), p.endxy(1), nsamples)';
    yy = linspace(p.startxy(2), p.endxy(2), nsamples)';
    
elseif p.orbital
    
    [startTH, startPHI, startR] = cart2sph(p.startxy(1), p.startxy(2), 0);
    [endTH, endPHI, endR] = cart2sph(p.endxy(1), p.endxy(2), 0);
    
    TH = linspace(startTH, endTH, nsamples)'; 
    PHI = linspace(startPHI, endPHI, nsamples)';
    R = linspace(startR, endR, nsamples)';
    [xx, yy, z] = sph2cart(TH, PHI, R);
    
end


end