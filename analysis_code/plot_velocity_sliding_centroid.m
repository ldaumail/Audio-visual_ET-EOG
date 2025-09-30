
clear all;
close all;

subjects; 

whichgroup      = 2; % 1: ASD / 2: TD
whichsubject    = 10; % 10 (TD) is a good example

makeplots       = 1;
slideornot      = 1;

for whichblock = 4
    
    if whichgroup == 1
        subid = ASD{whichsubject};
        loadname = [subid '_' num2str(whichblock) '.mat'];
    elseif whichgroup == 2
        subid = TD{whichsubject};
        loadname = [subid '_' num2str(whichblock) '.mat'];
    end
    
    load(loadname);
    
    p = 1;     
    behavior = [];
    
    for whichtrial = 8
        for whicheye = 1:2
            eye_analyses_centroid;
        end
        p = p + 1;
    end
        
end


T_left
% T_right



