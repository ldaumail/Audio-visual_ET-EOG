function [cleanDat, blink]= removeBlinks(xPos,yPos, pupilSize) %, wind)
%trigdata = groupDat; = data triggered to stimulus onset
%wind = time window to remove around the blink (former version of blink
%removal)
%developped by Loic Daumail on 10/22/2021
%removes blinks from electroooculograms. replaces portions of blinks by
%NaNs 
% 
% xPos = x;
% yPos = y;
% pupilSize = pupSize;

cleanDat = nan(2,length(xPos));

xs = xPos;
ys = yPos;
% figure()
% subplot(3,1,1)
% plot(xs,ys);

%1) find wherever the pupil size is 0
blinkX = ismember(pupilSize, 0); %see where blinks occur (pupil size ==0)
lowBound = find(blinkX)- 150; %blink onset time may be 150ms before pupil size = 0
validLB = lowBound > 0;
highBound = find(blinkX) + 150; %blink offset time may be 150ms after pupil size = 0
validHB = highBound < length(pupilSize);
lowBound = lowBound(validLB & validHB);
highBound = highBound(validLB & validHB);

% figure();
% h = histogram(dur, 'NumBins', 50, 'Normalization','probability', 'EdgeColor', 'b', 'FaceColor', 'b');
% hold on
% xline(mean(dur), 'Color', 'r', 'LineWidth', 2);
% hold on
% xline(median(dur), 'Color', 'k', 'LineWidth', 2);

%the above step will make some bounds pretty redundant, but will ensure
%removal of all blink times.


%3) eliminate all blinks from all data

% xs(same) = NaN;
% subplot(3,1,2)
% plot(xs,ys)
for b =1:length(lowBound)
    xs(lowBound(b):highBound(b)) = NaN;
end
% subplot(3,1,3)
% plot(xs,ys)
ys(isnan(xs)) = NaN;
blink = isnan(xs);
%replace data
cleanDat(1,:) = xs;
cleanDat(2,:) = ys;

% xcenter =0;
% ycenter =0;
% [scrxs, scrys] = drawSquare(halfscrwidth, scrheight, xcenter,ycenter);
% figure(); plot(x1,y1); hold on; plot(scrxs, scrys);







