function EPEBarPlotSTD(yvar,conditions, ylab,ylims, leg,speeds)
%assumes data is 3D %subj resp x contrast level x cond
% yvar = yval;
% nlines = 9;
% cmaps = struct();
% cmaps(1).map =cbrewer2('Greys', nlines);
% cmaps(2).map =cbrewer2('Greys', nlines);
% cmaps(3).map =cbrewer2('Reds', nlines);
% cmaps(4).map =cbrewer2('Blues', nlines);
% % b(1).FaceColor = cmaps(1).map(4,:);
% % b(2).FaceColor = cmaps(2).map(1,:);
% % b(3).FaceColor = cmaps(3).map(4,:);
% % b(1).EdgeColor = cmaps(4).map(3,:);
% % b(2).EdgeColor = cmaps(4).map(3,:);
% % b(3).EdgeColor = cmaps(4).map(3,:);
% % 
% % b(1).LineWidth = 1;
% % b(2).LineWidth = 1;
% % b(3).LineWidth = 1;
% 
nlines = 7;
cmaps = struct();
cmaps(1).map =cbrewer2('Blues', nlines);
cmaps(2).map =cbrewer2('Greys', nlines);
cmaps(3).map =cbrewer2('Reds', nlines);
cmaps(4).map =cbrewer2('Greens', nlines);


colIdx = [3,5,7];

% mYvar = squeeze(mean(squeeze(yvar),1,'omitnan'));
% errhigh = nan(size(mYvar));
% errlow = nan(size(mYvar));
% for i =1:size(mYvar,2) %inducer type
    %95%CI
    % errhigh(:,i) =  + std(squeeze(yvar(:,:,i)),[],1,'omitnan')/sqrt(size(yvar,1));%nanmean(squeeze(yvar(:,:,s)),2)
    % errlow(:,i) =  - std(squeeze(yvar(:,:,i)),[],1,'omitnan')/sqrt(size(yvar,1)); %nanmean(squeeze(yvar(:,:,s)),2)
% end

% x = 1:size(mYvar,2);
figure('Renderer', 'painters', 'Position', [10 10 900 900]);
x = speeds;
for e =1:size(yvar,3)
    subplot(size(yvar,3),1,e)
    plot(x,yvar(:,:,e),"LineWidth",2)

    set(gca, 'box', 'off')
    xticklab = get(gca,'XTickLabel');
    set(gca, 'XTickLabel',xticklab,'fontsize',12, 'FontWeight', 'Bold')
    xlabel('Drifting Speed (deg/s)')
    ylabel(ylab,'fontweight','bold','fontsize',12)

    ylim(ylims)
    title(strcat(leg{e},' eye'))
    legend(conditions, 'Location','bestoutside')
end


% ax = gca;
% ax.XTick = [1:length(speeds)];
% ax.XTickLabels = speeds;



end
