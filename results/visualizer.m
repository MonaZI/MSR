% clear all
% close all

figure(1)
%% uniform p
sigma_vec = 10.^[-0.3:0.05:log10(2)];


% rewrite the 0 entries to be infinite

%% plotting the results
width = 4;     % Width in inches
height = 4;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 2;      % LineWidth
msz = 8;       % MarkerSize

% The properties we've been using in the figures
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);


ind = 9;
figure
subplot(2,1,1)
loglog(sigma_vec(1:ind),mse_EM(1:ind),'k-*','LineWidth',2,'MarkerSize',4);hold on
loglog(sigma_vec(1:ind),min(mse_cov(1:ind,:),[],2),'kx-','LineWidth',1)
legend({'EM','our approach'},'Location','southeast','FontSize',9)
xlabel({'\sigma'},'FontSize',10)
ylabel({'MSE'},'FontSize',10)
grid on


subplot(2,1,2)
loglog(sigma_vec(1:ind),time_EM(1:ind),'k-*','LineWidth',2,'MarkerSize',4);hold on
loglog(sigma_vec(1:ind),mean(time_cov(1:ind,:),2),'kx-','LineWidth',1)
legend({'EM','our approach'},'Location','southeast','FontSize',9)
xlabel({'\sigma'},'FontSize',10)
ylabel({'Computation time (s)'},'FontSize',10)
grid on

