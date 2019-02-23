% Code to replicate the results of the Uribe Schmitt-Grohe version of the
% Arellano (2008) paper
% jedmsarmstrong@nyu.edu | https://github.com/jedmsarmstrong
% Feb 2019

%#ok<*NOPTS>


cd('/home/jethro/Documents/Uni/Computational/Arellano_2008')

clear
clc
close all

%% Parameterization (pp 527)
% Grid and probabilitity transition matrix parameters
n_grid  = 100;
rho     = 0.9317;
sigma_e = 0.037;

% Construct the grid for searching over
grid_y  = exp(linspace(log(0.6523),log(1.5330),n_grid));
grid_d  = linspace(0,1.5,n_grid);

% Model parameters
sigma   = 2;
beta    = 0.85;
rs      = 0.01;
theta   = 0.0385;
a0      = 0;
a1      = -0.35;
a2      = 0.4403;

%
%% Discretize the transition probability matrix
% This takes a long time -- run only if needed!
%{
% Simulate the data using the AR(1) process
n_steps = 1e8;
model   = arima('Constant',0,'AR',rho,'Variance',sigma_e^2);
y       = simulate(model,n_steps);

% Discretize the y values into their index on the (log) grid
bins    = discretize(y,[-10 log(grid_y(2:end)) 5],'includededge','left');

% INitialize and update the grid
P = zeros(n_grid);
for tt = 1 : n_steps-1
    P(bins(tt),bins(tt+1)) = ...
        P(bins(tt),bins(tt+1)) + 1;  % increment it
end

% Turn into proper probabilities (rows sum to 1)
P = P./(repmat(sum(P,2),1,n_grid));
save P P
%}
% Also P100 for n_grid = 100 and P50 for n_grid = 50
load P

%% Define functions used in the model
% Loss function
L = @(y) max(0, a1*y + a2 * y.^2);
% Income in autarky
h = @(y) y - L(y);
% Utility function (CRRA)
u = @(c) (c .^ (1-sigma) - 1 ) / (1-sigma);

%% Turn all grids into 3-D
% Dim1 = ys       (YY)
% Dim2 = Dprime   (DP)
% Dim3 = D        (DD)
YY = repmat(grid_y',1,n_grid,n_grid);
DP = repmat(grid_d,n_grid,1,n_grid);
DD = repmat( reshape(grid_d,1,1,n_grid), n_grid,n_grid,1);

%% Initialize our value functions
Vd = zeros(n_grid,1);               % Value of defaulting
Vc = zeros(n_grid,n_grid);          % Value of continuing
V0 = max(Vc,repmat(Vd,1,n_grid));   % Optimal decision
Q  = 1/(1+rs) * ones(n_grid,n_grid);% Asset prices
debt_pol = zeros(n_grid,n_grid);    % Optimal debt police (Dprime given D)

% Utility in autarky is always the same
U_a = u(h(grid_y'));    

%% VFI til convergence
conv = false;

tic
while ~conv
    
    % Calculate consumption -- dimensions as above
    CC = DP.* repmat(Q,1,1,n_grid) - DD + YY;
    % Find utility, and remove negative consumption
    UU = u(CC);
    UU(CC < 0) = -inf;
    
    % Vc is the max over Dprime (Dim2). We also save the index of the
    % optimal choice of Dprime for the debt policy
    [Vcnew,debt_pol_new] = max(UU + beta*repmat(P * V0,1,1,n_grid),[],2);
    Vcnew = squeeze(Vcnew);
    debt_pol_new = squeeze(debt_pol_new);
    
    % Vd is easy
    Vdnew = repmat(U_a + beta * (P * (theta * V0(:,1) + (1-theta) * Vd(:,1))) ,1,n_grid);
    
    % Check for default and update prices
    Qnew = (1- P*(Vc < Vd))/(1+rs);
    
    % Check convergence
    conv = logical(norm(Q(:)-Qnew(:),inf) + max(abs(Vc(:)-Vcnew(:))) +...
        norm(Vd(:,1)-Vdnew(:,1),inf) + norm(debt_pol(:)-debt_pol_new(:),inf)  < 1e-8);
    
    % Update all matrices
    Vc = Vcnew;
    Vd = Vdnew;
    V0 = max(Vc,Vd);
    Q = Qnew;
    debt_pol = debt_pol_new;
    
end
toc

% Default sets (again, rows are Y, cols are Dprime)
DD = Vc < Vd;

%% SIMULATE
% Set periods for burn in and actual
n_burn = 1e5;
n_act = 1e6;
n_per = n_burn+n_act;

% Initialize arrays
ds      = nan(1,n_per);      % Debt
ys      = nan(1,n_per);      % Income without any loss
ya      = nan(1,n_per);      % Actual income including loss
qs      = nan(1,n_per);      % Debt price
default = nan(1,n_per); % Indicator for if economy starts in default
def2    = nan(1,n_per);    % Indicator for if economy goes into default

% Initialize values
yix = randi(n_grid);
dix = randi(n_grid);
def = 0;

% We need to generate the price of debt in autarky
CC_a    = repmat(h(grid_y'),1,n_grid);
mu_a    = CC_a .^ (-sigma);
CC_good = Q(sub2ind([n_grid n_grid],repmat((1:n_grid)',1,n_grid),debt_pol)) .* grid_d(debt_pol) + ...
            YY(:,:,1) - DP(:,:,1);
CC_g    = DD.*CC_a + (1-DD).*CC_good;
mu_g    = repmat(CC_g(:,1) .^ (-sigma),1,n_grid);
QQ_a    = (beta*theta*P*mu_g + beta*(1-theta)*P*mu_a) ./ mu_a;

% Simulate the data
tic
for ii = 1 : n_per
    
    ys(ii)   = grid_y(yix);
    ds(ii)   = grid_d(dix);
    back_in  = rand();          % Value for if economy is let back in
    def2(ii) = Vc(yix,dix) < Vd(yix);
    
    if ~def && (Vc(yix,dix) >= Vd(yix))   % No default
        
        def = 0;
        default(ii) = 0;
        ya(ii) = ys(ii);
        qs(ii) = Q(yix,debt_pol(yix,dix));
        dix = debt_pol(yix,dix);
        
    elseif ~def && (Vc(yix,dix) < Vd(yix)) % Defaulting now
        
        def = 1;
        default(ii) = 0;
        ya(ii) = h(ys(ii));
        qs(ii) = QQ_a(yix,dix);
        dix = 1;
        
    elseif def && (back_in <= theta) % Back in the game!
        
        def = 0;
        default(ii) = 0;
        ya(ii) = ys(ii);
        qs(ii) = Q(yix,debt_pol(yix,1));
        dix = debt_pol(yix,1);
        
    elseif def && (back_in > theta) % Sucks to suck!
        
        def = 1;
        default(ii) = 1;
        ya(ii) = h(ys(ii));
        qs(ii) = QQ_a(yix,dix);
        dix = 1;

    end
    % Update the y index
    yix = find(rand() < cumsum(P(yix,:)),1);
    
end
toc

% Trim burn in
ya      = ya(n_burn+1:end);
ds      = ds(n_burn+1:end);
qs      = qs(n_burn+1:end);
ys      = ys(n_burn+1:end);
default = default(n_burn+1:end);
def2    = def2(n_burn+1:end);

%% STATS

good_inds = ~default & ~def2;
mean(ds(good_inds)./ya(good_inds))*100 


qs_ann = ((1./qs).^4-1)*100;
sprd = qs_ann-((1+rs)^4-1)*100; %world interest rate (annualized)

mean(sprd(good_inds))
std(sprd(good_inds))
corr(sprd(good_inds)',ya(good_inds)')
mean(def2)*4*100

%{
ans =
   58.8278
ans =
    3.4826
ans =
    3.2128
ans =
   -0.5437
ans =
    2.6416
%}

%% Plots
%{
bb = bar( 1.1*default.*max(max([ys' ya'])) );
hold on
plot([ys' ya'])
bb.FaceColor = 0.8*[1 1 1];
bb.EdgeColor = 0.8*[1 1 1];


%%
bb = bar( 1.1*default.*max(max([ys' ya'])) );
hold on
plot(ds)
bb.FaceColor = 0.8*[1 1 1];
bb.EdgeColor = 0.8*[1 1 1];

%}
