% Code to replicate the results of the Aiyagari (1994) paper
% jedmsarmstrong@nyu.edu | https://github.com/jedmsarmstrong
% March 2019

%#ok<*NOPTS>

tic

%% 0. Housekeeping
cd('/home/jethro/Documents/Uni/Computational/Aiyagari_1994/')

clear
close all
clc

addpath(genpath('/home/jethro/Documents/MATLAB/compecon2011_64_20110718'));  % For ddsimul

%% 1. Generate the transition probability matrix and grids
% Production and preferences parameters
alp         = 0.36;
del         = 0.08;
beta        = 0.96;
gamma       = 2;

% TPM parameters
n_state     = 7;
rho         = 0.9;
sigma_eps   = 0.1;
[e_grid, P] = rouwenhorst(rho,sigma_eps,n_state);

% Find the stationary distribution and implied N
[~,e,V]     = eig(P);
stat_dist   = V(:,abs(diag(e)-1)<1e-6)./sum(V(:,abs(diag(e)-1)<1e-6));
N           = exp(e_grid)*stat_dist;

% Generate the assets grid
n_grid      = 250;
max_assets  = 50;    % Chosen by trial and error
debt_limit  = 0;
a_grid      = linspace(debt_limit,max_assets,n_grid)';

% Vectorize
a_vals      = repmat(a_grid,1,n_state);
e_vals      = repmat(e_grid,n_grid,1);

%% 2. Define some functions for use in the  
% We use the inverse marginal utility to find a consumption policy
u_prime          = @(c) c.^(-gamma);
u_prime_inverse  = @(c) c.^(-1/gamma);
c_update         = @(c_guess,r) u_prime_inverse(beta * (1+r) * u_prime(c_guess) * P');

% Update assets according to the budget constraint
A_update         = @(a_prime,r,c,e_vals,w) 1/(1+r) * (c + a_prime - w * exp(e_vals));

%% 3. Solve the problem using the equilibrium function below
myfun       = @(r) equilibrium(r,a_vals,e_vals,c_update,A_update,P,alp,del,N);
options     = optimset('display','iter','TolX',1e-9);

r0          = [0.0370 0.03701];     % Chosen by trial and error
rstar       = fzero(myfun,r0,options);

%% 4. Extract the asset distribution and plot
[~,as] = equilibrium(rstar,a_vals,e_vals,c_update,A_update,P,alp,del,N);

% We trim the first 1500 periods as burn in
% Plot a distribution of assets
figure();
[counts,edges] = histcounts(as(:,1501:end),n_grid);
bar(edges(1:end-1)+diff(edges)/2,counts/sum(counts));

% Plot a Lorenz Curve
end_assets = as(:,1501:end);
end_assets = sort(end_assets(:));
total_assets = sum(end_assets);
asset_prctl = cumsum(end_assets)/total_assets;
people = (1:length(end_assets))/length(end_assets);
figure();
ax = axes();
plot(people,asset_prctl);
ax.XLim = [0 1];
ax.YLim = [0 1];
rl = refline(1,0);
rl.LineStyle = '--';
axis square


%% Function definition for finding the equilibrium
function [difference,as,r_new,K] = equilibrium(r,a_vals,e_vals,c_update,A_update,P,alp,del,N)
%EQUILIBRIUM Given an interest rate r (and other parameters), find EQM

% Some varues for later
n_state = size(a_vals,2);
e_grid = e_vals(1,:);
% Calculate the equilibrium wage
wage = (1-alp) * (alp / (r+del)) ^ (alp/(1-alp));
% Initial consumption -- consume asset wealth plus labor income
c_0 = r * a_vals + wage * exp(e_vals);

% This loops over the consumption update function above to find the optimal
% consumption policy
dist = 1;
disp('Finding optimal consumption policy...')
while dist > 1e-6
    c_new = c_update(c_0,r);
    A_new = A_update(a_vals,r,c_new,e_vals,wage);
    
    % Interpolate these guys along the grid
    c_new_grid = zeros(size(a_vals));
    for ii = 1 : n_state
        c_new_grid(:,ii)  = interp1(A_new(:,ii),c_new(:,ii),a_vals(:,ii),'spline');
    end
    % In cases where the debt limit is binding we can just update using the
    % budget constraint
    c_new_binding = (1+r) * a_vals + wage * exp(e_vals);

    c_final = zeros(size(a_vals));
    for ii = 1 : n_state
        c_final(:,ii) = (a_vals(:,1)>A_new(1,ii)).*c_new_grid(:,ii)+(a_vals(:,ii)<=A_new(1,ii)).*c_new_binding(:,ii);
    end
    
    dist = norm((c_final-c_0)./c_0);
    c_0 = c_final;

end

% Do all of the epsilon simulation first
disp('Monte Carlo simulating...')

NN = 10000;
TT = 2000;
as = nan(NN,TT+1);
cs = nan(NN,TT);

eps_ind = ddpsimul(P,randi(n_state,NN,1),TT-1,1);
as(:,1) = a_vals(randi(size(a_vals,1),NN,1),1);
eps = exp(e_grid(eps_ind));

for tt = 1 : TT
    cs(:,tt)   = 0;
    for ii = 1 : n_state
        cs(:,tt) = cs(:,tt) + (eps_ind(:,tt)==ii).*interp1(a_vals(:,ii),c_0(:,ii),as(:,tt),'spline');
    end
    as(:,tt+1) = (1+r)*as(:,tt)+eps(:,tt)*wage-cs(:,tt);

end

K = mean(mean(as(:,TT-500:TT)));
r_new = alp * (K/N)^(alp-1) - del;

difference = (r_new-r)/r;

end

