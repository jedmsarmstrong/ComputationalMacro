cd('/home/jethro/Documents/Uni/Computational/Arellano_2008')

clear
clc
close all

%% Parameterization (pp 527)
sigma   = 2;
beta    = 0.85;
rs      = 0.01;
theta   = 0.0385;
a0      = 0;
a1      = -0.35;
a2      = 0.4403;
rho     = 0.9317;
sigma_e = 0.037;

n_grid = 10;
grid_y = linspace(0.6523,1.5330,n_grid);
grid_d = linspace(0,1.5,n_grid);

%% Discretize the transition probability matrix

n_steps = 1e7;

model = arima('Constant',0,'AR',rho,'Variance',sigma_e^2);
y = simulate(model,n_steps)+1.0926;

bins = discretize(y,[0 grid_y(2:end) 100],'includededge','left');
transmat_y = zeros(n_grid);

for tt = 1 : n_steps-1
    transmat_y(bins(tt),bins(tt+1)) = ...
        transmat_y(bins(tt),bins(tt+1)) + 1;  % increment it
end
transmat_y = transmat_y./(repmat(sum(transmat_y,2),1,n_grid));
heatmap(transmat_y)

%%
[zgrid, P] = rouwenhorst(rho, sigma_e, n_grid);
zgrid = zgrid + 1.0926;

%%
L = @(y) max(0, a1*y + a2 * y.^2);
h = @(y) y - L(y);
u = @(c) (c .^ (1-sigma) - 1 ) / (1-sigma);


% Let's do cols = B, rows = Y
Q = 0.1 * ones(ngrid); %% TODO: Fix this for consistency
Vc = zeros(

