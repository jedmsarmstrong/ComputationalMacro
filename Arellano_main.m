cd('/home/jethro/Documents/Uni/Computational/Arellano_2008')

%%
clear
clc
close all

%%
n_grid = 100;
rho     = 0.9317;
sigma_e = 0.037;

%% Parameterization (pp 527)
grid_y = exp(linspace(log(0.6523),log(1.5330),n_grid));
grid_d = linspace(0,1.5,n_grid);

sigma   = 2;
beta    = 0.85;
rs      = 0.01;
theta   = 0.0385;
a0      = 0;
a1      = -0.35;
a2      = 0.4403; %(1-a1)/2/max(grid_y);

%
%% Discretize the transition probability matrix
%{
n_steps = 1e8;
model = arima('Constant',0,'AR',rho,'Variance',sigma_e^2);
y = simulate(model,n_steps);

bins = discretize(y,[-10 log(grid_y(2:end)) 5],'includededge','left');
transmat_y = zeros(n_grid);

for tt = 1 : n_steps-1
    transmat_y(bins(tt),bins(tt+1)) = ...
        transmat_y(bins(tt),bins(tt+1)) + 1;  % increment it
end
transmat_y = transmat_y./(repmat(sum(transmat_y,2),1,n_grid));
%heatmap(transmat_y)

%%

%[zgrid, P1] = rouwenhorst(rho, sigma_e, n_grid);
%zgrid = zgrid*sigma_e + 1.0926;
P = transmat_y;
save P P
%}
load P
%

%%
L = @(y) max(0, a1*y + a2 * y.^2);
h = @(y) y - L(y);
u = @(c) (c .^ (1-sigma) - 1 ) / (1-sigma);

%%
% rows have Y, cols have Dp, slices have D
YY  = repmat(grid_y',1,n_grid,n_grid);
DP  = repmat(grid_d,n_grid,1,n_grid);
DD  = repmat( reshape(grid_d,1,1,n_grid), n_grid,n_grid,1);

%%
Vd = zeros(n_grid,1);
Vc = zeros(n_grid,n_grid);
V0 = max(Vc,repmat(Vd,1,n_grid));
Q  = 1/(1+rs) * ones(n_grid,n_grid);
conv = false;
its = 0;
debt_pol = zeros(n_grid,n_grid);
U_a = u(h(grid_y'));

%% Fast version
tic
while ~conv
    
    
    CC = DP.* repmat(Q,1,1,n_grid) - DD + YY;
    UU = u(CC);
    UU(CC < 0) = -inf;
    

    [MM,debt_pol_new] = max(UU + beta*repmat(P * V0,1,1,n_grid),[],2);
    Vcnew = squeeze(MM);
    debt_pol_new = squeeze(debt_pol_new);
    
    Vdnew = repmat(U_a + beta * (P * (theta * V0(:,1) + (1-theta) * Vd(:,1))) ,1,n_grid);
    
    % Check for default and update prices
    Qnew = (1- P*(Vc < Vd))/(1+rs);
    
    % Check convergence
    conv = logical(norm(Q(:)-Qnew(:),inf) + max(abs(Vc(:)-Vcnew(:))) + norm(Vd(:,1)-Vdnew(:,1),inf) + norm(debt_pol(:)-debt_pol_new(:),inf)  < 1e-8);
    
    Q = Qnew;
    Vc = Vcnew;
    Vd = Vdnew;
    V0 = max(Vc,Vd);
    debt_pol = debt_pol_new;

    its = its + 1;
    
end
toc

%heatmap(Q)

DD = Vc < Vd;
Qt = Q;

%% SIMULATE
n_burn = 1e5;
n_act = 1e6;
n_per = n_burn+n_act;
ds = nan(1,n_per);
ys = nan(1,n_per);
ya = nan(1,n_per);
qs = nan(1,n_per);
default = nan(1,n_per);
def2 = nan(1,n_per);

yix = randi(n_grid);
dix = randi(n_grid);
def = 0;

CC_a = repmat(h(grid_y'),1,n_grid);
mu_a = CC_a .^ (-sigma);

CC_good = Qt(sub2ind([n_grid n_grid],repmat((1:n_grid)',1,n_grid),debt_pol)) .* grid_d(debt_pol) + ...
    YY(:,:,1) - DP(:,:,1);

CC_g = DD.*CC_a + (1-DD).*CC_good;
mu_g = repmat(CC_g(:,1) .^ (-sigma),1,n_grid);

QQ_a = (beta*theta*P*mu_g + beta*(1-theta)*P*mu_a) ./ mu_a;
tic
for ii = 1 : n_per
    
    ys(ii) = grid_y(yix);
    ds(ii) = grid_d(dix);
    back_in = rand();
    def2(ii) = Vc(yix,dix) < Vd(yix);
    
    if ~def && (Vc(yix,dix) >= Vd(yix))   % No default
        
        def = 0;
        default(ii) = 0;
        ya(ii) = ys(ii);
        qs(ii) = Qt(yix,debt_pol(yix,dix));
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
        qs(ii) = Qt(yix,debt_pol(yix,1));
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

ya = ya(n_burn+1:end);
ds = ds(n_burn+1:end);
qs = qs(n_burn+1:end);
ys = ys(n_burn+1:end);
default = default(n_burn+1:end);
def2 = def2(n_burn+1:end);

%%
%{
%plot(bs)
%%

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
   58.9130
ans =
    3.4748
ans =
    3.2130
ans =
   -0.5430
ans =
    2.7020

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



%%
%{
%% Full version
tic
while ~conv
    
    QQ = repmat(Q,1,1,n_grid);
    
    CC = DP.* QQ - DD + YY;
    UU = u(CC);
    UU(CC < 0) = -inf;
    

    EVal = P * V0;
    EVal = repmat(EVal,1,1,n_grid);
    [MM,debt_pol_new] = max(UU + beta*EVal,[],2);
    Vcnew = squeeze(MM);
    debt_pol_new = squeeze(debt_pol_new);
    
    vals = theta * V0(:,1) + (1-theta) * Vd(:,1);
    EVal = P * vals;
    Vdnew = repmat(u(h(grid_y')) + beta * EVal,1,n_grid);
    
    % Check for default and update prices
    F = Vc < Vd;
    Qnew = (1- P*F)/(1+rs);
    
    % Check convergence
    conv = logical(norm(Q(:)-Qnew(:),inf) + max(abs(Vc(:)-Vcnew(:))) + norm(Vd(:,1)-Vdnew(:,1),inf) + norm(debt_pol(:)-debt_pol_new(:),inf)  < 1e-8);
    if mod(its,100) < 1
        disp(its);
        %disp(conv);
        disp(norm(Q(:)-Qnew(:),inf) + max(abs(Vc(:)-Vcnew(:))) + norm(Vd(:,1)-Vdnew(:,1),inf) + norm(debt_pol(:)-debt_pol_new(:),inf));
    end

Q = Qnew;
    Vc = Vcnew;
    Vd = Vdnew;
    V0 = max(Vc,Vd);
    debt_pol = debt_pol_new;

    its = its + 1;
    
end
toc















%}




































