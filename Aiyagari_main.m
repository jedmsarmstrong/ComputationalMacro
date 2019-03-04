% Code to replicate the results of the Aiyagari (1994) paper
% jedmsarmstrong@nyu.edu | https://github.com/jedmsarmstrong
% March 2019

%#ok<*NOPTS>
%tic

cd('/home/jethro/Documents/Uni/Computational/Aiyagari_1994/')

clear
close all
clc

%% Simple case -- 2 states
% Log utility
% Made up P
P = [ 0.9 0.1 %0
      0.2 0.8 ]%0.2
      %0.2 0.1 0.7 ];
  
[~,e,V] = eig(P);
sd = V(:,diag(e)==1)./sum(V(:,diag(e)==1));
n = [0.1 1.0];
%n = [0.1 0.6 1.0];


R = 1 + 0.3;
alp = 1/3;
del = 0.05;
am = 18;
as = 200;
beta = 0.96;

A = 1;
N = n*sd;


w_r = @(r) A * (1-alp) * (A*alp / (r+del)) ^ (alp/(1-alp));
r_K = @(K) A * alp * (N/K) ^ (1-alp) - del;

n_grid = 100;
k_grid = linspace(1,4,n_grid);

n_state = length(n);
%%
K = 2;

r = r_K(K);
w = w_r(r);

% Cols = k, rows = n, 3rd = kprime
V = zeros(n_grid,n_state);
g = zeros(n_grid,n_state);

dist = 1;

while dist > 1e-5
    c = (R - del) * k_grid' + w*n - repmat(permute(k_grid,[3 1 2]),n_grid,n_state);
    u = log(c);
    u(c<0) = -inf;

    Ev = V*P';
    Vn = u + beta * repmat(permute(Ev,[3 2 1]),n_grid,1);

    [Vn,gn] = max(Vn,[],3);
    dist = max(max(abs(V-Vn))) + max(max(abs(g - gn)));

    V = Vn;
    g = gn;

end

%%
% Big transition matrix
mu = zeros(n_grid*n_state);

inds = reshape(g,[n_grid*n_state 1]);
for ii = 1 : length(inds)
    nind = floor((ii-1)/n_grid) + 1;
    mu(ii,inds(ii)+(0:n_grid:(n_state-1)*n_grid)) = P(nind,:);
end
% Update K
[~,e,V_mu] = eig(mu);
sd_mu = V_mu(:,abs(diag(e)-1)<1e-6)./sum(V_mu(:,abs(diag(e)-1)<1e-6));

%%
NN = 1000;
ks = randi(n_grid,NN,1);
zs = randi(n_state,NN,1);
mn = mean(ks);
sd = std(ks);

dist = 1;
while dist > 1e-3 
    ksp = g(sub2ind([n_grid n_state],ks,zs));
    zsp = randi(n_state,NN,1);
    
    mnp = mean(ksp);
    sdp = std(ksp);

    dist = max(abs(mnp - mn)) + max(abs(sdp - sd))
    
    ks = ksp;
    mn = mnp;
    sd = sdp;
    zs = zsp;
end



%%
K = 2;
dist2 = 1;
while dist2 > 1e-5
    r = r_K(K);
    w = w_r(r);

    % Cols = k, rows = n, 3rd = kprime
    V = zeros(n_grid,n_state);
    g = zeros(n_grid,n_state);

    dist = 1;

    while dist > 1e-5
        c = (R - del) * k_grid' + w*n - repmat(permute(k_grid,[3 1 2]),n_grid,n_state);
        u = log(c);
        u(c<0) = -inf;

        Ev = V*P';
        Vn = u + beta * repmat(permute(Ev,[3 2 1]),n_grid,1);

        [Vn,gn] = max(Vn,[],3);
        dist = max(max(abs(V-Vn))) + max(max(abs(g - gn)));

        V = Vn;
        g = gn;

    end

    % Big transition matrix
    mu = zeros(n_grid*n_state);

    inds = reshape(g,[n_grid*n_state 1]);
    for ii = 1 : length(inds)
        nind = floor((ii-1)/n_state) + 1;
        mu(ii,[inds(ii) inds(ii)+n_grid inds(ii)+2*n_grid]) = P(nind,:);
    end
    % Update K
    [~,e,V_mu] = eig(mu);
    sd_mu = V_mu(:,abs(diag(e)-1)<1e-6)./sum(V_mu(:,abs(diag(e)-1)<1e-6));
    Kn = k_grid(inds) * sd_mu;

    dist2 = abs(Kn-K)
    Kn = K
end
plot(V)

