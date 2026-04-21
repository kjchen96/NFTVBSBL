clear all;
close all;

%% Near-Field Wideband Total Variation-Regularized Block Sparse Bayesian Learning
N = 64;
Nt = 2 * N + 1;    % Number of Transmit Antenns
f = 30e9;          % Carrier Frequency
c = 3e8;           % Speed of Light
lambda = c/f;      % Wavelength
d = lambda/2;      % Antenna Spacing
AntennaIndex = (-N : 1 : N)';


B = 3e9;           % Bandwidth
S = 2;
M = 2 * S + 1;     % Number of Subcarriers
fm = linspace(-B/2,B/2,M) + f;   % Subcarrier Frequency
lambdam = c./fm;                 % Subcarrier Wavelength

%% Parameters
D = Nt * d;                      % Array Aperture
RaleighDistance = 2* D^2/lambda; 
FresnelDistance = 0.5 * sqrt(D^3/lambda);
bmin = 0;
bmax = lambda/4/FresnelDistance;
ThetaMin = -1;
ThetaMax = 1;

%% Dictionary Generation
ThetaQuantizedNum = Nt;          % S in Eq.(11)
bQuantizeNum = 10;               % T in Eq.(11)
ThetaQuantize = -1 + (2 * (1 : Nt)-1)/Nt; 
bQuantize = linspace(bmin,bmax,bQuantizeNum);

DeltaTheta = ThetaQuantize(2) - ThetaQuantize(1);
Deltab = bQuantize(2) - bQuantize(1);
Theta_Resolution = DeltaTheta/200;   % Gradient Descent Angle Step
b_Resolution = Deltab/200;           % Gradient Descent Distance Step

U = ThetaQuantizedNum * bQuantizeNum; % U below Eq. (13)
Theta_GridValue = zeros(1,U);
b_GridValue = zeros(1,U);


% Dictionary Generation for multiple subcarriers
PhiTotal = zeros(Nt,U,M);
for pp = 1 : M
    u = 0;
    for mm = 1 : bQuantizeNum
        for nn = 1 : ThetaQuantizedNum
            bTemp = bQuantize(mm);
            ThetaTemp = ThetaQuantize(nn);
            b_real = lambda/lambdam(pp) * bTemp;
            Theta_real = lambda/lambdam(pp) * ThetaTemp;
            SteeringVector =  exp(1j * pi * (Theta_real * AntennaIndex -  b_real * AntennaIndex.^2));
            u = u + 1;
            PhiTotal(:,u,pp) = SteeringVector;
        end
    end
end

% Off-Grid Values Store
u = 0;
for mm = 1 : bQuantizeNum
    for nn = 1 : ThetaQuantizedNum
        bTemp = bQuantize(mm);
        ThetaTemp = ThetaQuantize(nn);
        u = u + 1;
        b_GridValue(u) = bTemp;
        Theta_GridValue(u) = ThetaTemp;
     end
end



%% Block Generation for Channel
NumBlock = 3;
BlockSize = 3;
HalfBlockSize = (BlockSize-1)/2;
ThetaBlockCNumber = Nt;
bBlockCNumber = 10;

AngleBlockCenter = randperm(ThetaBlockCNumber,NumBlock);
DistanceBlockCenter = randperm(bBlockCNumber,NumBlock);

AngleIndex = cell(NumBlock,1);
DistanceIndex = cell(NumBlock,1);
for ii = 1 : NumBlock
     LeftIndex  = max(AngleBlockCenter(ii) - HalfBlockSize,1);
     RightIndex = min(AngleBlockCenter(ii) + HalfBlockSize,ThetaBlockCNumber);
     AngleIndex{ii} = LeftIndex : 1 : RightIndex;
end
for ii = 1 : NumBlock
     LeftIndex  = max(DistanceBlockCenter(ii) - HalfBlockSize,1);
     RightIndex = min(DistanceBlockCenter(ii) + HalfBlockSize,bBlockCNumber);
     DistanceIndex{ii} = LeftIndex : 1 : RightIndex;
end

ChannelPath = 0;
for ii = 1 : NumBlock
    ChannelPath = ChannelPath + length(AngleIndex{ii}) * length(DistanceIndex{ii});
end

%% Channel Generation
ThetaBlockQuantize = -1 + (2 * (1 : ThetaBlockCNumber)-1)/ThetaBlockCNumber;
bBlockQuantize =  linspace(bmin,bmax,bBlockCNumber);
DeltaBlockTheta = ThetaBlockQuantize(2) - ThetaBlockQuantize(1);
DeltaBlockb = bBlockQuantize(2) - bBlockQuantize(1);

 H = zeros(Nt,M);

% Channel Gains
MaxBlockNum = 3;
AngleStore = zeros(NumBlock,MaxBlockNum,MaxBlockNum);
DistanceStore = zeros(NumBlock,MaxBlockNum,MaxBlockNum);
ChannelStore = zeros(NumBlock,MaxBlockNum,MaxBlockNum);
for mm = 1 : NumBlock
    for nn = 1 : length(AngleIndex{mm})
        for ll = 1 : length(DistanceIndex{mm})
            ChannelThetaIndex = AngleIndex{mm}(nn);
            ChannelDistanceIndex = DistanceIndex{mm}(ll);
            AngleStore(mm,nn,ll) = ThetaBlockQuantize(ChannelThetaIndex) + (rand - 1/2) * DeltaBlockTheta;
            DistanceStore(mm,nn,ll) = bBlockQuantize(ChannelDistanceIndex) + (rand -1/2) * DeltaBlockb;
            ChannelStore(mm,nn,ll) = (randn + 1j * randn)/sqrt(2);
        end
    end
end
    
 
for pp = 1 : M
    h = zeros(Nt,1);
    for mm = 1 : NumBlock
        for nn = 1 : length(AngleIndex{mm})
            for ll = 1 : length(DistanceIndex{mm})
                ChannelThetaTemp = AngleStore(mm,nn,ll);
                ChannelDistanceTemp = DistanceStore(mm,nn,ll);
                ChannelThetaReal = ChannelThetaTemp * lambda/lambdam(pp);
                ChannelDistanceReal = ChannelDistanceTemp * lambda/lambdam(pp);
                ChannelGainTemp = ChannelStore(mm,nn,ll);
                 SteeringVector =  exp(1j * pi * (ChannelThetaReal * AntennaIndex -  ChannelDistanceReal * AntennaIndex.^2));
                h =  h + ChannelGainTemp * SteeringVector;
            end
        end
    end
    H(:,pp) = h;
end

%% Received Signals
SNR = 15;
P = 60;    
Pnoise     = 1/(10^(SNR/10));
F = (2 * round(rand(Nt,P)) - 1)/sqrt(Nt);   % W_p in Eq. (2)
Y = zeros(P,M);
for mm = 1 : M
    Y(:,mm) = F' * H(:,mm) + (randn(P,1) + 1j * randn(P,1))/sqrt(2) * sqrt(Pnoise);
end

PhiM = zeros(P,U,M);
for mm = 1 : M
    PhiM(:,:,mm) = F' * PhiTotal(:,:,mm);
end
%%  weights and neighborhood
[ei, ej, geo_w] = line_edges_2d(ThetaQuantizedNum, bQuantizeNum);
gamma  = ones(U,1);   % Initialization of gamma
z_prev = log(max(gamma,1e-12));
tv_scale = 1;
% M = 1;
w_prev = build_tv_paper(z_prev, ei, ej, U, tv_scale,  geo_w);

% Pre-compute edge-wise dictionary coherence across subcarriers.
% This is used later to slightly relax the sparsity penalty in
% highly coherent regions of the dictionary.
G_edge_all = precompute_edge_coherence(PhiM, ei, ej);
%% SBL-EM
k_max = 350;             % Maximum Iteration
pmax   = 100;            % PDHG Maximum Iteration
ptol   = 5e-5;           % PDHG Tolerance
g_clip = [-14,+14];      % Range of log(gamma)  
f_val = 1e-4;
Theta_OffGrid_Value = Theta_GridValue;
b_OffGrid_Value = b_GridValue;
MaxUpdateEtaBeta = 50;
 
Rho = ones(U,M); % Rho is set as one because all paths have equal energy in this simulation
alpha = 1;
gamma_prev = Inf * ones(U,1);
a1 = 1e-4;
b1 = 1e-4;
for k=1:k_max
    
    %%  posterior covariance 
    SigmaTotal = cell(M);
    for mm = 1 : M
        inv_g = diag(1./(gamma.* Rho(:,mm)));
        Phi = PhiM(:,:,mm);
        PhiInvG = Phi .* diag(inv_g)';
        C = eye(P) + alpha * PhiInvG * Phi';
        inv_C = inv(C);
        PhiInvGincC = PhiInvG' * inv_C;
        Sigma = inv_g - alpha *  PhiInvGincC * PhiInvG;
        SigmaTotal{mm} = Sigma;
    end
    %% posterior mean
    Mu = zeros(U,M);
    SigmaPhiM = cell(M);
    for mm = 1 : M
        SigmaPhiM{mm} = SigmaTotal{mm} * (PhiM(:,:,mm))';
        mu = alpha * SigmaPhiM{mm} * Y(:,mm);
        Mu(:,mm) = mu;
    end
    %%  alpha Update
      up_alpha = M * P + a1;
      down_alpha = b1;
      for mm = 1 : M
        down_alpha = down_alpha + norm(Y(:,mm) - PhiM(:,:,mm) * Mu(:,mm))^2 + abs(trace(PhiM(:,:,mm) * SigmaPhiM{mm}));
      end
      down_alpha = real(down_alpha);
      alpha = up_alpha/down_alpha;
  
    %% Chi
     chi = zeros(U,1);
     for mm = 1 : M
         Sigma = SigmaTotal{mm};
        chi = chi + Rho(:,mm).*(abs(Mu(:,mm)).^2 + real(diag(Sigma))); 
     end
     
    %% d Update
    % Parameters for the adaptive sparsity strength (can be tuned, but fixed in code)
    d_params.d_min    = 0;      % min TV weight at strong components
    d_params.d_max    = 3;      % max TV weight at background
    d_params.rho_ema  = 0.30;   % EMA factor over EM iterations
    d_params.beta_coh = 0.50;   % coherence correction strength
    d_params.L_tgt    = 27;     % paths reserved

    if k == 1
        d_vec_prev = [];
    else
        d_vec_prev = d_vec;
    end

    d_vec = update_d_vec(chi, SigmaTotal, Mu, ei, ej, G_edge_all, d_vec_prev, d_params);
%     d_vec = zeros(U,1);
 %% Gamma Update
    gamma = pdhg_prec_df(gamma, chi, ei, ej, w_prev, pmax, ptol, U, g_clip, M, d_vec, f_val);
    z_now  = log(max(gamma,1e-12));
    w_prev = build_tv_paper(z_now, ei, ej, U, tv_scale,geo_w);
    
 %% Gradient Descent
    [Value,Index] = sort(mean(abs(Mu),2),'descend');
    PartialBeta = zeros(U,1);
    PartialEta = zeros(U,1);
    for mm = 1 : M
        Phi = PhiTotal(:,:,mm);
        PartialPhiBeta = diag(AntennaIndex) * Phi * 1j * pi * lambda/lambdam(mm);
        PartialPhiEta = -diag(AntennaIndex.^2) * Phi * 1j * pi * lambda/lambdam(mm);
        y = Y(:,mm);
        mu = Mu(:,mm);
         Sigma = SigmaTotal{mm};
        FFPhi = F * F' * Phi;
        PhiSigma = FFPhi * Sigma;
        FFPhiMu = FFPhi * mu;
        for tt = 1 : MaxUpdateEtaBeta
                uu = Index(tt);
                Partial_Phi_betau = PartialPhiBeta(:,uu);
                partial_L1_betau = -2 * real(y' * F' * Partial_Phi_betau * mu(uu)) + ...
                                    2 * real(mu(uu)' * Partial_Phi_betau' * FFPhiMu);
                partial_L2_betau = 2 * real(Partial_Phi_betau' * PhiSigma(:,uu));
                PartialBeta(uu) = PartialBeta(uu) + partial_L1_betau + partial_L2_betau;


                Partial_Phi_etau = PartialPhiEta(:,uu);
                partial_L1_etau = -2 * real(y' * F' * Partial_Phi_etau * mu(uu)) + ...
                                    2 * real(mu(uu)' * Partial_Phi_etau' * FFPhiMu);
                partial_L2_etau = 2 * real(Partial_Phi_etau' * PhiSigma(:,uu));
                PartialEta(uu) = PartialEta(uu) + partial_L1_etau + partial_L2_etau;
        end
    end
    
    for uu = 1 : MaxUpdateEtaBeta
        ThetaUpdate = Theta_OffGrid_Value(Index(uu)) - sign(PartialBeta(Index(uu))) * Theta_Resolution;
        bUpdate = b_OffGrid_Value(Index(uu)) - sign(PartialEta(Index(uu))) * b_Resolution;
        Theta_OffGrid_Value(Index(uu)) =  ThetaUpdate;
        b_OffGrid_Value(Index(uu)) = bUpdate;
    end
     
   for mm = 1 : M
        for uu = 1 : MaxUpdateEtaBeta
            SteeringVector =  exp(1j * pi * (Theta_OffGrid_Value(Index(uu)) * lambda/lambdam(mm) * AntennaIndex -  b_OffGrid_Value(Index(uu)) * lambdam(mm)/lambda * AntennaIndex.^2));
            PhiTotal(:,Index(uu),mm) = SteeringVector;
            PhiM(:,Index(uu),mm) = F' * PhiTotal(:,Index(uu),mm);
        end
   end
end

%% NMSE 
MidIndex = 3;
[Value_Gamma,Index_Gamma] = sort(mean(abs(1./gamma),2),'descend');
Index_Gamma2 = find(1./gamma>=0.001*max(1./gamma));
Lest = length(Index_Gamma2);
He = zeros(Nt,Lest);
for ll = 1 : Lest
    ChannelThetaTemp = Theta_OffGrid_Value(Index_Gamma2(ll));
    ChannelDistanceTemp = b_OffGrid_Value(Index_Gamma2(ll));
    SteeringVector =  exp(1j * pi * (ChannelThetaTemp * AntennaIndex -  ChannelDistanceTemp * AntennaIndex.^2));
    He(:,ll) = SteeringVector;
end
HeW = F' * He;
ChannelGainEst = pinv(HeW) * Y(:,MidIndex);
ChannelEst = He * ChannelGainEst;
NMSE1 = norm(H(:,MidIndex) - ChannelEst)/norm(H(:,MidIndex));
disp(['Proposed Method ----------- NMSE:  ',num2str(NMSE1)]);

 
 
%% Genie-Aided LS
AngleStoreVec = AngleStore(:);
AngleStoreVec(AngleStoreVec == 0) = [];
DistanceStoreVec = DistanceStore(:);
DistanceStoreVec(DistanceStoreVec == 0) = [];

He = zeros(Nt,ChannelPath);
for ll = 1 : ChannelPath
    ChannelThetaTemp = AngleStoreVec(ll);
    ChannelDistanceTemp = DistanceStoreVec(ll);
    SteeringVector =  exp(1j * pi * (ChannelThetaTemp * AntennaIndex -  ChannelDistanceTemp * AntennaIndex.^2));
    He(:,ll) = SteeringVector;
end
HeW = F' * He;
ChannelGainEst = inv(HeW' * HeW) * HeW' * (Y(:,MidIndex));
ChannelEst = He * ChannelGainEst;
NMSE2 = norm(H(:,MidIndex) - ChannelEst)/norm(H(:,MidIndex));
disp(['Genie-Aided LS ----------- NMSE:  ',num2str(NMSE2)]);

%% Functions
%  Calculate the neighborhood
function [ei, ej, geo_w] = line_edges_2d(S, T)
     ei = []; ej = []; geo_w = [];

    %  four directions
    for t = 1:T
        for s = 1:S
            u = (t-1)*S + s;

            % right (s+1, t)
            if s < S
                v = (t-1)*S + (s+1);
                ei(end+1,1) = u; ej(end+1,1) = v; geo_w(end+1,1) = 1.0;  
            end

            % down (s, t+1)
            if t < T
                v = t*S + s;
                ei(end+1,1) = u; ej(end+1,1) = v; geo_w(end+1,1) = 1.0;  
            end
        end
    end
end


% =============== Total Variation-Regularized Weights ===============
% Following the works in 
% H. Djelouat, R. Leinonen, M. J. Sillanpää, B. D. Rao, and M. Juntti,
% “Adaptive and self-tuning SBL with total variation priors for block-
% sparse signal recovery,” IEEE Signal Process. Lett., vol. 32, pp. 1555–
% 1559, Apr. 2025.
function w = build_tv_paper(z, ei, ej, U, tv_scale,geo_w)
    if nargin < 5 || isempty(tv_scale), tv_scale = 1.0; end
    d = real(z(ei) - z(ej));
    beta_e = exp(-(d).^2) .* geo_w;
    rowsum = accumarray([ei; ej], [beta_e; beta_e], [U, 1]); 
    rowsum(rowsum == 0) = 1;
    w = 0.5 * ( beta_e ./ rowsum(ei) + beta_e ./ rowsum(ej) );
    w = (tv_scale) * max(w, 0);
end

% PDHG for gamma update
function gamma = pdhg_prec_df(gamma_in, chi, ei, ej, w, pmax, ptol, N, g_clip, M, d_vec, f_val)
z = log(max(gamma_in,1e-12));
E = numel(ei);
Ku  = @(zv)(zv(ei)-zv(ej));
KTy = @(y)(accumarray(ei,y,[N,1]) - accumarray(ej,y,[N,1]));
deg = accumarray([ei;ej],1,[N,1]); L2=max(1,2*max(deg));
tau=0.9/sqrt(L2); sigma=tau; theta=1.0;

y=zeros(E,1); zcur=z; zbar=zcur;
for t=1:pmax
    % dual
    y = y + sigma*Ku(zbar);
    y = min(max(y,-w), w);

    % primal
    z_old = zcur;
    v     = zcur - tau*KTy(y);
    zcur  = prox_prec_df(v, chi, tau, M, d_vec, f_val);

    % box
    zcur = min(max(zcur, g_clip(1)), g_clip(2));
    zbar = zcur + theta*(zcur - z_old);

    % stop
    if t>50 && norm(zcur-z_old)/max(1,norm(zcur))<ptol, break; end
end
gamma = exp(zcur);
gamma = min(max(gamma,1e-12), exp(g_clip(2)));
end

 
%  Newton's method
function z = prox_prec_df(v, chi, tau, M, d_vec, f_val)
z = v;            
for it = 1:80
    ez = exp(z);
    F  = (z - v) + tau*((chi + f_val).*ez - (M + d_vec));
    dF = 1 + tau*(chi + f_val).*ez;
    step = F ./ max(dF, 1e-16);
    z    = z - step;
    if max(abs(step(:))) < 1e-8, break; end
end
end


function d_vec = update_d_vec(chi, SigmaTotal, Mu, ei, ej, G_edge_all, d_vec_prev, params)
% Build spatially varying TV weight d_vec from posterior statistics.
% Key ideas:
%   1) use chi and graph neighbors to form a smoothed activity score s;
%   2) use a soft Top-K gating with target L_tgt;
%   3) slightly relax penalty in coherent regions;
%   4) map to [d_min, d_max] and apply EMA over EM iterations.

    U = numel(chi);
    M = size(Mu,2);

    %% 1) Local activity score s (posterior power + neighbors)
    deg = accumarray([ei; ej], 1, [U,1]);

    chi_nei_sum = accumarray(ei, chi(ej), [U,1], @sum, 0) ...
                + accumarray(ej, chi(ei), [U,1], @sum, 0);

    s = (chi + chi_nei_sum) ./ max(1, 1 + deg);   % U x 1

    %% 2) Subcarrier weights and node-level coherence
    Beta_m = zeros(M,1);
    for m = 1:M
        Sigma = SigmaTotal{m};
        Beta_m(m) = sum(abs(Mu(:,m)).^2 + real(diag(Sigma)));
    end
    w_m = Beta_m / max(sum(Beta_m), eps);         % normalized

    G_edge = G_edge_all * w_m;                    % E x 1

    m1 = accumarray(ei, G_edge, [U,1], @max, 0);
    m2 = accumarray(ej, G_edge, [U,1], @max, 0);
    m_loc = max(m1, m2);                          % local coherence

    m_min = min(m_loc);
    m_max = max(m_loc);
    if m_max > m_min
        m_loc = (m_loc - m_min) / (m_max - m_min);
    else
        m_loc = zeros(U,1);
    end

    %% 3) Soft-Top-K gating with target L_tgt
    L_tgt = params.L_tgt;                         % target number of "paths"

    s_sorted = sort(s);
    q10 = s_sorted(max(1, round(0.10*numel(s_sorted))));
    q90 = s_sorted(max(1, round(0.90*numel(s_sorted))));
    t_gate = max(1e-8, 0.1 * (q90 - q10));        % gating scale

    lo = min(s) - 20*t_gate;
    hi = max(s) + 20*t_gate;
    for it = 1:30
        tau = 0.5 * (lo + hi);
        g = 1 ./ (1 + exp(-(s - tau)/t_gate));   % soft selection
        if sum(g) > L_tgt
            lo = tau;
        else
            hi = tau;
        end
    end
    tau = 0.5 * (lo + hi);
    g   = 1 ./ (1 + exp(-(s - tau)/t_gate));

    %% 4) Coherence-based relaxation
    beta_coh = params.beta_coh;                  % e.g. 0.5
    g = g .* (1 + 0.2 * beta_coh * m_loc);
    g = min(1, max(0, g));                       % clip to [0,1]

    %% 5) Map to TV weight range and EMA
    d_min   = params.d_min;
    d_max   = params.d_max;
    d_new = d_min + (d_max - d_min) * (1 - g);

    rho_ema = params.rho_ema;                    % EMA factor
    if isempty(d_vec_prev)
        d_vec = d_new;
    else
        d_vec = (1 - rho_ema) * d_vec_prev + rho_ema * d_new;
    end
end



function G_edge_all = precompute_edge_coherence(PhiM, ei, ej)
% Compute edge-wise dictionary coherence for all subcarriers.
% G_edge_all(e,m) is the magnitude of the inner product between
% the two normalized columns connected by edge e on subcarrier m.

    [~, U, M] = size(PhiM); %#ok<ASGLU>
    E = numel(ei);
    G_edge_all = zeros(E, M);

    for m = 1:M
        PhiF = PhiM(:,:,m);              % (P x U)
        col_norms = vecnorm(PhiF,2,1) + eps;
        PhiF_n = PhiF ./ col_norms;      % column-normalized

        Ge = abs(sum(conj(PhiF_n(:,ei)) .* PhiF_n(:,ej), 1)).'; % (E x 1)
        G_edge_all(:,m) = Ge;
    end
end

