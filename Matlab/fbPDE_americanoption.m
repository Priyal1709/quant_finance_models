% American Option Pricing using Finite Differences + PSOR (Crank-Nicolson)
%
% This MATLAB code implements a Crank–Nicolson finite-difference solver for
% American options under the Black–Scholes model, using PSOR to handle the
% early exercise (free boundary).
%
% Reference: Wilmott, D., Howison, S., & Dewynne, J. (1995).
%
% Example usage:
% [price, S, V, exc] = AmericanOptionFD_PSOR(50,50,0.05,0,0.4,0.5,'put');
clear
clc


function [price, S, V, exercise_boundary] = AmericanOptionFD_PSOR(S0,K,r,q,sigma,T,option_type)

    % Grid parameters
    Nx = 400;   % number of stock steps
    Nt = 400;   % number of time steps
    Smax = 5*max(S0,K);
    S = linspace(0,Smax,Nx)';
    dx = S(2)-S(1);
    dt = T/Nt;

    % Payoff
    if strcmp(option_type,'put')
        payoff = max(K - S,0);
    else
        payoff = max(S - K,0);
    end

    % Initialize option values at maturity
    V = payoff;

    % Precompute coefficients for interior points
    i = (2:Nx-1)';
    Si = S(i);
    a = 0.5*sigma^2*Si.^2;
    b = (r - q)*Si;

    alpha = a/dx^2 - b/(2*dx);
    beta  = -2*a/dx^2 - r;
    gamma = a/dx^2 + b/(2*dx);

    % Sparse matrices
    M_left  = spdiags([-alpha, 1 - dt*0.5*beta, -gamma],[-1,0,1],Nx-2,Nx-2);
    M_right = spdiags([alpha, 1 + dt*0.5*beta, gamma],[-1,0,1],Nx-2,Nx-2);

    % PSOR parameters
    tol   = 1e-6;
    maxit = 10000;
    omega = 1.3;

    exercise_boundary = nan(Nt+1,1);

    % Step back in time
    for n = Nt:-1:1
        rhs = M_right * V(2:end-1);

        % Boundary conditions
        if strcmp(option_type,'put')
            Vlow  = K*exp(-r*(T - (n-1)*dt));
            Vhigh = 0;
        else
            Vlow  = 0;
            Vhigh = Smax - K*exp(-r*(T - (n-1)*dt));
        end

        rhs(1)  = rhs(1)  + alpha(1)*Vlow;
        rhs(end)= rhs(end)+ gamma(end)*Vhigh;

        % Initial guess
        V_new = V(2:end-1);
        payoff_interior = payoff(2:end-1);

        % Extract diagonals
        diagM   = diag(M_left);
        lowerM  = [0; diag(M_left,-1)];
        upperM  = [diag(M_left,1); 0];

        % PSOR loop
        for it=1:maxit
            maxdiff = 0;
            for j=1:length(V_new)
                res = rhs(j);
                if j>1
                    res = res - lowerM(j)*V_new(j-1);
                end
                if j<length(V_new)
                    res = res - upperM(j)*V_new(j+1);
                end
                v_old = V_new(j);
                v_unrelaxed = res/diagM(j);
                v_relaxed = v_old + omega*(v_unrelaxed - v_old);
                v_proj = max(v_relaxed, payoff_interior(j));
                V_new(j) = v_proj;
                maxdiff = max(maxdiff, abs(v_proj - v_old));
            end
            if maxdiff < tol
                break
            end
        end

        % Update full vector with boundaries
        V = [Vlow; V_new; Vhigh];

        % Approximate exercise boundary
        diffV = V - payoff;
        idx = find(diffV <= 1e-6,1,'first');
        if ~isempty(idx)
            exercise_boundary(n) = S(idx);
        end
    end

    % Interpolate price
    price = interp1(S,V,S0);
end

% Example script to run and plot

S0 = 50; K = 50; r = 0.05; q = 0; sigma = 0.4; T = 0.5;
[price,S,V,exc] = AmericanOptionFD_PSOR(S0,K,r,q,sigma,T,'put');
fprintf('American put price (FD+PSOR) = %.6f\n', price);

figure(1)
plot(S,V,'b-',S,max(K-S,0),'r--');
xlabel('Stock Price S'); 
ylabel('Option Value');
title('American Put Value at t=0'); 
legend('Option','Payoff'); 
grid on;

figure(2); 
plot(linspace(0,T,length(exc)),exc,'o-');
xlabel('Time'); 
ylabel('Exercise Boundary'); 
title('Early Exercise Boundary');
set(gca,'XDir','reverse'); 
grid on;

