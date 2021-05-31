% Calculates log-likelihood function value for mixed logit model
% Written by Kenneth Train, July 27, 2006, revised July 31, 2006
%
% This code is input to Matlab's funcmin command
%
% Input param is a column vector of parameters, dimension (NF+NV+NV)x1
%     containing the fixed coefficients, the first parameters of the random
%     coefficients, and then the second parameters of the random coefficients
% Output ll is the scalar value of the negative of the simulated log-likelihood 
%     at the input parameters

function [ll, g] =loglik(param)

global NV NF IDV WANTWGT WGT

if NF>0
  f=param(1:NF,1);
else
  f=[];
end

if NV>0
  if sum(IDV(:,2) == 5) >0;
     b=zeros(NV,1);
     b(IDV(:,2) ~= 5,1)=param(NF+1:NF+sum(IDV(:,2) ~= 5),1);
     w=param(NF+sum(IDV(:,2) ~= 5)+1:end,1);
  else;
     b=param(NF+1:NF+NV,1);
     w=param(NF+NV+1:NF+NV+NV,1);
  end;
else
  b=[];
  w=[];
end

[p g]=llgrad2(f,b,w); 

if WANTWGT == 0
    ll=-sum(log(p),2);
    g=-sum(g,2);
else
    ll=-sum(WGT.*log(p),2);
    g=-sum(repmat(WGT,size(g,1),1).*g,2);
end

if NV>0 & sum(IDV(:,2)==5) >0 ;  %Zero mean error components
   z=[ones(NF,1) ; IDV(:,2) ~= 5 ; ones(NV,1)];
   g=g(z==1,1);
end


