% This script calls check to check the data and input specifications, transforms the data
% into a more easily useable form, calls estimation routine, and prints out results.
% Written by Kenneth Train on July 19, 2006, and latest edits on Sept 24, 2006.


% Check the input data and specifications

disp('Checking inputs.');
ok=check;
if ok == 1;
    disp('Inputs have been checked and look fine.');
else
    return;
end

% Create Global variables to use in estimation

disp('Creating data arrays for run.');
cp=XMAT(:,1); % person
 
nn=zeros(NCS,1);
for n=1:NCS;
    nn(n,1)=sum(XMAT(:,2) == n,1);
end;
NALTMAX=max(nn);  %Maximum number of alternatives in any choice situation

nn=zeros(NP,1);
for n=1:NP;
   k=(XMAT(:,1)==n);
   k=XMAT(k,2);
   nn(n,1)=1+k(end,1)-k(1,1);
end;
NCSMAX=max(nn);  %Maximum number of choice situations faced by any person

NTAKES=NDRAWS./NMEM; %This tells code how many passes through the draws are needed 
                           % given NMEM people in each pass.
if WANTWGT == 1;
    WGT=zeros(1,NP);
    for r=1:NP
        WGT(1,r)=mean(XMAT(cp == r,IDWGT),1);
    end
    WGT=WGT.*(NP./sum(WGT,2));
else
    WGT=[];
end

% Data arrays 
% All variables are differenced from the chosen alternative
% Only nonchosen alternatives are included, since V for chosen alt =0
% This reduces number of calculations for logit prob and eliminates need to
% retain dependent variable.

X=zeros(NALTMAX-1,NCSMAX,NV,NP); % Explanatory variables with random coefficients 
%                                  for all choice situations, for each person 
XF=zeros(NALTMAX-1,NCSMAX,NF,NP); % Explanatory variables with fixed coefficients 
%                                  for all choice situations, for each person 
S=zeros(NALTMAX-1,NCSMAX,NP); % Identification of the alternatives in each choice situation, for each person

for n=1:NP;  %loop over people
 cs=XMAT(cp == n,2);
 yy=XMAT(cp == n,3);
 if NV > 0
    xx=XMAT(cp == n, IDV(:,1));
 end
 if NF > 0
    xxf=XMAT(cp == n, IDF(:,1));
 end
 t1=cs(1,1);
 t2=cs(end,1);
 for t=t1:t2; %loop over choice situations
     k=sum(cs==t)-1; %One less than number of alts = number of nonchosen alts
     S(1:k,1+t-t1,n)=ones(k,1);
     if NV>0
        X(1:k,1+t-t1,:,n)=xx(cs==t & yy == 0,:)-repmat(xx(cs==t & yy == 1,:),k,1);
     end
     if NF>0
        XF(1:k,1+t-t1,:,n)=xxf(cs==t & yy == 0,:)-repmat(xxf(cs==t & yy == 1,:),k,1);
     end;
 end
end


clear global XMAT 
clear cp cs yy xx t1 t2 k nn
    
randn('state',SEED1)  %For draws from normal
rand('state',SEED1)   %For draws from uniform

%Create draws

if DRAWTYPE ~= 5
   disp('Creating draws.');
   DR=makedraws;   %NMEMxNPxNV
   if NTAKES == 1
      DR=permute(DR,[3,2,1]);   %To make NVxNPxNDRAWS
   else 
      MDR=memmapfile(PUTDR,'Format',{'double',[NDRAWS,NP,NV],'drs'});
   end
end

if NV>0 & NF>0
   param=[F;B(IDV(:,2)~=5,1);W];
elseif NV>0 & NF==0
   param=[B(IDV(:,2)~=5,1);W];
elseif NV==0 & NF>0;
   param=F;
else
   disp('Model has no explanatory variables.');
   disp('IDV and IDF are both empty.');
   disp('Program terminated.');
   return
end

disp('Start estimation');
disp('The negative of the log-likelihood is minimized,');
disp('which is the same as maximizing the log-likelihood.');
tic;
options=optimset('LargeScale','off','Display','iter','GradObj','on',...
    'MaxFunEvals',10000,'MaxIter',MAXITERS,'TolX',PARAMTOL,'TolFun',LLTOL,'DerivativeCheck','off');
[paramhat,fval,exitflag,output,grad,hessian]=fminunc(@loglik,param,options);

disp(' ');
disp(['Estimation took ' num2str(toc./60) ' minutes.']);
disp(' ');
if exitflag == 1
  disp('Convergence achieved.');
elseif exitflag == 2
  disp('Convergence achieved by criterion based on change in parameters.');
  if size(PARAMTOL,1)>0
     disp(['Parameters changed less than PARAMTOL= ' num2str(PARAMTOL)]);
  else
     disp('Parameters changed less than PARAMTOL=0.000001, set by default.');
  end
  disp('You might want to check whether this is actually convergence.');
  disp('The gradient vector is');
  grad
elseif exitflag == 3
  disp('Convergence achieved by criterion based on change in log-likelihood value.');
  if size(PARAMTOL,1)>0
     disp(['Log-likelihood value changed less than LLTOL= ' num2str(LLTOL)]);
  else
     disp('Log-likelihood changed less than LLTOL=0.000001, set by default.');
  end
     disp('You might want to check whether this is actually convergence.');
     disp('The gradient vector is');
     grad
else
    disp('Convergence not achieved.');
    disp('The current value of the parameters and hessian');
    disp('can be accesses as variables paramhat and hessian.');
    disp('Results are not printed because no convergence.');
    return
end

disp(['Value of the log-likelihood function at convergence: ' num2str(-fval)]);

%Calculate standard errors of parameters
disp(' ');
disp('Taking inverse of hessian for standard errors.');
disp(' ');
ihess=inv(hessian);
stderr=sqrt(diag(ihess));
disp(['The value of grad*inv(hessian)*grad is: ' num2str(grad'*ihess*grad)]);

%Segment parameters and account for unestimated zero mean in distribution 5
if NF>0
  fhat=paramhat(1:NF,1);
  fsd=stderr(1:NF,1);
end

if NV>0
  if sum(IDV(:,2) == 5) >0;
     bhat=zeros(NV,1);
     bsd=zeros(NV,1)
     bhat(IDV(:,2) ~= 5,1)=paramhat(NF+1:NF+sum(IDV(:,2) ~= 5),1);
     bsd(IDV(:,2) ~= 5,1)=stderr(NF+1:NF+sum(IDV(:,2) ~= 5),1);
     what=paramhat(NF+sum(IDV(:,2) ~= 5)+1:end,1);
     wsd=stderr(NF+sum(IDV(:,2) ~= 5)+1:end,1);
  else;
     bhat=paramhat(NF+1:NF+NV,1);
     bsd=stderr(NF+1:NF+NV,1);
     what=paramhat(NF+NV+1:NF+NV+NV,1);
     wsd=stderr(NF+NV+1:NF+NV+NV,1);
  end;
end


disp('RESULTS');
disp(' ');
disp(' ')
if NF>0
disp('FIXED COEFFICIENTS');
disp(' ');
disp('                      F      ');
disp('              ------------------ ');
disp('                Est         SE ');
for r=1:length(NAMESF);
    fprintf('%-10s %10.4f %10.4f\n', NAMESF{r,1}, [fhat(r,1) fsd(r,1)]);
end
disp(' ');
end

if NV>0;
disp('RANDOM COEFFICIENTS');

disp(' ');
disp('                      B                      W');
disp('              ------------------   -----------------------');
disp('                 Est     SE            Est         SE');
for r=1:length(NAMES);
    fprintf('%-10s %10.4f %10.4f %10.4f %10.4f\n', NAMES{r,1}, [bhat(r,1) bsd(r,1) what(r,1) wsd(r,1)]);
end


%Create draws of coefficients from B-hat and W-hat

C=trans(bhat,what,DR);
C=reshape(C,NV,NP*NMEM);

disp('Distribution of coefficients in population implied by B-hat and W-hat.');
disp('using last NMEM draws.');
disp(' ');
jj={'normal';'lognormal';'truncnormal';'S_B';'normal0mn';'triangular'};
disp('                            Mean      StdDev     Share<0    Share=0');
kk=[mean(C,2) std(C,0,2) mean((C < 0),2) mean((C == 0),2)];
for r=1:length(NAMES);
    mm=IDV(r,2);
    fprintf('%-10s %-11s %10.4f %10.4f %10.4f %10.4f\n', NAMES{r,1}, jj{mm,1},kk(r,:));
end
disp(' ');

end

disp(' ');
disp('ESTIMATED PARAMETERS AND FULL COVARIANCE MATRIX.');
disp('The estimated values of the parameters are:');
paramhat
disp('The covariance matrix for these parameters is:');
ihess

disp(' ');
disp('You can access the estimated parameters as variable paramhat,');
disp('the gradient of the negative of the log-likelihood function as variable grad,');
disp('the hessian of the negative of the log-likelihood function as variable hessian,');
disp('and the inverse of the hessian as variable ihess.');
disp('The hessian is calculated by the BFGS updating procedure.');




    

