% This code checks the input data and specifications that the user provides in mxlhb.m
% Written by Kenneth Train, July 26, 2006, latest edits on Sept 24,2006.

function ok=check

global NP NCS NROWS
global IDV NV NAMES B W
global IDF NF NAMESF F
global DRAWTYPE NDRAWS SEED1 PUTDR NMEM
global WANTWGT IDWGT 
global NALTMAX NCSMAX
global X XF S DR
global XMAT


% Check for positive intergers

ok=0;

if ceil(NP) ~= NP | NP < 1;
   disp(['NP must be a positive integer, but it is set to ' num2str(NP)]);
   disp('Program terminated.');
   return
end

if ceil(NCS) ~= NCS | NCS < 1;
   disp(['NCS must be a positive integer, but it is set to ' num2str(NCS)]);
   disp('Program terminated.');
   return
end

if ceil(NROWS) ~= NROWS | NROWS < 1;
   disp(['NROWS must be a positive integer, but it is set to ' num2str(NROWS)]);
   disp('Program terminated.');
   return
end

if ceil(NDRAWS) ~= NDRAWS | NDRAWS < 1;
   disp(['NDRAWS must be a positive integer, but it is set to ' num2str(NDRAWS)]);
   disp('Program terminated.');
   return
end

if ceil(DRAWTYPE) ~= DRAWTYPE | DRAWTYPE < 1;
   disp(['DRAWTYPE must be a positive integer, but it is set to ' num2str(DRAWTYPE)]);
   disp('Program terminated.');
   return
end


if ceil(SEED1) ~= SEED1 | SEED1 < 1;
   disp(['SEED1 must be a positive integer, but it is set to ' num2str(SEED1)]);
   disp('Program terminated.');
   return
end

if ceil(NMEM) ~= NMEM | NMEM < 1;
   disp(['NMEM must be a positive integer, but it is set to ' num2str(NMEM)]);
   disp('Program terminated.');
   return
end


if NV>0 & sum(sum(ceil(IDV) ~= IDV | IDV < 1),2) ~ 0;
   disp('IDV must contain positive integers only, but it contains other values.');
   disp('Program terminated.');
   return
end

if NF>0 & sum(sum(ceil(IDF) ~= IDF | IDF < 1),2) ~ 0;
   disp('IDF must contain positive integers only, but it contains other values.');
   disp('Program terminated.');
   return
end

% Checking XMAT %
if ( size(XMAT,1) ~= NROWS)
      disp(['XMAT has ' num2str(size(XMAT,1)) ' rows']);
      disp(['but it should have NROWS= '  num2str(NROWS)   ' rows.']);
      disp('Program terminated.');
      return
end

if sum(XMAT(:,1) > NP) ~= 0
     disp(['The first column of XMAT has a value greater than NP= ' num2str(NP)]);
     disp('Program terminated.');
     return
end

if sum(XMAT(:,1) < 1) ~= 0
     disp('The first column of XMAT has a value less than 1.');
     disp('Program terminated.');
     return
end

k=(XMAT(2:NROWS,1) ~= XMAT(1:NROWS-1,1)) & (XMAT(2:NROWS,1) ~= (XMAT(1:NROWS-1,1)+1));
if sum(k) ~= 0
    disp('The first column of XMAT does not ascend from 1 to NP.');
    disp('Program terminated.')
    return
end

if sum(XMAT(:,2) > NCS) ~= 0
     disp(['The second column of XMAT has a value greater than NCS= ' num2str(NCS)]);
     disp('Program terminated.');
     return
end

if sum(XMAT(:,2) < 1) ~= 0
     disp('The second column of XMAT has a value less than 1.');
     disp('Program terminated.');
     return
end

k=(XMAT(2:NROWS,2) ~= XMAT(1:NROWS-1,2)) & (XMAT(2:NROWS,2) ~= (XMAT(1:NROWS-1,2)+1));
if sum(k) ~= 0
    disp('The second column of XMAT does not ascend from 1 to NCS.');
    disp('Program terminated.')
    return
end


if sum(XMAT(:,3) ~= 0 & XMAT(:,3) ~= 1) ~= 0
     disp('The third column of XMAT has a value other than 1 or 0.');
     disp('Program terminated.');
     return
end

for s=1:NCS
    k=(XMAT(:,2) == s);
    if sum(XMAT(k,3)) > 1
       disp('The third column of XMAT indicates more than one chosen alternative');
       disp(['for choice situation ' num2str(s)]);
       disp('Program terminated.');
       return
    end
    if sum(XMAT(k,3)) < 1
       disp('The third column of XMAT indicates that no alternative was chosen');
       disp(['for choice situation ' num2str(s)]);
       disp('Program terminated.');
       return
    end 
end

if sum(sum(isnan(XMAT)),2) ~= 0
   disp('XMAT contains missing data.');
   disp('Program terminated.');
   return
end;

if sum(sum(isinf(XMAT)),2) ~= 0
   disp('XMAT contains an infinite value.');
   disp('Program terminated.');
   return
end;

if NV>0 & size(IDV,2) ~= 2;
   disp(['IDV must have 2 columns and yet it is set to have ' num2str(size(IDV,2))]);
   disp('Program terminated.');
   return
end;

if NV>0 & sum(IDV(:,1) > size(XMAT,2)) ~= 0;
   disp('IDV identifies a variable that is outside XMAT.');
   disp('The first column of IDV is');
   IDV(:,1)
   disp('when each element of this column must be no greater than')
   disp([num2str(size(XMAT,2)) ' which is the number of columns in XMAT.']);
   disp('Program terminated.');
   return
end;

if NV>0 & sum(IDV(:,1) <= 3) ~= 0;
   disp('Each element in the first column of IDV must exceed 3');
   disp('since the first three variables in XMAT cannot be explanatory variables.');
   disp('But the first column of IDV is');
   IDV(:,1)
   disp('which has an element below 3.')
   disp('Program terminated.');
   return
end;

if NV>0 & sum(IDV(:,2) < 0 | IDV(:,2) > 6) ~= 0;
   disp('The second column of IDV must be integers 1-6 identifying the distributions.');
   disp('But the second column of IDV is specified as');
   IDV(:,2)
   disp('which contains a number other than 1-6.')
   disp('Program terminated.');
   return
end;


if NV>0 & size(NAMES,2) ~= 1;
   disp(['NAMES must have 1 columns and yet it is set to have ' num2str(size(NAMES,2))]);
   disp('Be sure to separate names by semicolons.');
   disp('Program terminated.');
   return
end;

if NV>0 & size(IDV,1) ~= size(NAMES,1);
   disp(['IDV and NAMES must have the same length but IDV has length ' num2str(size(IDV,1))]);
   disp(['while NAMES has length ' num2str(size(NAMES,1))]);
   disp('Program terminated.');
   return
end; 

if NV>0 & size(B,2) ~= 1;
   disp(['B must have 1 column and yet it is set to have ' num2str(size(B,2))]);
   disp('Be sure to separate values by semicolons.');
   disp('Program terminated.');
   return
end;
  
if NV>0 & size(B,1) ~= size(IDV,1);
   disp(['B must have the same length as IDV but instead has length ' num2str(size(B,1))]);
   disp('Program terminated.');
   return
end; 

if NV>0 & size(W,2) ~= 1;
   disp(['W must have 1 column and yet it is set to have ' num2str(size(W,2))]);
   disp('Be sure to separate values by semicolons.');
   disp('Program terminated.');
   return
end;
  
if NV>0 & size(W,1) ~= size(IDV,1);
   disp(['W must have the same length as IDV but instead has length ' num2str(size(W,1))]);
   disp('Program terminated.');
   return
end; 


if NF>0 & size(IDF,2) ~= 1;
   disp(['IDF must have 1 column and yet it is set to have ' num2str(size(IDF,2))]);
   disp('Be sure to separate elements by semicolons.');
   disp('Program terminated.');
   return
end;

if NF>0 & sum(IDF > size(XMAT,2)) ~= 0;
   disp('IDF identifies a variable that is outside XMAT.');
   disp('IDF is');
   IDF
   disp('when each element must be no greater than');
   disp([num2str(size(XMAT,2)) ' which is the number of columns in XMAT.']);
   disp('Program terminated.');
   return
end;

if NF>0 & sum(IDF <= 3) ~= 0;
   disp('Each element of IDF must exceed 3 since the first three variables');
   disp('of XMAT cannot be explanatory variables.');
   disp('But IDV is');
   IDV(:,1)
   disp('which contains an element below 3.')
   disp('Program terminated.');
   return
end;

if NF>0 & size(NAMESF,2) ~= 1;
   disp(['NAMESF must have 1 column and yet it is set to have ' num2str(size(NAMESF,2))]);
   disp('Be sure to separate names by semicolons.');
   disp('Program terminated.');
   return
end;

if NF>0 & size(IDF,1) ~= size(NAMESF,1);
   disp(['IDF and NAMESF must have the same length but IDF has length ' num2str(size(IDF,1))]);
   disp(['while NAMESF has length ' num2str(size(NAMESF,1))]);
   disp('Program terminated.');
   return
end; 

if NF>0 & size(F,2) ~= 1;
   disp(['F must have 1 column and yet it is set to have ' num2str(size(F,2))]);
   disp('Be sure to separate values by semicolons.');
   disp('Program terminated.');
   return
end;
  
if NF>0 & size(F,1) ~= size(IDF,1);
   disp(['F must have the same length as IDF but instead has length ' num2str(size(F,1))]);
   disp('Program terminated.');
   return
end; 

if sum(DRAWTYPE < 0 | DRAWTYPE > 5) ~= 0;
   disp('DRAWTYPE must be an integer 1-5 identifying the type of draws.');
   disp(['But DRAWTYPE is set to' num2str(DRAWTYPE)]);
   disp('Program terminated.');
   return
end;

if DRAWTYPE == 5
   if size(DR,1) ~= NV | size(DR,2) ~= NP | size(DR,3) ~= NDRAWS
     disp('The DR that you loaded or created has the wrong dimensions.');
     disp('It has dimensions');
     size(DR)
     disp('when it should have dimensions NVxNPxNDRAWS, which is');
     disp([NV NP NDRAWS]);
     disp('Program terminated.');
     return
    end
end

if DRAWTYPE == 5 & NMEM~=NDRAWS
    disp('When DRAWTYPE=5, NMEM must equal NDRAWS.');
    disp(['However, NMEM is ' num2str(NMEM) ' while NDRAWS is ' num2str(NDRAWS)]);
    disp('Program terminated.');
    return
end

if ceil(NDRAWS./NMEM) ~= (NDRAWS./NMEM);
  disp('NDRAWS must be an integer multiple of NMEM.');
  disp(['However NDRAWS./NMEM is ' num2str(NDRAWS./NMEM)]);
  disp('Program terminated')
  return
end

if WANTWGT ~= 0 & WANTWGT ~= 1
    disp(['WANTWGT must be 0 or 1, but it is set to ' num2str(WANTWGT)]);
    disp('Program terminated')
  return
end

if WANTWGT==1 & size(IDWGT,1) ~= 1
    disp('When WANTWGT==1, as you have set it, then');
    disp('IDWGT must be an scalar that identifies a variable in XMAT for the weights');
    disp(['But IDWGT is set to ' num2str(IDWGT)]);
    disp('Program terminated')
  return
end

if WANTWGT== 1 & IDWGT > size(XMAT,2)
    disp('When WANTWGT==1, as you have set it, then');
    disp('IDWGT must identify a variable in XMAT for the weights');
    disp(['But IDWGT is set to ' num2str(IDWGT)]);
    disp(['when XMAT has only ' num2str(size(XMAT,2)) ' variables.']);
    disp('Program terminated')
  return
end

if WANTWGT == 1 & ( IDWGT < 1 | ceil(IDWGT) ~= IDWGT )
    disp('IDWGT must be a positive integer indentifying a variable in XMAT');
    disp(['but it is set to ' num2str(IDWGT)]);
    disp('Program terminated')
  return
end

if WANTWGT==1
    cp=XMAT(:,1);
    for r=1:NP
        if sum(XMAT(cp==r,IDWGT)~= mean(XMAT(cp==r,IDWGT)))>0
            disp(['Variable identified by IDWGT is ' num2str(IDWGT)]);
            disp('This weight variable must be the same for all rows of data for each person');
            disp(['However, it is not the same for all rows for person ' num2str(r)]);
            disp('and maybe for people after that person (not checked).');
            disp('Program terminated.')
            return
        end
    end
end

if NMEM<NDRAWS & size(PUTDR,1)==0
    disp('You need to give a name to the file PUTDR.');
    disp('Program terminated.')
    return
end

ok=1;