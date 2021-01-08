% Calculate draws of standardized random terms for random coefficients and save if specified
% Written by Kenneth Train, August 6, 2006.
 
% Array of draws has dimension NMEMxNPxNV.


function dr=makedraws
 
global IDV NV NP DRAWTYPE NMEM NTAKES NDRAWS PUTDR

if NTAKES>1
  fid=fopen(PUTDR,'w'); 
  dr=[];
else
  dr=zeros(NMEM,NP,NV);
end


if DRAWTYPE == 1              %Random draws
  if NTAKES == 1              %Create all at once
      for j=1:NV
        if IDV(j,2) ~=6              %Based on normal: not triangular
           dr(:,:,j)=randn(NMEM,NP);
        else                         %Triangular
           draws=rand(NMEM,NP);
           dr(:,:,j)=(sqrt(2.*draws)-1) .* (draws<=.5) + (1-sqrt(2.*(1-draws))) .* (draws >.5);
        end
     end
  else                       %Create and write NMEM draws for each person
                             %   since draws are to be read as NMEMxNP for each NV 
     for j=1:NV
         for p=1:NP
             for k=1:NTAKES
                if IDV(j,2)~=6
                   draws=randn(NMEM,1);
                else
                   draws=rand(NMEM,1);
                   draws=(sqrt(2.*draws)-1) .* (draws<=.5) + (1-sqrt(2.*(1-draws))) .* (draws >.5);
                end
                fwrite(fid,draws,'double');
             end
         end
     end
   end
end

if DRAWTYPE == 2 | DRAWTYPE == 3   % Halton draws
   h=primes(100);                                % Must create for all people together
   k=1;
   while size(h,2) < NV
       h=primes(k.*100);
       k=k+1;
   end
   h=h(1,1:NV);
   for j=1:NV
       hh=h(1,j);
       draws=[0];
       test=0;
       b=1;
       while test == 0
            drawsold=draws;
            for m=1:(hh-1);
                dd=m./(hh.^b);
                draws=[draws ; drawsold + dd];
                test=size(draws,1) >= ((NP.*NDRAWS) + 10);
                if test == 1
                   break
                end
            end
            b=b+1;    
       end
       draws=draws(11:(NP.*NDRAWS)+10,1);
       if DRAWTYPE == 3
            draws=draws+rand(1,1);               %Shift: one shift for entire sequence
            draws=draws-floor(draws);
            draws=reshape(draws,NDRAWS,NP);
            for n=1:NP                           %Shuffle for each person separately
               rr=rand(NDRAWS,1);
               [rr rrid]=sort(rr);
               draws(:,n)=draws(rrid,n);
            end;
            draws=reshape(draws,NP.*NDRAWS,1);
       end
       if IDV(j,2)~=6
           draws=-sqrt(2).*erfcinv(2.*draws);  %Take inverse cum normal
       else
           draws=(sqrt(2.*draws)-1) .* (draws<=.5) + (1-sqrt(2.*(1-draws))) .* (draws >.5); 
       end
       if NTAKES == 1
           dr(:,:,j)=reshape(draws,NDRAWS,NP);
       else
           fwrite(fid,draws,'double');
       end
    end
end

if DRAWTYPE == 4   % MLHS

  h=0:(NDRAWS-1);
  h=h'./NDRAWS;
  for j=1:NV
    for n=1:NP
       draws=h+rand(1,1)./NDRAWS;    %Shift: Different shift for each person
       rr=rand(NDRAWS,1);
       [rr rrid]=sort(rr);
       draws=draws(rrid,1);          %Shuffle
       if IDV(j,2)~=6
           draws=-sqrt(2).*erfcinv(2.*draws);  %Take inverse cum normal
       else
           draws=(sqrt(2.*draws)-1) .* (draws<=.5) + (1-sqrt(2.*(1-draws))) .* (draws >.5); 
       end
       if NTAKES == 1
         dr(:,n,j)=draws;
       else
         fwrite(fid,draws,'double');
       end
     end
   end
end

if NTAKES>1
  fclose(fid);
end