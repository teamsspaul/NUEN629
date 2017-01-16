L=Lengths/2;L1=Lines(j,1);L2=Lines(j,2);

    if(Lines(j,3)==0) %m=0, sin(b)=0, cos(b)=1 (Horizontal Lines)
       if(x2<L2-L||x2>L2+L)
        CHECK=0;
       end
    elseif(isnan(Lines(j,3))) %m=INF,sin(b)=1,cos(b)=0 (Vertical Lines)
       if(y2<L1-L||y2>L1+L)
        CHECK=0; 
       end
	   
	   