function [distance,P,P_Store,nn]=Distances(Vector,R,xt,yt,Em,Ef,distance,P,P_Store,nn,Length,mfp,Min_dis,Q_CHECK)
passed=0;

%Please note...this code will fail if you start in the fuel. Min_dis will
%need to be modified.



%Determine Distance/FLUX Stuff
a=1+Vector(1,3)^2;
b=2*(Vector(1,3)*Vector(1,1)-Vector(1,2)*(Vector(1,3)^2));
c=(Vector(1,3)^2)*(Vector(1,2)^2)+Vector(1,1)^2-2*Vector(1,3)*Vector(1,2)*Vector(1,1)-R^2;
%Determine if we passed through fuel

debug=0;
%For Debugging purposes
%disp('       mv        x1        y1      R            a        b          c');
%disp([Vector(1,3),Vector(1,2),Vector(1,1),R,a,b,c]);
%debug=1;
%disp(' V_sin(x4)    V_cos(y5)');
%disp([Vector(1,4),Vector(1,5)]);

if (abs(Vector(1,5))<0.000001 && isreal((R^2-xt^2)^0.5)) %If cos(b)=0 %Vertical  
    xcmin=xt;xcmax=xt;
    ycmin=-1*(R^2-xt^2)^0.5;
    ycmax=(R^2-xt^2)^0.5;
    passed=1;
elseif (abs(Vector(1,4))<0.000001 && isreal((R^2-yt^2)^0.5)) %If sin(b)=0 %Horizontal   
    ycmin=yt;ycmax=yt;
    xcmin=-1*(R^2-yt^2)^0.5;
    xcmax=(R^2-yt^2)^0.5;
    passed=1;
elseif (isreal((b^2-4*a*c)^0.5) && abs(Vector(1,5))>0.000001 && abs(Vector(1,4))>0.000001)
    xcmin=(-b-(b^2-4*a*c)^0.5)/(2*a);
    xcmax=(-b+(b^2-4*a*c)^0.5)/(2*a);
    ycmin=Vector(1,3)*(xcmin-Vector(1,2))+Vector(1,1);
    ycmax=Vector(1,3)*(xcmax-Vector(1,2))+Vector(1,1);
    passed=1;
end

%For Debugging purposes
%if(passed==1)
%disp('       x2        y2      xcmax    ycmax     xcmin    ycmin');
%disp([xt,yt,xcmax,ycmax,xcmin,ycmin]);
%end

if(passed==1)%If we passed through fuel
    %If the vector is left traveling the dot with -x should be less than 90
    dot=acosd(-1*Vector(1,5));
    distance_fuel=((xcmax-xcmin)^2+(ycmax-ycmin)^2)^0.5;
    if(distance_fuel<0.00001||Length<Min_dis) %Either on edge of circle
        passed=0; %Or starting in the moderator somewhere
    %Below is redundant, and sometimes causes errors, but I want to keep it
    %to remind myself about where I came from. 
    %elseif(Vector(1,4)==0)%traveling at 180 or 0 degrees
    %    disp('I was here')
    %    distance_before=((xcmax-Vector(1,2))^2+(ycmax-Vector(1,1))^2)^0.5;
    %    distance_after=((xt-xcmin)^2+(yt-ycmin)^2)^0.5;
    elseif(dot<90) %Traveling leftward
        distance_before=((xcmax-Vector(1,2))^2+(ycmax-Vector(1,1))^2)^0.5; %distance traveled before the fuel
        distance_after=((xt-xcmin)^2+(yt-ycmin)^2)^0.5;
    elseif(dot>90) %Traveling rightward
        distance_before=((xcmin-Vector(1,2))^2+(ycmin-Vector(1,1))^2)^0.5;
        distance_after=((xt-xcmax)^2+(yt-ycmax)^2)^0.5;
    end

    if(passed==1) %Some redundancy is good.
        if(abs(Length-distance_fuel-distance_before-distance_after)>0.000001)
          disp('Your lengths are not adding up...do not worry about it');
          disp('    Fuel D    Before     After    Length');
          disp([distance_fuel,distance_before,distance_after,Length]);
        end
        distance=distance+distance_before;
        if(Q_CHECK==0)
            P_Store(nn)=distance*Em+P_Store(nn-1); %part for moderator
            nn=nn+1; %for the fuel
            P_Store(nn)=distance_fuel*Ef+P_Store(nn-1);
            P=P_Store(nn);
            nn=nn+1; %new era of moderator
            distance=distance_after; %Resetting distance
        else
            P_Store(nn)=distance*Em*0+P_Store(nn-1); %part for moderator
            nn=nn+1; %for the fuel
            P_Store(nn)=distance_fuel*Ef+P_Store(nn-1)+distance*Ef;
            P=P_Store(nn);
            nn=nn+1; %new era of moderator
            distance=distance_after; %Resetting distance 
        end
    else %Some Redundancy is bad
        distance=distance+Length;
    end
else %If we did not pass through fuel
distance=distance+Length;
end

if(Q_CHECK==0) %Q check is to check if code is working
    if(P_Store(nn-1)+distance*Em>mfp)   %Make sure we don't go up to our mfps
        P_Store(nn)=distance*Em+P_Store(nn-1); %part for moderator
        nn=nn+1; %for the fuel
        P_Store(nn)=P_Store(nn-1)+0*Ef;
        P=P_Store(nn);
        nn=nn+1; %new era of moderator
        distance=0; %Resetting distance
    end
else
    if(P_Store(nn-1)+distance*Ef>mfp)   %Make sure we don't go up to our mfps
        P_Store(nn)=distance*Em*0+P_Store(nn-1); %part for moderator
        nn=nn+1; %for the fuel
        P_Store(nn)=P_Store(nn-1)+distance*Ef;
        P=P_Store(nn);
        nn=nn+1; %new era of moderator
        distance=0; %Resetting distance 
    end
end
%For Debugging Purposes:
%disp('    Distance  Length       P_Store      P     mfp, theta');
%disp([distance,Length,P_Store(nn-1),P,distance*Em,acosd(Vector(1,4))]);




A=exist('distance_fuel','var');
B=exist('distance_before','var');
if(A && debug==1 && B)
    disp('    Fuel D    Before     After    Length');
    disp([distance_fuel,distance_before,distance_after,Length]);
    disp('  Distance      P1        P2        P3      P4');
    findingstuff=find(P_Store(2:end)==0,2);
    disp([distance,P_Store(2),P_Store(3:findingstuff(2))]);
elseif(debug==1)
    disp('  Distance      P1        P2        P3      P4');
    findingstuff=find(P_Store(2:end)==0,2);
    disp([distance,P_Store(2),P_Store(3:findingstuff(2))]);
end
