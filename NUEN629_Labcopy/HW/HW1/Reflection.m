function [Vector]=Reflection(Rows,Vector,Lines,Pitch,LT)
%This function will reflect across a cell given input values

%Translation Matrix (I am sure there is a better way to do this)
XYT=zeros(Rows,3); %Values saved [Ytranslated,Xtranslated,Angle]

tol=0.0001;
%Looping over all our lines
for j=1:Rows
%We need an inversion in logic: One for reflection, another for
%Right_Direction
if(LT==1)
    TF=abs(Lines(j,5)*(Vector(1,1)-Lines(j,1))-Lines(j,4)*(Vector(1,2)-Lines(j,2)))<tol;
elseif(LT==0)
    TF=abs(Lines(j,5)*(Vector(1,1)-Lines(j,1))-Lines(j,4)*(Vector(1,2)-Lines(j,2)))>tol;
end

%For Debugging Purposes:
%Value=abs(Lines(j,5)*(Vector(1,1)-Lines(j,1))-Lines(j,4)*(Vector(1,2)-Lines(j,2)));
%disp('     Cos_L(x) Sin_L(y)   Lx           Ly      Vx     Vy             Val');
%disp([Lines(j,5),Lines(j,4),Lines(j,2),Lines(j,1),Vector(1,2),Vector(1,1),Value]);

    %Check if on line (or check if not on line in the case of a
    %Right_Direction
    if(TF)
        %Find Normal to the line:
        XN=-1*Lines(j,4);
        YN=Lines(j,5);
        %Find the inward normal
        d1=((Lines(j,2)+XN)^2+(Lines(j,1)+YN)^2)^0.5;
        d2=((Lines(j,2)-XN)^2+(Lines(j,1)-YN)^2)^0.5;
        
        %For Debugging Purposes
        %disp('      j        LXN         VXN        d1     d2       LX          LY');
        %disp([j,XN,YN,d1,d2,Lines(j,2),Lines(j,1)]);
        
        if(d2<d1)
            XN=-XN;
            YN=-YN;
        end
        %Check if we are going the right way (dot product should be less
        %than 90. It it is greater, then we store a matrix which holds
        %reflection coordinates. The reason we have multiple reflection
        %coordinates is because we might be on top of two lines
        
        dot=acosd(XN*Vector(1,5)+YN*Vector(1,4));
        
        %For Debugging Purposes
        %disp('      j        LXN       VXN          LYN     VYN        dot');
        %disp([j,XN,Vector(1,5),YN,Vector(1,4),dot]);
        
        
        
    if(LT==1)
        TF2=dot>90;
    elseif(LT==0)
        TF2=dot>90;
    end

        if(TF2)
            XYT(j,2)=Vector(1,2)+XN*Pitch;
            XYT(j,1)=Vector(1,1)+YN*Pitch;
            XYT(j,3)=acosd(XN*Vector(1,5)+YN*Vector(1,4));
        end
        
    end
end

if(any(XYT)) %If any elements of XYT are non zero
    n=find(XYT(:,3)==max(XYT(:,3))); %Find the line of the max angle
    Vector(1,2)=XYT(n(1),2);
    Vector(1,1)=XYT(n(1),1);
end

%For Debugging purposes
%disp('XYT...Did we find anything?');
%disp([XYT]);


