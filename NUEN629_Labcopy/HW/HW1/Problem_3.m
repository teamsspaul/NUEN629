function [Flux]=Problem_3(theta,r,CHOICE,mfp)
%This program attempts to solve problem 3 of homework 1
%of NUEN 629 Fall 2015, for details refer to the problem statement
format short
format compact
%Units cm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Lattice (Note: Hexigons should have flat faces with constant y values)
%Note Make sure x1 and y1 values are centered
%Form of input: [y1,x1,m,sin(b),cos(b)]

%Central Radius
R=0.41;
%Hexagon 1/2 pitch
RH=0.66;



%CHOICE=2;
if CHOICE==1
    %Square Lattice:
    Lines(1,1:5)=[0,0.63,nan,1,0]; %Positive X Line
    Lines(2,1:5)=[0,-0.63,nan,1,0]; %Negative X line
    Lines(3,1:5)=[0.63,0,0,0,1]; %Positive Y line
    Lines(4,1:5)=[-0.63,0,0,0,1];  %Negative Y line
else %Hex Lattice:
    Lines(1,1:5)=[0.66,0,0,0,1]; %Positive Y line
    Lines(2,1:5)=[-0.66,0,0,0,1];  %Negative Y line 
    Lines(3,1:5)=[sind(30)*RH,cosd(30)*RH,tand(120),sind(120),cosd(120)]; %Right Side Negative Slope
    Lines(4,1:5)=[-sind(30)*RH,cosd(30)*RH,tand(60),sind(60),cosd(60)]; %Right Side Positive Slope
    Lines(5,1:5)=[-sind(30)*RH,-cosd(30)*RH,tand(120),sind(120),cosd(120)]; %Left Side Negative Slope
    Lines(6,1:5)=[sind(30)*RH,-cosd(30)*RH,tand(60),sind(60),cosd(60)];  %Left Side Positive Slope
end

Em=0.08;Ef=0.1414;I=1/(4*pi*Ef); %Values for Flux Determination
Q_CHECK=0;
%In Order to Check if my solution is working, need a flat profile
%Q_CHECK=1;Em=0.1414;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Find Length of Edges
if (any(Lines(:,3)))
   Lengths=(RH*2)/(3^0.5); 
   Lengthx=Lengths*cosd(60);
   Pitch=2*RH;
   Min_dis=(Pitch/2)/sind(60)-R;
else
   Lengths=max(Lines(:,2))-min(Lines(:,2));
   Pitch=Lengths;
   Lengthx=1;
   Min_dis=(Pitch/(2^0.5))-R;
end

Rows=size(Lines,1); %Used for Looping
limits=zeros(Rows,4); %Limits for lines [ymin,ymax,xmin,xmax]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Plot Boundaries to visually observe %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xplot=zeros(1,100);
yplot=xplot;

for j=1:Rows %Loop over all Boundary Lines
    slope=1;
    if(Lines(j,3)<0) %Check for negative Slopes
       slope=-1; 
    end
        if(Lines(j,3)==0) %m=0, sin(b)=0, cos(b)=1
            yplot(1,:)=Lines(j,1); %y is constant
            xplot(1,:)=linspace(Lines(j,2)-Lengths/2,Lines(j,2)+Lengths/2);
            limits(j,1)=Lines(j,1);
            limits(j,2)=Lines(j,1);
            limits(j,3)=Lines(j,2)-Lengths/2;
            limits(j,4)=Lines(j,2)+Lengths/2;
        elseif(isnan(Lines(j,3))) %m=INF,sin(b)=1,cos(b)=0
            xplot(1,:)=Lines(j,2); %x is constant
            yplot(1,:)=linspace(Lines(j,1)-Lengths/2,Lines(j,1)+Lengths/2);
            limits(j,1)=Lines(j,1)-Lengths/2;
            limits(j,2)=Lines(j,1)+Lengths/2;
            limits(j,3)=Lines(j,2);
            limits(j,4)=Lines(j,2);
        else %m is nonzero and non inf
            xplot(1,:)=linspace(Lines(j,2)-Lengthx/2,Lines(j,2)+Lengthx/2);
            yplot(1,:)=Lines(j,3).*(xplot-Lines(j,2))+Lines(j,1);
            limits(j,1)=slope.*Lines(j,3)*(xplot(1,1)-Lines(j,2))+Lines(j,1);
            limits(j,2)=slope.*Lines(j,3)*(xplot(end)-Lines(j,2))+Lines(j,1);
            limits(j,3)=Lines(j,2)-Lengthx/2;
            limits(j,4)=Lines(j,2)+Lengthx/2;
        end
    plot(xplot,yplot,'b','LineWidth',3);
    hold on
end

%Plot The Circle
xplot=linspace(-R,R,50);
yplot=(R.^2-xplot.^2).^0.5;
plot(xplot,-yplot,'b','LineWidth',3);
hold on
plot(xplot,yplot,'b','LineWidth',3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculations  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Direction and Magnitude This variable holds our location and direction
Vector(1,1:5)=[r(2),r(1),tand(theta),sind(theta),cosd(theta)];
Vector(Vector==inf)=nan;
Vector(Vector==-inf)=nan;

P=0;P_Store=zeros(1,2000);distance=0;nn=2;
while(P<mfp) %Loop until we have answers...
%For debugging purposes
%for kk=1:100

%If heading outside the system Turn Around: Do twice, just incase you need
%two translations
Vector=Reflection(Rows,Vector,Lines,Pitch,1);
Vector=Reflection(Rows,Vector,Lines,Pitch,1);

%%d will be variable for length till change medium

%Translation Matrix (I am sure there is a better way to do this)
XYT=zeros(Rows+1,3); %Values saved [Ytranslated,Xtranslated,Distance]
%The Plus 1 is for the circle consideration

%Lets go for one interaction to the next Boundary Line:
for j=1:Rows %Loop over all Boundary Lines
        %If we have a sloped line (This should include horizontal lines)
        if(Lines(j,5)~=0 && Vector(1,5)~=0 && Lines(j,3)~=Vector(1,3))
            %There is an intersection
            x2=(Lines(j,3)*Lines(j,2)-Vector(1,3)*Vector(1,2)+Vector(1,1)-Lines(j,1))/(Lines(j,3)-Vector(1,3));
            y2=Vector(1,3)*(x2-Vector(1,2))+Vector(1,1);
            %Check if intersection is in the right direction
            XYT=Right_Direction(XYT,Vector,x2,y2,j,limits,Lines);
        end
        %If my vector is traveling up and down
        if(Lines(j,5)~=0 && Vector(1,5)==0)
            x2=Vector(1,2);
            y2=Lines(j,3)*(x2-Lines(j,2))+Lines(j,1);
            XYT=Right_Direction(XYT,Vector,x2,y2,j,limits,Lines);
        end
        %If my intersecting line is up and down
        if(Lines(j,5)==0 && Vector(1,5)~=0)
            x2=Lines(j,2);
            y2=Vector(1,3)*(x2-Vector(1,2))+Vector(1,1);
            XYT=Right_Direction(XYT,Vector,x2,y2,j,limits,Lines);
        end
end

if(any(XYT)) %If any elements of XYT are non zero
    A=XYT(:,3);
    n=find(A==min(A(A>0)));
    xt=XYT(n(1),2);
    yt=XYT(n(1),1);
    Length=XYT(n(1),3); %this is the track length
end

%%Checking If My tracing is done correctly
if(isnan(Vector(1,3))) %If we have an infinite slope
    xplot=ones(1,100).*Vector(1,2);
    yplot=linspace(Vector(1,1),Vector(1,1)+Vector(1,4)*Length,100);
else
    xplot=linspace(Vector(1,2),Vector(1,2)+Length*Vector(1,5),100);
    yplot=Vector(1,3).*(xplot-Vector(1,2))+Vector(1,1);
end
plot(xplot,yplot,'r','LineWidth',3);
hold on;

%Calculations for distances
[distance,P,P_Store,nn]=Distances(Vector,R,xt,yt,Em,Ef,distance,P,P_Store,nn,Length,mfp,Min_dis,Q_CHECK);


Vector(1,1)=yt; %New Location for Vector
Vector(1,2)=xt;
%Plot Our Vector for its path
Length=0.3;
if(isnan(Vector(1,3))) %If we have an infinite slope
    xplot=ones(1,100)*Vector(1,2);
    yplot=linspace(Vector(1,1),Vector(1,1)+Vector(1,4)*Length);
else
    xplot=linspace(Vector(1,2),Vector(1,2)+Length*Vector(1,5));
    yplot=Vector(1,3).*(xplot-Vector(1,2))+Vector(1,1);
end
plot(xplot,yplot,'g','LineWidth',3);
hold on
grid on

end

sum=0;

for L=2:nn-1
    sum=sum+((-1)^L)*exp(-P_Store(L));
end
Flux=I*sum;




