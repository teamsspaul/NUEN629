D=1;a=1;q=1;alpha=0.999;
%Boundary Conditions:
CL=0.5;CR=0;BL=0;BR=1;AL=1;AR=(1-alpha)/(2*(1+alpha));

%Plot Solution
C1=((AL/AR)*(CR+q*a*BR)-BL*q*a-CL)/(a*AL+(AL*BR*D)/AR+BL*D);
C2=CR/AR+((q*a*BR)/AR+(q*(a^2))/(D*4))-C1*(a/2+(BR*D)/AR);

x=linspace(-0.8,0.8);
phi=(-q./D).*(x.^2)+C1.*(x)+C2;
plot(x,phi,'r','LineWidth',2);

%Plot Bars
hold on
y=linspace(0,1.7);
xn=ones(1,100).*-0.5;xp=ones(1,100).*0.5;
plot(xn,y,'b','LineWidth',3);
plot(xp,y,'b','LineWidth',3);
grid on;xlabel 'X';ylabel '\phi';
legend ('\phi Dirichlet Left Reflecting Right','Boundaries');
ylim([0,0.8]);





