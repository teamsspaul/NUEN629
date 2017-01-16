Es=1;X=1;q=1;

%Plot Solution
B=(exp(-1)+3)/(exp(-3/2)-9*exp(1/2));
x=linspace(-1.41,1.41);
phi=B.*exp(x)+B.*exp(-x)+1;
plot(x,phi,'r','LineWidth',2);

%Plot Bars
hold on
y=linspace(0,0.55);
xn=ones(1,100).*-0.5;xp=ones(1,100).*0.5;
plot(xn,y,'b','LineWidth',3);
plot(xp,y,'b','LineWidth',3);
grid on;xlabel 'X';ylabel '\phi';
legend ('\phi Marshak','Boundaries');
ylim([0,0.55]);





