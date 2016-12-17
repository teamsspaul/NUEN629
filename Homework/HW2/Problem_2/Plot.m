A =[4.60248-2.91518   0            0            0            0
    -1.67039      18.2957-14.0197   0            0            0
    -0.0167099    -4.23255    20.4196-16.0183     0            0
    -0.00166937   -0.0423257     -4.35086     22.6605-4.53876 -1.53831
    -1.04698e-6   -0.000424355   -0.0439445     -17.9734     29.77991-27.9344];



A=A*0.000751;

b=[0.99136;0.01379;0;0;0];

flux=A^-1*b

x1=logspace(5,7.3011,30);
x2=logspace(3,5,30);
x3=logspace(1,3,30);
x4=logspace(-1,1,30);
x5=logspace(-3,-1,30);

semilogx(x1,flux(1,1)*ones(1,30),'b','LineWidth',3);
hold on; grid on;xlabel 'Energy (eV)';ylabel '\phi';
semilogx(x2,flux(2,1)*ones(1,30),'b','LineWidth',3);
semilogx(x3,flux(3,1)*ones(1,30),'b','LineWidth',3);
semilogx(x4,flux(4,1)*ones(1,30),'b','LineWidth',3);
semilogx(x5,flux(5,1)*ones(1,30),'b','LineWidth',3);

%Below Zero?
semilogx(x1,zeros(1,30),'r','LineWidth',3);
hold on; grid on;xlabel 'Energy (eV)';ylabel '\phi';
semilogx(x2,zeros(1,30),'r','LineWidth',3);
semilogx(x3,zeros(1,30),'r','LineWidth',3);
semilogx(x4,zeros(1,30),'r','LineWidth',3);
semilogx(x5,zeros(1,30),'r','LineWidth',3);


