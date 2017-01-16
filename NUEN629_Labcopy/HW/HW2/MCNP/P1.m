n=4000;
E=logspace(-12,3,n);
P=zeros(1,n);PP=P;
sum=0;sum2=0;
Ef=1.35;E0=2.53000e-8;
for i=1:n
    P(i)=0.4865*sinh((2*E(i))^0.5)*exp(-1.*E(i)); %Fission Spectrum 1
    PP(i)=(((E(i)/Ef)^0.5)/(Ef))*exp(-E(i)/Ef);   %Fission Spectrum 2
    if i~=1
        sum=sum+P(i)*(E(i)-E(i-1));    %Sum 2 is smaller
        sum2=sum2+PP(i)*(E(i)-E(i-1)); %Sum 2 is smaller
    end
end

semilogx(E,P,'r','LineWidth',1.8)
hold on
%semilogx(E,PP)
grid on

hold on

loglog(x(:,1),x(:,2),'b','LineWidth',1.8)
xlabel ('Energy (MeV)');
ylabel ('\phi(E) (MeV^{-1})');
axis([10e-10 10E1 10e-14 10])


legend('MCNP Output No C12','Fission Spectrum')