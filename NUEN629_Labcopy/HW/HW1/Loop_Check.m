function [Flux]=Loop_Check(n,mfp,r,CHOICE)
theta=linspace(0,360,n);
Flux=zeros(1,n);

%You can pick which problem to work on
for i=1:n
   %Flux(1,i)=Problem_3(theta(i),[0.63,0.63],1,mfp);
   %Flux(1,i)=Problem_3(theta(i),[0.381051,0.66],2,mfp);
   %Flux(1,i)=Problem_3(theta(i),[0.41,0.41],1,mfp);
   %Flux(1,i)=Problem_3(theta(i),[0.63,0.41],1,mfp);
   %Flux(1,i)=Problem_3(theta(i),[0.57157676649,0.33],2,mfp);
   Flux(1,i)=Problem_3(theta(i),r,CHOICE,mfp);
end


hold off
plot(theta,Flux);
xlabel 'Angle (Degrees)'
ylabel 'Flux'
grid on
