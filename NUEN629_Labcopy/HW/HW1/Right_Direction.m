function [XYT]=Right_Direction(XYT,Vector,x2,y2,j,limits,Lines)

tol=0.000000001;CHECK=1;

%Find the distance you need to go
distance=((Vector(1,2)-x2)^2+(Vector(1,1)-y2)^2)^0.5;

if (distance>tol)
    %If you are within the bounds of habitiation, This is all you need
    %after the first step, but if you do not start on the edge, you
    %need to check normals to make sure you are going in the right
    %direction.
    if(y2+tol<limits(j,1)||y2>limits(j,2)+tol||x2+tol<limits(j,3)||x2>limits(j,4)+tol)
        CHECK=0;
    end
    if(CHECK==1)%Check Normals and directions
        CHECK2(1:2)=Vector(1,1:2);
        DOS=Reflection(1,Vector,Lines(j,:),90,0); %will update
    else
        DOS=1;CHECK2=1;
    end
    if(CHECK==1 && (CHECK2(1)~=DOS(1,1)||CHECK2(2)~=DOS(1,2))) %Update worthy Values
         XYT(j,2)=x2;
         XYT(j,1)=y2;
         XYT(j,3)=distance;
    end
end

%For Debugging purposes
%disp('       x2        x1     xmin       xmax       y2         y1       ymin     ymax')
%disp([x2,Vector(1,2),limits(j,3),limits(j,4),y2,Vector(1,1),limits(j,1),limits(j,2)]);
%disp('    Line     Slope       Dist_cal  CHECK(1=keep)  CHECK2/=DOS(1,1)');
%disp([j,Vector(1,3),distance,CHECK,CHECK2,DOS(1,1)]);
