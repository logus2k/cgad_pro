
function [He Pe]=Robin_quadr (x1,y1,x2,y2,x3,y3,p,gama)
%------------- Inicializacoes --------------------------
b=zeros(3,1)
He=zeros(3,3)
Pe=zeros(3,1)
%-------------------------------------------------------
nip=3
[xi wi]=Genip1D (nip)   %   regras 1D Gauss-Legendre
%-------------------------------------------------------
%
for ip=1:nip
    csi=xi(ip)
    %   calcula funcoes de forma
    b(1)=0.5*csi*(csi-1)
    b(2)=1-csi*csi
    b(3)=0.5*csi*(csi+1)
        %   calcula derivadas das funcoes de forma
    db(1)= csi-0.5
    db(2)=-2*csi
    db(3)= csi+0.5
        %   calcula derivadas de x e de y
xx= db(1)*x1+db(2)*x2+db(3)*x3
yy= db(1)*y1+db(2)*y2+db(3)*y3
%-------------------------------------------------------
jaco = sqrt(xx^2+yy^2)      %   jacobiano
%-------------------------------------------------------
wip =jaco*wi(ip)
wipp =wip*p
wipg =wip*gama
%-------------------------------------------------------
He = He + wipp*b*b'
Pe = Pe + wipg*b
end
end

function [xi wi]=Genip1D (nip)
%-------------------------------------------------------
%   gera regras 1D de Gauss-Legendre
%-------------------------------------------------------
if (nip == 2)   % 2 pontos, grau 3
G=sqrt(1.0/3.0)
xi=[-G G];
wi=[1 ; 1];
end
if (nip == 3)   % 3 pontos, grau 5
G=sqrt(0.6)
xi=[-G 0 G];
wi=[5 ; 8; 5]/9;
end
end