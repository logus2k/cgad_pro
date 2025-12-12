function [B psi Detj]=Shape_N_Der8 (XN,csi,eta)
%----------------------------------------------------------------
%   Matriz XN(8,2) contem as coordenadas locais deste quad de 8 nos
%----------------------------------------------------------------
psi=zeros(8,1);

%   para cada ponto dado, calcular

%   1) funcoes de forma do quad-8, vector psi (8x1)
psi(1) = (csi-1)*(eta+csi+1)*(1-eta)/4;
psi(2) = (1+csi)*(1-eta)*(csi-eta-1)/4;
psi(3) = (1+csi)*(1+eta)*(csi+eta-1)/4;
psi(4) = (csi-1)*(csi-eta+1)*(1+eta)/4;
psi(5) = (1-csi*csi)*(1-eta)/2;
psi(6) = (1+csi)*(1-eta*eta)/2;
psi(7) = (1-csi*csi)*(1+eta)/2;
psi(8) = (1-csi)*(1-eta*eta)/2;

%   2) derivadas parciais em (csi,eta), Matriz Dpsi(8x2)
Dpsi(1,1) = (2*csi+eta)*(1-eta)/4;
Dpsi(2,1) = (2*csi-eta)*(1-eta)/4;
Dpsi(3,1) = (2*csi+eta)*(1+eta)/4;
Dpsi(4,1) = (2*csi-eta)*(1+eta)/4;
Dpsi(5,1) = csi*(eta-1);
Dpsi(6,1) = (1-eta*eta)/2;
Dpsi(7,1) = -csi*(1+eta);
Dpsi(8,1) = (eta*eta-1)/2;
%
Dpsi(1,2) = (2*eta+csi)*(1-csi)/4;
Dpsi(2,2) = (2*eta-csi)*(1+csi)/4;
Dpsi(3,2) = (2*eta+csi)*(1+csi)/4;
Dpsi(4,2) = (2*eta-csi)*(1-csi)/4;
Dpsi(5,2) = (csi*csi-1)/2;
Dpsi(6,2) = -(1+csi)*eta;
Dpsi(7,2) = (1-csi*csi)/2;
Dpsi(8,2) = (csi-1)*eta;

%   3) derivadas parciais da matriz jacobiana (2x2) de x e y
jaco = XN'*Dpsi;
%   4) jacobiano da transformacao
Detj = det(jaco);
%   5) derivadas parciais da transformacao inversa (csi,eta) em funcao de
%     (x,y), matriz (2x2)
Invj = inv(jaco);
%   6) finalmente a matriz B (8x2) das derivadas parciais das funcoes de 
%      forma em (x,y)
B = Dpsi*Invj;
%----------------------------------------------------------------
end