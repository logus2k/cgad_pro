%--------------------------------------------------------------------------
%	Tarefa 52 : Funcoes de Forma, Derivadas e Jacobiano para elementos
%   quadraticos triangulares de 6-nós
%--------------------------------------------------------------------------
function [B psi Detj]=Shape_N_Der6 (XN,csi,eta)
%----------------------------------------------------------------
%   Matriz XN(6,2) contem as coordenadas locais deste triangulo de 6 nos
%----------------------------------------------------------------
psi=zeros(6,1) ;
%   para cada ponto dado, calcular
%   1) funcoes de forma do tri-6, vector psi (6x1)
psi(1) = (1-csi-eta)*(1-2*csi-2*eta);
psi(2) = csi*(2*csi-1);
psi(3) = eta*(2*eta-1);
psi(4) = 4*(1-csi-eta)*csi;
psi(5) = 4*csi*eta;
psi(6) = 4*(1-csi-eta)*eta;

%   2) derivadas parciais em (csi,eta), Matriz Dpsi(6x2)
Dpsi(1,1) = 4*csi+4*eta-3;
Dpsi(2,1) = 4*csi-1;
Dpsi(3,1) = 0;
Dpsi(4,1) = 4 -8*csi -4*eta;
Dpsi(5,1) = 4*eta;
Dpsi(6,1) = -4*eta;
%
Dpsi(1,2) = 4*csi+4*eta-3;
Dpsi(2,2) = 0;
Dpsi(3,2) = 4*eta-1;
Dpsi(4,2) = -4*csi;
Dpsi(5,2) = 4*csi;
Dpsi(6,2) = 4 -4*csi -8*eta;
%   3) derivadas parciais da matriz jacobiana (2x2) de x e y
jaco = XN'*Dpsi ;
%   4) jacobiano da transformacao
Detj = det(jaco) ;
%   5) derivadas parciais da transformacao inversa (csi,eta) em funcao de
%     (x,y), matriz (2x2)
Invj = inv(jaco) ;
%   6) finalmente a matriz B (6x2) das derivadas parciais das funcoes de 
%      forma em (x,y)
B = Dpsi*Invj ;
%----------------------------------------------------------------
end