%--------------------------------------------------------------------------
%	Tarefa 34 : criacao da funcao Shape_N_Der4
%--------------------------------------------------------------------------
function [B psi Detj]=Shape_N_Der4 (XN,csi,eta)
%--------------------------------------------------------------------------
%   Entrada : Matriz XN(4,2) define as coordenadas (x,y) deste quad-4
%   Entrada : (csi,eta) - coordenadas standard do ponto dado
%--------------------------------------------------------------------------
%   Saida : Matriz B(4,2) das derivadas parciais das 4 funcoes de forma
%           em ordem a x (1a coluna) e a y (2a coluna) neste quad-4
%   Saida : Vector coluna psi(4,1) das 4 funcoes de forma
%   Saida : Detj - Jacobiano, determinante da matriz Jacobiana
%--------------------------------------------------------------------------
psi=zeros(4,1);
%   para cada ponto dado, calcular
%   1) funcoes de forma do quad-4, vector psi (4x1)
psi(1) = (1-csi)*(1-eta)/4;
psi(2) = (1+csi)*(1-eta)/4;
psi(3) = (1+csi)*(1+eta)/4;
psi(4) = (1-csi)*(1+eta)/4;
%   2) derivadas parciais em (csi,eta), Matriz Dpsi(4x2)
Dpsi(1,1) = (eta-1)/4;
Dpsi(2,1) = (1-eta)/4;
Dpsi(3,1) = (1+eta)/4;
Dpsi(4,1) = -(1+eta)/4;
%
Dpsi(1,2) = (csi-1)/4;
Dpsi(2,2) = -(1+csi)/4;
Dpsi(3,2) = (1+csi)/4;
Dpsi(4,2) = (1-csi)/4;
%--------------------------------------------------------------------------
%   3) matriz jacobiana (2x2), das derivadas parciais de x e y
jaco = XN'*Dpsi ;
%   4) jacobiano da transformacao, um escalar com sinal
Detj = det(jaco) ;
%   5) derivadas parciais da transformacao inversa (csi,eta) em funcao de
%     (x,y), Invj, uma matriz (2x2)
Invj = inv(jaco) ;
%   6) finalmente a matriz B (4x2) das derivadas parciais das funcoes de 
%      forma em (x,y)
B = Dpsi*Invj ;
%--------------------------------------------------------------------------
end