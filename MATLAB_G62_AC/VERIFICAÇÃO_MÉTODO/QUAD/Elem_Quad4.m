%--------------------------------------------------------------------------
%	Tarefa 35 : criacao da funcao Elem_Quad4
%--------------------------------------------------------------------------
function [Ke fe]=Elem_Quad4 (XN,fL)
%--------------------------------------------------------------------------
%   Matriz XN(4,2) que contem as coordenadas (x,y) deste quad de 4 nos
%   fL - intensidade uniforme da força por unidade de area
%--------------------------------------------------------------------------
%   inicializar Ke e fe
    Ke = zeros(4,4);
    fe = zeros(4,1);
%--------------------------------------------------------------------------
%   gerar os pontos de integracao
%nip = 4;
nip = 9;
[xp wp]=Genip2DQ (nip);
%--------------------------------------------------------------------------
%   percorrer os pontos de integracao
for ip=1:nip;	%	ciclo para os pontos de integracao
%   para cada ponto
csi = xp(ip,1);
eta = xp(ip,2);
%   Obter :
%   1) funcoes de forma do quad-4, vector psi (4x1)
%   4) jacobiano da transformacao, Detj
%   6) a matriz B(4,2) das derivadas parciais das funcoes de 
%      forma em (x,y)
%---------------------------------------------------------------
[B psi Detj]=Shape_N_Der4 (XN,csi,eta);
%---------------------------------------------------------------
%   7) peso transformado
wip = wp(ip)*Detj;
%   8) calcular produto B*B', pesar e somar a Ke
Ke = Ke + wip*B*B' ;    %   matriz de rigidez
%   9) somar fe
wipf = fL*wip ;
fe = fe + wipf*psi ;    %   vector de forcas
%	proximo ponto de integracao
end	%   fim de ciclo
%--------------------------------------------------------------------------
end