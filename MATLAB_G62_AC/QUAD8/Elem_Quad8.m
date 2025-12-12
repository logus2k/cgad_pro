function [Ke fe]=Elem_Quad8 (XN,fL);
%   Matriz XN(8,2) contem as coordenadas locais deste quad de 8 nos 
%   inicializar Ke e fe
 Ke = zeros(8,8);
 fe = zeros(8,1);
 psi = zeros(8,1);
%   gerar pontos de integracao
nip = 9;
[xp wp]=Genip2DQ (nip);

%   percorrer os pontos de integracao
for ip=1:nip;

csi = xp(ip,1);
eta = xp(ip,2);
%   para cada ponto calcular
%-------------------------------------------------------
[B psi Detj]=Shape_N_Der8 (XN,csi,eta);
%-------------------------------------------------------
%   4) peso transformado
wip = wp(ip)*Detj;

%   7) calcular produto B*B', pesar e somar a Ke
Ke = Ke + wip*B*B';
%   8) somar fe
wipf = fL*wip;
fe = fe + wipf*psi;
%   fim de ciclo
end
end