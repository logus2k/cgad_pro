
%----------------------------------------------------------------------
%	Tarefa 53 : Elemento quadrático triangular de 6-nós
%----------------------------------------------------------------------
function [Ke fe]=Elem_TRI6 (XN,fL)
%   Matriz XN(6,2) contem as coordenadas locais deste triangulo de 6 nos 
%   inicializar Ke e fe
 Ke = zeros(6,6) ;
 fe = zeros(6,1) ;
%   gerar pontos de integracao
nip = 7 ;
[xp wp]=GenipT (nip) ;

%   percorrer os pontos de integracao
for ip=1:nip
    
csi = xp(ip,1) ;
eta = xp(ip,2) ;
%----------------------------------------------------------------
[B psi Detj]=Shape_N_Der6 (XN,csi,eta) ;
%----------------------------------------------------------------
%   5) peso transformado
wip = wp(ip)*Detj ;
%   6) ponderacao da carga no elemento
wipf = fL*wip ;
%   7) calcular e acumular fe, vector (6x1)
fe = fe + wipf*psi ;
%   10) calcular produto B*B' (6x6), pesar e somar a Ke
Ke = Ke + wip*B*B' ;
%
end     %   fim de ciclo de integracao
end     %   fim de funcao
