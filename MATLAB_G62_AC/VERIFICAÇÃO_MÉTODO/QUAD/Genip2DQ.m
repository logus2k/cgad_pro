%----------------------------------------------------------------
%	Tarefa 33 : criacao da funcao Genip2DQ
%----------------------------------------------------------------
function [xp wp]=Genip2DQ (nip)
%----------------------------------------------------------------
%   gera regras 2D de produto de Gauss-Legendre 1D
%----------------------------------------------------------------
if (nip == 4)   % 2x2 = 4 pontos, grau 3
G=sqrt(1.0/3.0);
xp=[-G G G -G;-G -G G G]' ;
wp=[1 1 1 1]';
end
if (nip == 9)   % 3x3 = 9 pontos, grau 5
G=sqrt(0.6);
xp=[-G 0 G -G 0 G -G 0 G;-G -G -G 0 0 0 G G G]' ;
wp=[25 40 25 40 64 40 25 40 25]'/81;
end
end