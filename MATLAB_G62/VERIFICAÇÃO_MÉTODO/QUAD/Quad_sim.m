
%IMPORT x, y, QUAD
x=x/1000;
y=y/1000;
%mm->m
%   Tarefa 39, 1a parte : seccao para desenhar a malha de quads-4
%--------------------------------------------------------------------------
%
Nels=size(QUAD,1);       % numero de elementos quad-4      
plot(x,y,'ro');hold
for i=1:Nels;
edofs=[QUAD(i,:) QUAD(i,1)]; %   conectividade deste quad
plot(x(edofs),y(edofs),'b');hold on
end

%   fim da visualizacao da malha
%-----------------------------------------------------
Nnds=size(x,1);

Kg=zeros(Nnds,Nnds);
fg=zeros(Nnds,1);
%   
for i=1:Nels   % ciclo para os elementos

edofs=[QUAD(i,:)]; %   conectividade deste quad
  %     coordenadas do elemento aqui
  XN(1:4,1)=x(edofs);
  XN(1:4,2)=y(edofs);
  %     calculos no elemento
fL= 0;
[Ke fe]=Elem_Quad4 (XN,fL);

  %     assemblagem
  Kg(edofs,edofs)= Kg(edofs,edofs) + Ke;  % 
  fg(edofs,1)= fg(edofs,1) + fe;          % 
 
end %for i

%u=Kg\fg;
Kr= Kg;
fr= fg;

%--------------------------------------------------------------------------
%
boom = 1.0e+14 ;     %   como se adicionasse uma mola de rigidez (quase) infinita
%

%Parede de cima
Kr(16,16)= boom;
fr(16) = 0;
Kr(11,11)= boom;
fr(11) = 0;
Kr(12,12)= boom;
fr(12) = 0;
Kr(15,15)= boom;
fr(15) = 0;


gama=-10;
index=[14 6 5 13]
	for i=1:4
        if (i~=4)
		h = sqrt((x(index(i+1))-x(index(i)))^2+(y(index(i+1))-y(index(i)))^2);
		fr(index(i))= fr(index(i)) + gama*h/2;   
    	fr(index(i+1))= fr(index(i+1)) + gama*h/2;
        end
		%if (i==1)
			%fr(index(i))= fr(index(i)) + gama*h/2;
		%end
		%if (i==7)
			%fr(index(i))= fr(index(i)) + gama*h/2;
		%end

	end

%----------------------------------------------------------
%   Tarefa 39 - seccao para desenhar a solucao aprox. a 3D
%----------------------------------------------------------
figure
%
Kr=sparse(Kr);
u=Kr\fr ;




for i=1:Nels;
edofs=[QUAD(i,:)];
fill3 (x(edofs),y(edofs),u(edofs),u(edofs));hold on
%plot(x(edofs),y(edofs),'r');hold on
end
plot(x,y,'ro');
%-----------------------------------------------------------

%-----------------------------------------------------------
%   Tarefa 40 - calcular (gradiente) fluxo nos centroides
%-----------------------------------------------------------
figure
for i=1:Nels;
edofs=[QUAD(i,:) QUAD(i,1)]; %   conectividade deste quad
plot(x(edofs),y(edofs),'k:');hold on
end
for i=1:Nels;
edofs=[QUAD(i,:)]; %   conectividade deste quad
  XN(1:4,1)=x(edofs);
  XN(1:4,2)=y(edofs);
%   O centroide esta na origem
csi=0;
eta=0;
%   para cada centroide, calcular
%----------------------------------------------------------------
[B psi Detj]=Shape_N_Der4 (XN,csi,eta);
%----------------------------------------------------------------
uint = psi'*u(edofs);
xpint = XN'*psi;    %   posicao (x,y) do centroide
gradu = B'*u(edofs);
fluxu = -gradu/100;
%fill (x(edofs),y(edofs),u(edofs));hold on
%plot(x(edofs),y(edofs),'k');hold on
%plot(xpint(1),xpint(2),'bx');hold on
%plot(x(edofs),y(edofs),'k:');hold on
quiver(xpint(1),xpint(2),fluxu(1),fluxu(2),'MaxHeadSize',0.8,'Color','r','LineWidth',1.1, 'AutoScale','off');hold on
end
plot(x,y,'ko');
