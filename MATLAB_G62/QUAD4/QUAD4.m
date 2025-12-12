x=xlsread('quad_data_mesh.xlsx','cord','A1:A182')
x=x/1000; %mm to m
y=xlsread('quad_data_mesh.xlsx','cord','B1:B182')
y=y/1000; %mm to m
QUAD=xlsread('quad_data_mesh.xlsx','conec','A1:D142')

%IMPORT x, y, QUAD
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
Kr(30,30)= boom;
fr(30) = 0;
Kr(31,31)= boom;
fr(31) = 0;
Kr(27,27)= boom;
fr(27) = 0;

%Parede de baixo
Kr(182,182)= boom;
fr(182) = 0;
Kr(181,181)= boom;
fr(181) = 0;
Kr(180,180)= boom;
fr(180) = 0;

gama=2.5;
index=[14 5 6 13 102 103 108]
	for i=1:7
        if (i~=7)
		h = sqrt((x(index(i+1))-x(index(i)))^2+(y(index(i+1))-y(index(i)))^2);
		fr(index(i))= fr(index(i)) + gama*h/2;   
    	fr(index(i+1))= fr(index(i+1)) + gama*h/2;
        end
		
	end

%----------------------------------------------------------
%   Tarefa 39 - seccao para desenhar a solucao aprox. a 3D
%----------------------------------------------------------
figure
%
Kr=sparse(Kr);
u=Kr\fr ;



figure
for i=1:Nels;
edofs=[QUAD(i,:)];
fill (x(edofs),y(edofs),u(edofs));hold on
plot(x(edofs),y(edofs),'r');hold on
end

figure
for i=1:Nels;
edofs=[QUAD(i,:)];
fill3 (x(edofs),y(edofs),u(edofs),u(edofs));hold on
plot(x(edofs),y(edofs),'r');hold on
end

%-----------------------------------------------------------

%-----------------------------------------------------------
%   Tarefa 40 - calcular (gradiente) fluxo nos centroides
%-----------------------------------------------------------
figure
for i=1:Nels;
edofs=[QUAD(i,:) QUAD(i,1)]; %   conectividade deste quad
plot(x(edofs),y(edofs),'k:');hold on
end
abs_vel_nds=zeros(1,Nnds);
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
vel(i,1)=gradu(1);
vel(i,2)=gradu(2);
abs_vel(i)=sqrt((vel(i,1))^2+(vel(i,2))^2);
fluxu = -gradu/35;
vel(i,1)=gradu(1);
vel(i,2)=gradu(2);
abs_vel(i)=sqrt((vel(i,1))^2+(vel(i,2))^2)
abs_vel_nds(edofs)=abs_vel(i);
pressure(i)=101328.8281-0.6125*(abs_vel(i))^2;
xc(i)=(x(edofs(1))+x(edofs(2))+x(edofs(3))+x(edofs(4)))/4;
yc(i)=(y(edofs(1))+y(edofs(2))+y(edofs(3))+y(edofs(4)))/4;
%fill (xc(i),yc(i),abs_vel(i));hold on
%plot(x(edofs),y(edofs),'k');hold on
%plot(xpint(1),xpint(2),'bx');hold on
plot(x(edofs),y(edofs),'k:');hold on
quiver(xpint(1),xpint(2),fluxu(1),fluxu(2),'MaxHeadSize',0.8,'Color','r','LineWidth',1.1, 'AutoScale','off');
end
%plot3 (xc,yc,abs_vel);hold on
%plot(x,y,'ko');

figure
%plot(x,y,'ro');
for i=1:Nels;
edofs=[QUAD(i,:)];
fill (x(edofs),y(edofs),abs_vel_nds(edofs));hold on
plot(x(edofs),y(edofs),'k:');hold on
end


for i=1:Nnds
    %index=parede(i);
    p(i)=101328.8281-0.6125*(abs_vel_nds(i))^2;
end


F1=0;
F2=0;
F3=0;
F4=0;
F5=0;
F6=0;

parede1=[14 10 9 16 69 94];
for i=1:5
    L=sqrt((x(parede1(i))-x(parede1(i+1)))^2+(y(parede1(i))-y(parede1(i+1)))^2);
    F1=F1+L*(101325-(p(parede1(i))+p(parede1(i+1)))/2);
end

parede2=[94 77 76 75 74 73];
for i=1:5
    L=sqrt((x(parede2(i))-x(parede2(i+1)))^2+(y(parede2(i))-y(parede2(i+1)))^2);
    F2=F2+L*(101325-(p(parede2(i))+p(parede2(i+1)))/2);
end

parede3=[74 73 72 71 70 95];
for i=1:5
    L=sqrt((x(parede3(i))-x(parede3(i+1)))^2+(y(parede3(i))-y(parede3(i+1)))^2);
    F3=F3+L*(101325-(p(parede3(i))+p(parede3(i+1)))/2);
end

parede4=[95 92 91 90 89 88 28 23 22 21 30];
for i=1:10
    L=sqrt((x(parede4(i))-x(parede4(i+1)))^2+(y(parede4(i))-y(parede4(i+1)))^2);
    F4=F4+L*(101325-(p(parede4(i))+p(parede4(i+1)))/2);
end

parede5=[96 85 86 87 97];
for i=1:4
    L=sqrt((x(parede5(i))-x(parede5(i+1)))^2+(y(parede5(i))-y(parede5(i+1)))^2);
    F5=F5+L*(101325-(p(parede5(i))+p(parede5(i+1)))/2);
end

parede6=[97 93 29 24 25 26 31];
for i=1:6
    L=sqrt((x(parede6(i))-x(parede6(i+1)))^2+(y(parede6(i))-y(parede6(i+1)))^2);
    F6=F6+L*(101325-(p(parede6(i))+p(parede6(i+1)))/2);
end

F=[F1 F2 F3 F4 F5 F6]';


format longG
Pmax=max(p);
Pmin=min(p);
Umax=max(abs_vel_nds);
Umin=min(abs_vel_nds);
POTmax=max(u);

% Para que os rsultados sejam impressos no ficheiro excel desejado, tem de
% se alterar a diretoria de onde se encontra agora o excel
nos_num=[1:Nnds,1]';
%xlswrite('.\QUAD4\Results_quad4.xlsx',nos_num,1,'A2')
%xlswrite('.\QUAD4\Results_quad4.xlsx',u,1,'B2')

ele_num=[1:Nels,1]';
%xlswrite('.\QUAD4\Results_quad4.xlsx',ele_num,1,'D2')
%xlswrite('.\QUAD4\Results_quad4.xlsx',vel(:,1),1,'E2')
%xlswrite('.\QUAD4\Results_quad4.xlsx',vel(:,2),1,'F2')
%xlswrite('.\QUAD4\Results_quad4.xlsx',abs_vel',1,'G2')
%xlswrite('.\QUAD4\Results_quad4.xlsx',pressure',1,'H2')
%xlswrite('.\QUAD4\Results_quad4.xlsx',POTmax',1,'K2')
%xlswrite('.\QUAD4\Results_quad4.xlsx',Umax',1,'K3')
%xlswrite('.\QUAD4\Results_quad4.xlsx',Umin',1,'K4')
%xlswrite('.\QUAD4\Results_quad4.xlsx',Pmax',1,'K5')
%xlswrite('.\QUAD4\Results_quad4.xlsx',Pmin',1,'K6')
%xlswrite('.\QUAD4\Results_quad4.xlsx',F,1,'K10')