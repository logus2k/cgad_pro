x=xlsread('mesh_data_quad8.xlsx','coord','A1:A201')
x=x/1000; %mm to m
y=xlsread('mesh_data_quad8.xlsx','coord','B1:B201')
y=y/1000; %mm to m
quad8=xlsread('mesh_data_quad8.xlsx','conec','A1:H52')




figure(1);
Nels=size(quad8,1);

for i=1:Nels;
    no1=quad8(i,1);
    no2=quad8(i,2);
    no3=quad8(i,3);
    no4=quad8(i,4);
    no5=quad8(i,5);
    no6=quad8(i,6);
    no7=quad8(i,7);
    no8=quad8(i,8);
edofs=[no1 no5 no2 no6 no3 no7 no4 no8]; % ordem para desenhar

plot(x(edofs),y(edofs),'b');hold on
end
plot(x,y,'ro');

Nels=size(quad8,1);       % numero de elementos quads

Nnds =size(x,1);        % numero de nos

%   inicializacao a zeros
Kg=zeros(Nnds,Nnds);
fg=zeros(Nnds,1);

%--------------------------------------------------------------
%   Tarefa 80 : Assemblagem dos elementos Quad-8 
%--------------------------------------------------------------
for i=1:Nels;

edofs=[quad8(i,:)]; %   conectividade deste quad
  XN(1:8,1)=x(edofs);
  XN(1:8,2)=y(edofs);
  %     calculos no elemento
fL= 0;
[Ke, fe]=Elem_Quad8 (XN,fL);

  %     assemblagem
  Kg(edofs,edofs)= Kg(edofs,edofs) + Ke;  % 
  fg(edofs,1)= fg(edofs,1) + fe;          % 
end %for i

%--------------------------------------------------------------
%   Tarefa 81 : Condicoes de fronteira e solucao do sistema  
%--------------------------------------------------------------
boom=1.0e+12;
kount=0;
entry_nodes=[19 7 6 8 18 115 116 117 124];
exit_nodes=[33 30 34 201 199 200];

for i=1:6;
    j=exit_nodes(i);
    Kg(j,j) = boom;
    fg(j,1)= boom*0;
    
end

%--------------------------------------------------------------------------
%   Tarefa 62 : Assemblagem de H e P, condicoes de Robin, lados curvos
%--------------------------------------------------------------------------
% Atencao: so os lados quadraticos sobre a fronteira


p=0
gama=2.5
[He Pe]=Robin_quadr(x(19),y(19),x(7),y(7),x(6),y(6),p,gama)
edofs =[19 7 6]  %   conectividade deste lado quadratico
%     assemblagem
  Kg(edofs,edofs)= Kg(edofs,edofs) + He  % 
  fg(edofs,1)= fg(edofs,1) + Pe          % 
  
[He Pe]=Robin_quadr(x(6),y(6),x(8),y(8),x(18),y(18),p,gama)
edofs =[6 8 18]  %   conectividade deste lado quadratico
%     assemblagem adicional de H e P
  Kg(edofs,edofs)= Kg(edofs,edofs) + He  % 
  fg(edofs,1)= fg(edofs,1) + Pe          %
  
  [He Pe]=Robin_quadr(x(18),y(18),x(115),y(115),x(116),y(116),p,gama)
edofs =[18 115 116]  %   conectividade deste lado quadratico
%     assemblagem adicional de H e P
  Kg(edofs,edofs)= Kg(edofs,edofs) + He  % 
  fg(edofs,1)= fg(edofs,1) + Pe          %
  
  [He Pe]=Robin_quadr(x(116),y(116),x(117),y(117),x(124),y(124),p,gama)
edofs =[116 117 124]  %   conectividade deste lado quadratico
%     assemblagem adicional de H e P
  Kg(edofs,edofs)= Kg(edofs,edofs) + He  % 
  fg(edofs,1)= fg(edofs,1) + Pe          %



u=Kg\fg;

%--------------------------------------------------------------
%   Tarefa 82 : Visualizacao da solucao em 3D usando fill3  
%--------------------------------------------------------------
figure
%
for i=1:Nels;
    no1=quad8(i,1);
    no2=quad8(i,2);
    no3=quad8(i,3);
    no4=quad8(i,4);
    no5=quad8(i,5);
    no6=quad8(i,6);
    no7=quad8(i,7);
    no8=quad8(i,8);
edofs=[no1 no5 no2 no6 no3 no7 no4 no8]; % ordem para desenhar    
fill3 (x(edofs),y(edofs),u(edofs),u(edofs));hold on
plot(x(edofs),y(edofs),'r');hold on
end

figure
%
for i=1:Nels;
    no1=quad8(i,1);
    no2=quad8(i,2);
    no3=quad8(i,3);
    no4=quad8(i,4);
    no5=quad8(i,5);
    no6=quad8(i,6);
    no7=quad8(i,7);
    no8=quad8(i,8);
edofs=[no1 no5 no2 no6 no3 no7 no4 no8]; % ordem para desenhar    
fill (x(edofs),y(edofs),u(edofs));hold on
plot(x(edofs),y(edofs),'r');hold on
end

%-------------------------------------------------------------------
%   Tarefa 83 - calcular (gradiente) fluxo nos pontos interiores
%-------------------------------------------------------------------
abs_vel_nds=zeros(1,Nnds);
figure
L=0;
for i=1:Nels;
edofs=[quad8(i,:)]; %   conectividade deste quad
  XN(1:8,1)=x(edofs);
  XN(1:8,2)=y(edofs);
%   O centroide esta na origem
csi=0;
eta=0;
nip = 4 ;    %  pts de integracao reduzida 
[xp wp]=Genip2DQ (nip);

%   percorrer os pontos de integracao
for ip=1:nip
L=L+1 ;
csi = xp(ip,1);
eta = xp(ip,2);
%   para cada ponto calcular
%----------------------------------------------------------------
[B psi Detj]=Shape_N_Der8 (XN,csi,eta);
%----------------------------------------------------------------
uint = psi'*u(edofs);
xpint = XN'*psi;
gradu = B'*u(edofs);
fluxu = -gradu/35;
vel(i,1)=gradu(1);
vel(i,2)=gradu(2);
abs_vel_ip(ip)=sqrt((vel(i,1))^2+(vel(i,2))^2);
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
quiver(xpint(1),xpint(2),fluxu(1),fluxu(2),'MaxHeadSize',0.8,'Color','r','LineWidth',1.1, 'AutoScale','off'); hold on

end
abs_vel_nds(edofs)=(abs_vel_ip(1)+abs_vel_ip(2)+abs_vel_ip(3)+abs_vel_ip(4))/4;
abs_vel(i)=abs_vel_nds(edofs(1));
pressure(i)=101328.8281-0.6125*(abs_vel(i))^2;
no1=quad8(i,1);
    no2=quad8(i,2);
    no3=quad8(i,3);
    no4=quad8(i,4);
    no5=quad8(i,5);
    no6=quad8(i,6);
    no7=quad8(i,7);
    no8=quad8(i,8);
edofs=[no1 no5 no2 no6 no3 no7 no4 no8];
plot(x(edofs),y(edofs),'k:');hold on
end
%plot(x1,y1,'ro');

figure


for i=1:Nels;
    no1=quad8(i,1);
    no2=quad8(i,2);
    no3=quad8(i,3);
    no4=quad8(i,4);
    no5=quad8(i,5);
    no6=quad8(i,6);
    no7=quad8(i,7);
    no8=quad8(i,8);
edofs=[no1 no5 no2 no6 no3 no7 no4 no8]; % ordem para desenhar    
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

parede1=[19 13 12 14 21 74 106];
for i=1:6
    L=sqrt((x(parede1(i))-x(parede1(i+1)))^2+(y(parede1(i))-y(parede1(i+1)))^2);
    F1=F1+L*(101325-(p(parede1(i))+p(parede1(i+1)))/2);
end

parede2=[106 83 78 82 77 81];
for i=1:5
    L=sqrt((x(parede2(i))-x(parede2(i+1)))^2+(y(parede2(i))-y(parede2(i+1)))^2);
    F2=F2+L*(101325-(p(parede2(i))+p(parede2(i+1)))/2);
end

parede3=[81 76 80 75 79 107];
for i=1:5
    L=sqrt((x(parede3(i))-x(parede3(i+1)))^2+(y(parede3(i))-y(parede3(i+1)))^2);
    F3=F3+L*(101325-(p(parede3(i))+p(parede3(i+1)))/2);
end

parede4=[107 103 101 104 100 102 31 26 24 25 33];
for i=1:10
    L=sqrt((x(parede4(i))-x(parede4(i+1)))^2+(y(parede4(i))-y(parede4(i+1)))^2);
    F4=F4+L*(101325-(p(parede4(i))+p(parede4(i+1)))/2);
end

parede5=[108 98 95 99 96 97 109];
for i=1:6
    L=sqrt((x(parede5(i))-x(parede5(i+1)))^2+(y(parede5(i))-y(parede5(i+1)))^2);
    F5=F5+L*(101325-(p(parede5(i))+p(parede5(i+1)))/2);
end

parede6=[109 105 32 29 27 28 34];
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
%xlswrite('.\QUAD8\Results_quad8.xlsx',nos_num,1,'A2')
%xlswrite('.\QUAD8\Results_quad8.xlsx',u,1,'B2')

ele_num=[1:Nels,1]';
%xlswrite('.\QUAD8\Results_quad8.xlsx',ele_num,1,'D2')
%xlswrite('.\QUAD8\Results_quad8.xlsx',vel(:,1),1,'E2')
%xlswrite('.\QUAD8\Results_quad8.xlsx',vel(:,2),1,'F2')
%xlswrite('.\QUAD8\Results_quad8.xlsx',abs_vel',1,'G2')
%xlswrite('.\QUAD8\Results_quad8.xlsx',pressure',1,'H2')
%xlswrite('.\QUAD8\Results_quad8.xlsx',POTmax',1,'K2')
%xlswrite('.\QUAD8\Results_quad8.xlsx',Umax',1,'K3')
%xlswrite('.\QUAD8\Results_quad8.xlsx',Umin',1,'K4')
%xlswrite('.\QUAD8\Results_quad8.xlsx',Pmax',1,'K5')
%xlswrite('.\QUAD8\Results_quad8.xlsx',Pmin',1,'K6')
%xlswrite('.\QUAD8\Results_quad8.xlsx',F,1,'K10') 
