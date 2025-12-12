x=xlsread('mesh_data_TRI6.xlsx','coord','A1:A169')
x=x/1000; %mm to m
y=xlsread('mesh_data_TRI6.xlsx','coord','B1:B169')
y=y/1000; %mm to m
tri6=xlsread('mesh_data_TRI6.xlsx','conec','A1:F66')
%--------------------------------------------------------------------------
%	Tarefa 57 A : Representacao grafica 2D da malha gerada a partir daqui
%--------------------------------------------------------------------------
    figure(1)
%-------------------------------------
Nelt=size(tri6,1)
for i=1:Nelt;
    no1=tri6(i,1);
    no2=tri6(i,2);
    no3=tri6(i,3);
    no4=tri6(i,4);
    no5=tri6(i,5);
    no6=tri6(i,6);
edofs=[no1 no4 no2 no5 no3 no6];  % ordem adequada para desenhar

%fill (x(edofs),y(edofs),u(edofs));hold on
plot(x(edofs),y(edofs),'b');hold on
end
plot(x,y,'ro');
%----------------------------------------------------------------------
Nelt=size(tri6,1)        % numero de triangulos

Nnds =size(x,1)         % numero de nos
%   inicializacao a zeros
Kg=zeros(Nnds,Nnds);
fg=zeros(Nnds,1);

%----------------------------------------------------------------------
%	Tarefa 54 : Assemblagem de elementos triangulares de 6-nós
%----------------------------------------------------------------------
for i=1:Nelt    
    no1=tri6(i,1);
    no2=tri6(i,2);
    no3=tri6(i,3);
    no4=tri6(i,4);
    no5=tri6(i,5);
    no6=tri6(i,6);
  edofs =[no1 no2 no3 no4 no5 no6];  %   conectividade deste triangulo
  XN(1:6,1)=x(edofs);
  XN(1:6,2)=y(edofs);
  %     calculos no elemento
fL= 0;
[Ke fe]=Elem_TRI6 (XN,fL);

  %     assemblagem
  Kg(edofs,edofs)= Kg(edofs,edofs) + Ke ;  % 
  fg(edofs,1)= fg(edofs,1) + fe    ;       % 
end %for i
Kg;
fg;
%----------------------------------------------------------------------
%	Tarefa 55 : Condicoes de fronteira essenciais 
%----------------------------------------------------------------------
boom=1.0e+12;
nout=[33 30 34 169 167 168]
for i=1:6
    j=nout(i)
    Kg(j,j) = boom  ;
    fg(j,1)= boom*0 ;
end

%gama=2.5;
%index=[17 8 16 98 103]
	%for i=1:5
       % if (i~=5)
		%h = sqrt((x(index(i+1))-x(index(i)))^2+(y(index(i+1))-y(index(i)))^2);
		%fg(index(i))= fg(index(i)) + gama*h/2;   
    	%fg(index(i+1))= fg(index(i+1)) + gama*h/2;
        %end
    %end
    
    %--------------------------------------------------------------------------
%   Tarefa 62 : Assemblagem de H e P, condicoes de Robin, lados curvos
%--------------------------------------------------------------------------
% Atencao: so os lados quadraticos sobre a fronteira
p=0
gama=2.5
[He Pe]=Robin_quadr(x(17),y(17),x(8),y(8),x(16),y(16),p,gama)
edofs =[17 8 16]  %   conectividade deste lado quadratico
%     assemblagem
  Kg(edofs,edofs)= Kg(edofs,edofs) + He  % 
  fg(edofs,1)= fg(edofs,1) + Pe          % 
[He Pe]=Robin_quadr(x(16),y(16),x(98),y(98),x(103),y(103),p,gama)
edofs =[16 98 103]  %   conectividade deste lado quadratico
%     assemblagem adicional de H e P
  Kg(edofs,edofs)= Kg(edofs,edofs) + He  % 
  fg(edofs,1)= fg(edofs,1) + Pe          %


    
%----------------------------------------------------------------------
%	Tarefa 56 : Resolver sistema de equações por backslash
%----------------------------------------------------------------------
Kr=sparse(Kg) ;
u=Kr\fg ;

%
%
%-----------------------------------------------------
figure
Nelt=size(tri6,1) ;
for i=1:Nelt
    no1=tri6(i,1) ;
    no2=tri6(i,2) ;
    no3=tri6(i,3) ;
    no4=tri6(i,4) ;
    no5=tri6(i,5) ;
    no6=tri6(i,6) ;
%----------------------------------------------------------------------
%	Tarefa 57 A : Representacao da malha e da solucao a 2D
%----------------------------------------------------------------------
edofs=[tri6(i,1) tri6(i,4) tri6(i,2) tri6(i,5) tri6(i,3) tri6(i,6)] ;
fill (x(edofs),y(edofs),u(edofs));hold on
plot(x(edofs),y(edofs),'b');hold on
end


figure
Nelt=size(tri6,1) ;
for i=1:Nelt
    no1=tri6(i,1) ;
    no2=tri6(i,2) ;
    no3=tri6(i,3) ;
    no4=tri6(i,4) ;
    no5=tri6(i,5) ;
    no6=tri6(i,6) ;
%----------------------------------------------------------------------
%	Tarefa 57 B : Representacao da malha e da solucao a 3D
%----------------------------------------------------------------------
edofs=[tri6(i,1) tri6(i,4) tri6(i,2) tri6(i,5) tri6(i,3) tri6(i,6)] ;
fill3 (x(edofs),y(edofs),u(edofs),u(edofs));hold on
plot(x(edofs),y(edofs),'r');hold on
end


%----------------------------------------------------------------------
%	Tarefa 58 : Calcular (gradiente) fluxo nos centroides
%----------------------------------------------------------------------
Nelt=size(tri6,1) ;
psi=zeros(6,1)    ;
figure
plot(x,y,'ro');hold on
for i=1:Nelt
    no1=tri6(i,1) ;
    no2=tri6(i,2) ;
    no3=tri6(i,3) ;
    no4=tri6(i,4) ;
    no5=tri6(i,5) ;
    no6=tri6(i,6) ;
edofs =[no1 no2 no3 no4 no5 no6] ;  %   conectividade deste triangulo
  XN(1:6,1)=x(edofs) ;
  XN(1:6,2)=y(edofs) ;  
csi=1/3 ;
eta=1/3 ;
%----------------------------------------------------------------
[B psi Detj]=Shape_N_Der6 (XN,csi,eta) ;
%----------------------------------------------------------------

xpint = XN'*psi ;
uint = psi'*u(edofs) ;
gradu = B'*u(edofs) ;
fluxu = -gradu/25 ;
%--------------------------------------------------------------------------
    umex(i) = 2*xpint(1);
    vmex(i) = 2*xpint(2);

%--------------------------------------------------------------------------
vel(i,1)=gradu(1);
vel(i,2)=gradu(2);
abs_vel(i)=sqrt((vel(i,1))^2+(vel(i,2))^2)
abs_vel_nds(edofs)=abs_vel(i);
pressure(i)=101328.8281-0.6125*(abs_vel(i))^2;
edofs=[tri6(i,1) tri6(i,4) tri6(i,2) tri6(i,5) tri6(i,3) tri6(i,6)];
plot(x(edofs),y(edofs),'k:');hold on
quiver(xpint(1),xpint(2),fluxu(1),fluxu(2),'MaxHeadSize',0.8,'Color','r','LineWidth',1.1, 'AutoScale','on');hold on
end
plot(x,y,'ko');

figure
for i=1:Nelt;
    no1=tri6(i,1);
    no2=tri6(i,2);
    no3=tri6(i,3);
    no4=tri6(i,4);
    no5=tri6(i,5);
    no6=tri6(i,6);
edofs=[no1 no4 no2 no5 no3 no6]; % ordem para desenhar    
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

parede1=[17 13 12 14 19 67 87];
for i=1:6
    L=sqrt((x(parede1(i))-x(parede1(i+1)))^2+(y(parede1(i))-y(parede1(i+1)))^2);
    F1=F1+L*(101325-(p(parede1(i))+p(parede1(i+1)))/2);
end

parede2=[87 73 70 74 69];
for i=1:4
    L=sqrt((x(parede2(i))-x(parede2(i+1)))^2+(y(parede2(i))-y(parede2(i+1)))^2);
    F2=F2+L*(101325-(p(parede2(i))+p(parede2(i+1)))/2);
end

parede3=[69 71 68 72 88];
for i=1:4
    L=sqrt((x(parede3(i))-x(parede3(i+1)))^2+(y(parede3(i))-y(parede3(i+1)))^2);
    F3=F3+L*(101325-(p(parede3(i))+p(parede3(i+1)))/2);
end

parede4=[88 84 83 85 31 26 24 25 33];
for i=1:8
    L=sqrt((x(parede4(i))-x(parede4(i+1)))^2+(y(parede4(i))-y(parede4(i+1)))^2);
    F4=F4+L*(101325-(p(parede4(i))+p(parede4(i+1)))/2);
end

parede5=[89 82 80 81 90];
for i=1:4
    L=sqrt((x(parede5(i))-x(parede5(i+1)))^2+(y(parede5(i))-y(parede5(i+1)))^2);
    F5=F5+L*(101325-(p(parede5(i))+p(parede5(i+1)))/2);
end

parede6=[90 86 32 29 27 28 34];
for i=1:6
    L=sqrt((x(parede6(i))-x(parede6(i+1)))^2+(y(parede6(i))-y(parede6(i+1)))^2);
    F6=F6+L*(101325-(p(parede6(i))+p(parede6(i+1)))/2);
end

F=[F1 F2 F3 F4 F5 F6]';

%figure
%for i=1:Nelt;
    %no1=tri6(i,1);
    %no2=tri6(i,2);
    %no3=tri6(i,3);
    %no4=tri6(i,4);
    %no5=tri6(i,5);
    %no6=tri6(i,6);
%edofs=[no1 no4 no2 no5 no3 no6]; % ordem para desenhar    
%fill (x(edofs),y(edofs),p(edofs));hold on
%plot(x(edofs),y(edofs),'k:');hold on
%end

format longG
Pmax=max(p);
Pmin=min(p);
Umax=max(abs_vel_nds);
Umin=min(abs_vel_nds);
POTmax=max(u);

% Para que os rsultados sejam impressos no ficheiro excel desejado, tem de
% se alterar a diretoria de onde se encontra agora o excel
%nos_num=[1:Nnds,1]';
%xlswrite('.\TRI6\Results.xlsx',nos_num,1,'A2')
%xlswrite('.\TRI6\Results.xlsx',u,1,'B2')

%ele_num=[1:Nelt,1]';
%xlswrite('.\TRI6\Results.xlsx',ele_num,1,'D2')
%xlswrite('.\TRI6\Results.xlsx',vel(:,1),1,'E2')
%xlswrite('.\TRI6\Results.xlsx',vel(:,2),1,'F2')
%xlswrite('.\TRI6\Results.xlsx',abs_vel',1,'G2')
%xlswrite('.\TRI6\Results.xlsx',pressure',1,'H2')
%xlswrite('.\TRI6\Results.xlsx',POTmax',1,'K2')
%xlswrite('.\TRI6\Results.xlsx',Umax',1,'K3')
%xlswrite('.\TRI6\Results.xlsx',Umin',1,'K4')
%xlswrite('.\TRI6\Results.xlsx',Pmax',1,'K5')
%xlswrite('.\TRI6\Results.xlsx',Pmin',1,'K6')
%xlswrite('.\TRI6\Results.xlsx',F,1,'K10')