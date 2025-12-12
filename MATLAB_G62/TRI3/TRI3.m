%IMPORT DATA FROM EXCEL x y TRI
x=xlsread('mesh_data_tri3.xlsx','cord','A1:A173')
x=x/1000; %mm to m
y=xlsread('mesh_data_tri3.xlsx','cord','B1:B173')
y=y/1000; %mm to m
TRI=xlsread('mesh_data_tri3.xlsx','conec','A1:C264')
triplot(TRI, x, y);

Nelt=size(TRI,1);
Nnds=173;

%   inicializacao a zeros
Kg=zeros(Nnds,Nnds);
fg=zeros(Nnds,1);	% declarado como vector coluna

%--------------------------------------------------------------------------
%   ciclo para assemblagem dos elementos
for i=1:Nelt
    no1 = TRI(i,1);
    no2 = TRI(i,2);
    no3 = TRI(i,3);
  edofs =[no1 no2 no3];  %   guardar a conectividade deste triangulo

  %     calculos neste elemento i, P15b)
  [Ke fe]= Elem_TRI (x(no1),y(no1),x(no2),y(no2),x(no3),y(no3),0);   % <- carregamento f=0 aqui

  %     assemblagem do elemento 
  Kg(edofs,edofs)= Kg(edofs,edofs) + Ke ; % 
  fg(edofs,1)= fg(edofs,1) + fe ;         % 
end %	ciclo for i

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%	Atencao: Guardar uma copia do sistema original antes de o modificar
Kr= Kg;
fr= fg;

%   Vamos precisar do sistema após introduzir as condições de fronteira naturais
%   para verificar o equilibrio estatico e Calcular as reacoes nos apoios   
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%
boom = 1.0e+14 ;     %   como se adicionasse uma mola de rigidez (quase) infinita
%

%Parede de cima
Kr(34,34)= boom;
fr(34) = 0;
Kr(31,31)= boom;
fr(31) = 0;
Kr(35,35)= boom;
fr(35) = 0;

%Parede de baixo
Kr(173,173)= boom;
fr(173) = 0;
Kr(171,171)= boom;
fr(171) = 0;
Kr(172,172)= boom;
fr(172) = 0;

gama=2.5;
index=[18 7 8 17 100 101 107]
	for i=1:7
        if (i~=7)
		h = sqrt((x(index(i+1))-x(index(i)))^2+(y(index(i+1))-y(index(i)))^2);
		fr(index(i))= fr(index(i)) + gama*h/2;   
    	fr(index(i+1))= fr(index(i+1)) + gama*h/2;
        end
	end

Kr=sparse(Kr);
u=Kr\fr ;
figure
trisurf(TRI, x, y, u); grid on

abs_vel_nds=zeros(1,Nnds);
for i=1:Nelt;
    no1 = TRI(i,1);
    no2 = TRI(i,2);
    no3 = TRI(i,3);
    edofs=[TRI(i,:)];
    %   copia coordenadas
    x1=x(no1);
    x2=x(no2);
    x3=x(no3);
    y1=y(no1);
    y2=y(no2);
    y3=y(no3);
    %   calcula centroide
    xm(i) = (x1+x2+x3)/3.;
    ym(i) = (y1+y2+y3)/3.;
    %
    %   calcula vector gradiente no elemento   
    Ae2 = (x2 -x1)*(y3 -y1) -(y2 -y1)*(x3 -x1);
    %   derivadas parciais das funcoes de forma
    d1dx = (y2-y3)/Ae2;
    d1dy = (x3-x2)/Ae2;
    d2dx = (y3-y1)/Ae2;
    d2dy = (x1-x3)/Ae2;
    d3dx = (y1-y2)/Ae2;
    d3dy = (x2-x1)/Ae2;
    %   interpolacao e derivadas
    um(i) = -(d1dx*u(no1)+d2dx*u(no2)+d3dx*u(no3));
    vm(i) = -(d1dy*u(no1)+d2dy*u(no2)+d3dy*u(no3));
    %   sinal negativo para o fluxo
    abs_vel(i)=sqrt((um(i))^2+(vm(i))^2);
    abs_vel_nds(edofs)=abs_vel(i);
    pressure(i)=101328.8281-0.6125*(abs_vel(i))^2;
    
end
figure
for i=1:Nelt;
    edofs=[TRI(i,:) TRI(i,:)];
    fill (x(edofs),y(edofs),abs_vel_nds(edofs));hold on
end

figure
 plot (x,y,'ko'); hold
 quiver (xm,ym,um,vm,'r')
 triplot(TRI, x, y, 'k:');
 
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

parede1=[18 14 13 12 20 65 90];
for i=1:6
    L=sqrt((x(parede1(i))-x(parede1(i+1)))^2+(y(parede1(i))-y(parede1(i+1)))^2);
    F1=F1+L*(101325-(p(parede1(i))+p(parede1(i+1)))/2);
end

parede2=[90 73 72 71 70 69];
for i=1:5
    L=sqrt((x(parede2(i))-x(parede2(i+1)))^2+(y(parede2(i))-y(parede2(i+1)))^2);
    F2=F2+L*(101325-(p(parede2(i))+p(parede2(i+1)))/2);
end

parede3=[70 69 68 67 66 91];
for i=1:5
    L=sqrt((x(parede3(i))-x(parede3(i+1)))^2+(y(parede3(i))-y(parede3(i+1)))^2);
    F3=F3+L*(101325-(p(parede3(i))+p(parede3(i+1)))/2);
end

parede4=[91 88 87 86 85 84 32 27 26 25 34];
for i=1:10
    L=sqrt((x(parede4(i))-x(parede4(i+1)))^2+(y(parede4(i))-y(parede4(i+1)))^2);
    F4=F4+L*(101325-(p(parede4(i))+p(parede4(i+1)))/2);
end

parede5=[92 81 82 83 93];
for i=1:4
    L=sqrt((x(parede5(i))-x(parede5(i+1)))^2+(y(parede5(i))-y(parede5(i+1)))^2);
    F5=F5+L*(101325-(p(parede5(i))+p(parede5(i+1)))/2);
end

parede6=[93 89 33 28 29 30 35];
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
%nos_num=[1:Nnds,1]';
%xlswrite('.\TRIG3\Results_tri3.xlsx',nos_num,1,'A2')
%xlswrite('.\TRIG3\Results_tri3.xlsx',u,1,'B2')

%ele_num=[1:Nelt,1]';
%xlswrite('.\TRIG3\Results_tri3.xlsx',ele_num,1,'D2')
%xlswrite('.\TRIG3\Results_tri3.xlsx',um',1,'E2')
%xlswrite('.\TRIG3\Results_tri3.xlsx',vm',1,'F2')
%xlswrite('.\TRIG3\Results_tri3.xlsx',abs_vel',1,'G2')
%xlswrite('.\TRIG3\Results_tri3.xlsx',pressure',1,'H2')
%xlswrite('.\TRIG3\Results_tri3.xlsx',POTmax',1,'K2')
%xlswrite('.\TRIG3\Results_tri3.xlsx',Umax',1,'K3')
%xlswrite('.\TRIG3\Results_tri3.xlsx',Umin',1,'K4')
%xlswrite('.\TRIG3\Results_tri3.xlsx',Pmax',1,'K5')
%xlswrite('.\TRIG3\Results_tri3.xlsx',Pmin',1,'K6')
%xlswrite('.\TRIG3\Results_tri3.xlsx',F,1,'K10')
    