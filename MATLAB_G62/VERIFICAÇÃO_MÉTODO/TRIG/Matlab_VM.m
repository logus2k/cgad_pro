%IMPORT DATA FROM EXCEL x y TRI
x=x*1000;
y=y*1000;
%mm->m

triplot(TRI, x, y);

Nelt=size(TRI,1);
Nnds=9;

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
Kr(9,9)= boom;
fr(9) = 0;
Kr(5,5)= boom;
fr(5) = 0;
Kr(8,8)= boom;
fr(8) = 0;

gama=-10;
index=[7 2 6]
	for i=1:3
        if (i~=3)
		h = sqrt((x(index(i+1))-x(index(i)))^2+(y(index(i+1))-y(index(i)))^2);
		fr(index(i))= fr(index(i)) + gama*h/2;   
    	fr(index(i+1))= fr(index(i+1)) + gama*h/2;
        end
	end

Kr=sparse(Kr);
u=Kr\fr ;
figure
%trisurf(TRI, x, y, u); grid on
for i=1:Nelt;
edofs=[TRI(i,:)];
fill3 (x(edofs),y(edofs),u(edofs),u(edofs));hold on
%plot(x(edofs),y(edofs),'r');hold on
end
plot(x,y,'ro');

for i=1:Nelt;
    no1 = TRI(i,1);
    no2 = TRI(i,2);
    no3 = TRI(i,3);
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
end

figure
 plot (x,y,'ko'); hold
 quiver (xm,ym,um,vm,'MaxHeadSize',0.8,'Color','r','LineWidth',1.1, 'AutoScale','on')
 triplot(TRI, x, y,'k:');