%% Ejercicio 8: Propiedades Conmutativa, Asociativa y Distributiva de la Convolución, 
%% y su Aplicación a Sistemas Lineales e Invariantes. 

%%%%%%%%%% a %%%%%%%%%%
nx1=[0:9];
x1=[ones(1,5) zeros(1, 5)]

nh1=[0:4]
h1=[1 -1 3 0 0]

nh2=[0:4]
h2=[0 2 5 4 -1]

figure;
subplot(2,2,1);
stem(nx1,x1);
title('x1');
xlabel('nx1');

subplot(2,2,3);
stem(nh1,h1);
title('h1');
xlabel('nh1');

subplot(2,2,4);
stem(nh2,h2);
title('h2');
xlabel('nh2');

%%%%%%%%%% b %%%%%%%%%%
c1=conv(h1,x1)
c2=conv(x1,h1)
isequal(c1,c2)

nc=[0:13]

figure;
subplot(2,2,1);
stem(nc,c1);
title('c1');
xlabel('nc');

subplot(2,2,2);
stem(nc,c2);
title('c2');
xlabel('nc');

%%%%%%%%%% c %%%%%%%%%%
c1=conv(x1,h1+h2);
c2=conv(x1,h1) + conv(x1,h2);
isequal(c1,c2);

figure;
subplot(2,2,1);
stem(nc,c1);
title('x1*(h1+h2)');
xlabel('nc');
xlim([-1 14]);

subplot(2,2,2);
stem(nc,c2);
title('x1*h1+ x1*h2');
xlabel('nc');
xlim([-1 14]);


%%%%%%%%%% d %%%%%%%%%%
w=conv(x1,h1);
y1=conv(w,h2);
hseries=conv(h1,h2);
y2=conv(x1,hseries);

figure;
subplot(2,2,1);
nc=[0:13];
stem(nc,w);
title('w');
xlabel('nc');
xlim([-1 18]);

subplot(2,2,2);
nc=[0:17];
stem(nc,y1);
title('y1');
xlabel('nc');
xlim([-1 18]);

subplot(2,2,3); 
nc=[0:8];
stem(nc,hseries);
title('hseries');
xlabel('nc');
xlim([-1 18]);

subplot(2,2,4);
nc=[0:17];
stem(nc,y2);
title('y2');
xlabel('nc');
xlim([-1 18]);

%%%%%%%%%% e %%%%%%%%%%
he1=h1;
he2=[zeros(1,2) he1];
nhe1=nh1;
nhe2=[nhe1 5 6 ];

ye1=conv(he1,x1);
nye1 =[0:13];
ye2=conv(he2,x1);
ye3=ye1;
nye2=[0:15];
nye3=nye+2;

figure;

subplot(2,2,1);
stem(nye1,ye1);
title('ye1[n]=ye2[n]');

subplot(2,2,2);
stem(nye2,ye2);
title('ye1[n-2]');

%subplot(2,2,3);
%stem(nye3,ye3);
%title('ye1[n]');

%%%%%%%%%% f %%%%%%%%%%
%% POR ACABAR

inter=[0:13];

nw=nx1;
w=(nx1+1).*x;


yf1=conv(w,h1);

nc=[0:4]
hf1=[1 zeros(1,4)]

hseries=conv(hf1,h1);
%%%%%%%%%% g %%%%%%%%%%
