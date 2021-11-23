%% Ejercicio 5: operaciones aritméticas con señales

x1=sin((pi/4)*[0:30]);
x2=cos((pi/7)*[0:30]);

y1=x1+x2;
y2=x1-x2;
y3=x1.*x2;
y4=x1./x2;
y5=2*x1;
y6=x1.^x2;

n=[0:30];

figure(1);
stem(n,x1);
title('x1');
xlabel('n');

figure(2);
stem(n,x2);
title('x2');
xlabel('n');

figure(3);
stem(n,y1);
title('x1+x2');
xlabel('n');

figure(4);
stem(n,y2);
title('x1-x2');
xlabel('n');

figure(5);
stem(n,y3);
title('x1.*x2');
xlabel('n');

figure(6);
stem(n,y4);
title('x1./x2');
xlabel('n');

figure(7);
stem(n,y5);
title('2*x1');
xlabel('n');

figure(8);
stem(n,y6);
title('x1 elevado a x2');
xlabel('n');
