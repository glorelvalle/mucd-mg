%% Ejercicio6: scripts y funciones

n = [0:16];
x1 = cos(pi*n/4);
y1 = mean(x1);
stem(n,x1,'r')
title('x1[n]  = cos(pi*n/4) / media')
xlabel('Tiempo (Discreto)')
ylabel('x1[n]')
hold on
m1=y1*ones(1,17);
plot(n,m1,'g')
hold off
legend('Cos (pi*n/4)', 'Media (Cos (pi*n / 4))');