%% Ejercicio6b

n = [0:32];
x1 = sin(pi*n/4);
y1 = mean(x1);
stem(n,x1,'r')
title('x1[n] = sen(pi*n/4) / media')
xlabel('Tiempo (Discreto)')
ylabel('x1[n]')
hold on
m1=y1*ones(1,33);
plot(n,m1,'g')
hold off
legend('Sen (pi*n/4)', 'Media (Sen (pi*n / 4))');
