%% Ejercicio 3: representación de señales continuas
t=linspace(-5,5,101);
x=sin(pi*t/4);
figure(1);
plot(t,x);
title('sin(pi*t/4)');
xlabel('Tiempo (Continuo)')
ylabel('Valor')

t=[-4:1/8:4];
x1=sin(pi*t/4);
figure(2);
plot(t,x1,'r');
hold on
stem(t,x1,'r');
hold off
x2=cos(pi*t/4);
hold on
plot(t,x2,'g');
hold on
stem(t,x2,'g');
hold off

title('sin(pi*t/4) y cos(pi*t/4)');
xlabel('Tiempo (Continuo)')
ylabel('Valor')
legend('sin(pi*t/4)','','cos(pi*t/4)')