%% Ejercicio 4: representación de señales complejas
n=[0:32];
x=exp(j*(pi/8)*n);

figure(1);
stem(n,abs(x),'r');
hold on
stem(n,angle(x),'g');
hold off
title('Abs y Angle');
xlabel('Tiempo (Discreto)');
ylabel('Valor');
legend('Valor absoluto', 'Fase')

figure(2);
stem(n,real(x),'r');
hold on
stem(n,imag(x),'g');
hold off
title('Real y Imag');
xlabel('Tiempo (Discreto)');
ylabel('Valor');
legend('Parte Real', 'Parte Imaginaria')
