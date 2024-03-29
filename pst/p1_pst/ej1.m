%% Ejercicio 1: representación de una señal en un rango dado
% x[n] = 2n if -3 <= n 3 else 0
n=[-3:3];
x=2*n;
figure(1);
fig1=Figure(n,x);

n=[-5:5];
x=[0 0 x 0 0];
figure(2);
fig2=Figure(n,x);

n=[-100:100];
x=[zeros(1,95) x zeros(1,95)];
figure(3);
fig3=Figure(n,x);

function fig = Figure(n,x)
    fig = stem(n,x);
    t=sprintf('n[%d, %d]', n(1), n(end))
    title(t)
    xlabel('Tiempo (Discreto)')
    ylabel('Valor')
end