%% Ejercicio 2: representación de dos señales en un rango dado
% x1[n] = delta[n]; x2 = delta[n+2]
% delta[n]=1 if n=0 else 0
nx1=[0:10];
x1=[1 zeros(1,10)];
figure(1);
fig1=Figure(nx1,x1,1);

nx2=[-5:5];
x2=[zeros(1,3) 1 zeros(1,7)];
figure(2);
fig2=Figure(nx2,x2,2);

figure(3);
fig3=Figure2(x1,1);
figure(4);
fig4=Figure2(x2,2);

function fig = Figure(n,x,t)
    fig = stem(n,x);
    if t==1
        title('x[n]');
    else
        title('x[n+2]');
    end
    xlabel('Tiempo (Discreto)')
    ylabel('Valor')
end

function fig2 = Figure2(x,t)
    fig2 = stem(x);
    if t==1
        title('x[n-1]');
    else
        title('x[n-4]');
    end
    xlabel('Tiempo (Discreto)')
    ylabel('Valor')
end