%% Ejercicio 3: representación de señales continuas
%t=[-5:0.1:5];
t=linspace(-5,5,101);
x=sin(pi*t/4);
figure;
fig=Figure(t,x,1);

t=[-4:1/8:4];
x1=sin(pi*t/4);
fig1=Figure(t,x1,1);
x2=cos(pi*t/4);
hold on
fig2=Figure(t,x2,2);
hold off

function fig = Figure(t,x,op)
    fig = plot(t,x);
    hold on
    fig = stem(t, x);
    hold off
    if op == 1
        title('sin(pi*t/4)');
    elseif op == 2
        title('cos(pi*t/4)');
    end
    xlabel('Tiempo (Continuo)')
    ylabel('Valor')
end