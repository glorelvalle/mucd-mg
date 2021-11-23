n=[0:15];
x1=4*sin((pi/4)*n);
[y1,z1]=f_obtiene_yz(x1);
stem(n,x1);
hold on;
stem(n,y1,'r');
stem(n,z1,'g');
hold off;
title('y1[n]/z1[n]');
xlabel('Tiempo (Continuo)')
ylabel('Valor')
legend('z1[n]','y1[n]')

function [y,z] = f_obtiene_yz(x)
    % [y,z] = f_obtiene_yz(x) admite una señal ‘x’ y
    % devuelve dos señales, ‘y’ y ‘z’, donde ‘y’ vale 2*x
    % y ‘z’ vale (5/9)*(x-3) y = 2.*x;
    y = 2.*x;
    z = (5/9).*(x-3);
end
