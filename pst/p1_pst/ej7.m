%% Ejercicio 7: transformaciones de la variable independiente

    nx=[-3:11];
    x=[zeros(1, 3) 2 0 1 -1 3 zeros(1,7)];
    
    figure;
    stem(nx, x);
    title('Señal discreta X[n]');
    xlabel('Tiempo (Discreto)');
    ylabel('Valor');
    
    figure;
    subplot(2,2,1);
    y1=x;
    ny1=nx+2;
    stem(ny1,y1);
    title('x retrasada 2');
    xlabel('ny1');
    xlim([-11 13]);
    
    subplot(2,2,2);
    y2=x;
    ny2=nx-1;
    stem(ny2,y2);
    title('x adelantada 1');
    xlabel('ny2');
    xlim([-11 13]);
    
    subplot(2,2,3);
    ny3=-fliplr(nx);
    y3=fliplr(x);
    stem(ny3,y3);
    title('x invertida');
    xlabel('ny3');
    xlim([-11 13]);
    
    subplot(2,2,4);
    y4=y3;
    ny4=ny3+1;
    stem(ny4,y4);
    title('x invertida y adelantada 1');
    xlabel('ny4');
    xlim([-11 13]);

