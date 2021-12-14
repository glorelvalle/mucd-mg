% Propiedad de Modulación 
%%
% Apartado a)
%%
% señal
L = 21;
x = [ones(1,L)];
n = [0:L-1];

% exponencial compleja
w0 = 2*pi/sqrt(31);
N = 128;
e = exp(j*w0*n);

% señal multiplicada por exponencial compleja
xe = x.*e;

% Representamos señales
plot_dtft(x, N);
plot_dtft(xe, N);

% Para calcular el desplazamiento
[H, W] = dtft(xe, N);
[argvalue, argmax] = max(abs(H));
-1 + 2*(argmax-1)/N
w0 / pi

%%
% Apartado b) Repetir para w0 =5pi/2 
%%
w0 = 5*pi/2; % 2*pi + pi/2
e = exp(j*w0*n);
xe = x.*e;
plot_dtft(x, N);
plot_dtft(xe, N);

% El desplazamiento es mucho mayor, de hecho, w0 = 2*pi + pi/2, luego el
% desplazamiento es pi/2/pi = 0.5
[H, W] = dtft(xe, N);
[argvalue, argmax] = max(abs(H));
-1 + 2*(argmax-1)/N
w0 / pi

% Apartado c) Repetir el experimento anterior multiplicando el pulso
% por una función coseno a la misma frecuencia que en el apartado a). 
% Este tipo de modulación se denomina modulación AM en doble banda. 
% Dibuje de nuevo las gráficas y explique si le parece razonable 
% este nombre y por qué.

w0 = 2*pi/sqrt(31);
e = cos(w0*n);
xe = x.*e;
plot_dtft(x, N);
plot_dtft(xe, N);

% Es doble banda porque se suman dos señales debido a la descomposición
% del coseno en dos exponenciales
% Además, la máxima magnitud de la respuesta se divide entre dos.








