% Propiedad de Enventanado

% a) No entiendo que tengo que concluir

N = 1024;
L = 32;
x = [ones(1,L)];
n = [0:L-1];
w0 = 2*pi/sqrt(31);
e = exp(j*w0*n);
xe = e.*x;
[H, W] = dtft(xe, N);
norm_freq = W ./ pi;
figure;
plot(norm_freq, abs(H));
title('Magnitude Response');
xlabel('Normalized Frequency') ;
ylabel('|H(w)|');


N = 1024;
L = 1024;
x = [ones(1,L)];
n = [0:L-1];
w0 = 2*pi/sqrt(31);
e = exp(j*w0*n);
xe = e.*x;
[H, W] = dtft(xe, N);
norm_freq = W ./ pi;
figure;
plot(norm_freq, abs(H));
title('Magnitude Response');
xlabel('Normalized Frequency') ;
ylabel('|H(w)|');

% b)
N=1024;
w0 = 2*pi/sqrt(31);

L1=32;
x1 = [ones(1,L1)];
n1 = [0:L1-1];
e = exp(j*w0*n1);
xe1 = e.*x1;
[H1, W1] = dtft(xe1, N);
norm_freq1 = W1 ./ pi;

L2=64;
x2 = [ones(1,L2)];
n2 = [0:L2-1];
e = exp(j*w0*n2);
xe2 = e.*x2;
[H2, W2] = dtft(xe2, N);
norm_freq2 = W2 ./ pi;

L3=128;
x3 = [ones(1,L3)];
n3 = [0:L3-1];
e = exp(j*w0*n3);
xe3 = e.*x3;
[H3, W3] = dtft(xe3, N);
norm_freq3 = W3 ./ pi;

L4=256;
x4 = [ones(1,L4)];
n4 = [0:L4-1];
e = exp(j*w0*n4);
xe4 = e.*x4;
[H4, W4] = dtft(xe4, N);
norm_freq4 = W4 ./ pi;


figure;
subplot(2,2,1);
plot(norm_freq1, abs(H1));
title('Magnitude Response L=32');
xlabel('Normalized Frequency') ;
ylabel('|H(w)|');

subplot(2,2,2);
plot(norm_freq2, abs(H2));
title('Magnitude Response L=64');
xlabel('Normalized Frequency') ;
ylabel('|H(w)|');

subplot(2,2,3);
plot(norm_freq3, abs(H3));
title('Magnitude Response L=128');
xlabel('Normalized Frequency') ;
ylabel('|H(w)|');

subplot(2,2,4);
plot(norm_freq4, abs(H4));
title('Magnitude Response L=256');
xlabel('Normalized Frequency') ;
ylabel('|H(w)|');


% Ventana hanning
N=1024;
w0 = 2*pi/sqrt(31);
L =32;
n = [0:L-1];
e=exp(j*w0*n);


xh = hann(L)';
plot_dtft(xh, N);
xhe=xh.*e;


xw = [ones(1, L)];
plot_dtft(xw, N);
xwe=xw.*e;

plot_dtft(xhe, N);
plot_dtft(xwe, N);

