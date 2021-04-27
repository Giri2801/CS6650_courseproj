%% Use Android Sensors to Move the Mouse

%% Setup 
clear all
close all
import java.awt.Robot;
import java.awt.event.*;
import java.awt.PointerInfo;


%% Android Connect & Rotate
% Connect and prepare to get data from the Android phone
% target port as set in the app
port = 5555;
accel(1:3) = 0;
mag(1:3) = 0;
gyro(1:3) = 0;
qPhone = quaternion([1 0 0 0]);

% Medium  =  15 Hz
% Fast    =  50 Hz
% Fastest = 200 Hz
fuse = ahrsfilter('SampleRate', 400);
% fuse = complementaryFilter('SampleRate', 200, 'HasMagnetometer', true);

rob = Robot();
pause(2)

%% 

i=1;
f=0;
tic
disp('Hold still for 3 seconds')
posx = 1920/2;
posy = 1080/2;
rob.mouseMove(posx, posy);

while true
    time(i) = toc;
    [msg,~] = judp('RECEIVE',port,200);
    a = strtrim(split(char(msg)', ','));
    c(1:length(a)-1) = str2double(a(2:length(a)));
    accel = c(2:4);
    if length(c)>5
        gyro = c(6:8)*pi/180;
    end
    if length(c)>9
        mag = c(10:12)*10^-6;
    end
    
    qPhone(i+1) = fuse(accel, gyro, mag);
    if time(i)<3
        if time(i)<0.2
            qBase = qPhone(i);
        end
        qBase = slerp(qBase, qPhone(i), 0.5);
        qBase = qBase.normalize;
        i = i+1;
        continue
    end
    if time(i)>3 && f == 0
        disp('Done calibrating!');
        f = 1;
    end
    
    qRot(i) = qPhone(i).normalize*qBase.conj;
    eulRot(i,:) = quat2eul(qRot(i))*180/pi;
    
    posx = posx + (eulRot(i,2));
    posy = posy - (eulRot(i,3));
%     gr = gradient(eulRot);
%     posx =  1920/2 + eulRot(i,2)*32;
%     posy = 1080/2 - eulRot(i,3)*18;
    ppx(i) = posx;
    ppy(i) = posy;
    
    if posx>1920
        posx = 1920;
    end
    if posx<0
        posx = 0;
    end
    if posy>1080
        posy = 1080;
    end
    if posy<0
        posy = 0;
    end
    rob.mouseMove(posx, posy);
    i = i+1;
end

%% Plots
figure
scatter(ppx, 1080-ppy, 10, 1:length(ppx))
xlim([-10 1930])
ylim([-10 1090])

figure
plot(quat2eul(qPhone)*180/pi)

