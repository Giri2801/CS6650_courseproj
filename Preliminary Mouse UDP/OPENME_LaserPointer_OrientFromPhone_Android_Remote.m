%% Use Android Sensors to Move the Mouse

%% Setup 
clear all
% close all
import java.awt.Robot;
import java.awt.event.*;
import java.awt.PointerInfo;

R = 2000;

%% Android Connect & Rotate
% Connect and prepare to get data from the Android phone
% target port as set in the app
port = 5555;
accel(1:3) = 0;
mag(1:3) = 0;
gyro(1:3) = 0;
orient(1:3) = 0;
k(1:3) = 0;
qPhone = quaternion([1 0 0 0]);
qBase = quaternion([1 0 0 0]);

% Medium  =  15 Hz
% Fast    =  50 Hz
% Fastest = 200 Hz
% fuse = ahrsfilter('SampleRate', 400);
% fuse = complementaryFilter('SampleRate', 200, 'HasMagnetometer', true);

rob = Robot();
% pause(2)

%% 

i=2;
f=0;
tic
disp('Hold still for 3 seconds')
posx = 1920/2;
posy = 1080/2;
rob.mouseMove(posx, posy);

% viewer = HelperOrientationViewer;
viewer = HelperPoseViewer;

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
        if length(c)>13
        orient(i,:) = c(14:16);
        k = orient(i,:);
    end
    end
    
    
%     asd
%     PROBLEM HERE
    qPhone(i) = slerp(qPhone(i-1),quaternion(eul2quat(k*pi/180)), 0.5);
    
    if time(i)<3
        if time(i)<0.2
            qBase = qPhone(i);
        end
        qBase = slerp(qBase, qPhone(i), 0.2);
        i = i+1;
        continue
    end
    if time(i)>3 && f == 0
        disp('Done calibrating!');
        f = 1;
    end
    
    qRot(i) = qPhone(i)*qBase.conj;
    eulRot(i,:) = quat2eul(qRot(i))*180/pi;
    
%   Position Based
    posx = 1920/2 + R*tand(eulRot(i,1));
    posy = 1080/2 + R*tand(eulRot(i,3));

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
%     viewer([posx/1920 posy/1080 0],qRot(i)*quaternion([0,0,0,1]),[posx posy 0],qRot(i)*quaternion([0,0,0,1]))
%     viewer(qRot(i))
    i = i+1;
end

%% Plots
figure
scatter(ppx, 1080-ppy, 10, 1:length(ppx))
xlim([-10 1935])
ylim([-10 1095])
grid

figure
plot(eulRot)
legend
grid


