%% Use Android Sensors to Move the Mouse
% Uses the self made app, UDPMouse

% +ve axes
% ax = left
% ay = away from the screen
% az = vertically down

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
qPhone = quaternion([1 0 0 0]);
rob = Robot();
%% 
i=1;
f=0;
tic
disp('Move the mouse!')
posx = 1920/2;
posy = 1080/2;
rob.mouseMove(posx, posy);
[msg,~] = judp('RECEIVE',port,300, 10*1000);

while true
    [msg,~] = judp('RECEIVE',port,300, 5000);
    a = strtrim(split(char(msg)', ','));
    if isempty(a)
        break
    end
    b0 = split(a{1},'"'); 
    b0 = cell2mat(b0(4));
    if strcmp(b0,'update')
        b1 = split(a{2});
        b11 = split(b1(1), '[');
        orienttemp(1) = str2double(cell2mat(b11(2)));
        orienttemp(2) = str2double(cell2mat(b1(2)));
        orienttemp(3) = str2double(cell2mat(b1(3)));
        b12 = split(b1(4), ']');
        orienttemp(4) = str2double(cell2mat(b12(1)));
        accel(1) = str2double(cell2mat(extractBetween (a{3}, length('"linAccel":"[')+1, length(a{3}))));
        accel(2) = str2double(cell2mat(extractBetween (a{4},1, length(a{4}))));
        az1 = str2double(split(a{5},']'));
        accel(3) = az1(1);
    end
    if strcmp(b0,'click')
        c1 = split(a{2}, ':');
        c2 = split(c1{2}, '}');
        click = str2double(c2(1));
        switch click
            case -2
                rob.mousePress(InputEvent.BUTTON1_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON1_DOWN_MASK)
                rob.mousePress(InputEvent.BUTTON1_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON1_DOWN_MASK)
            case -1
                rob.mousePress(InputEvent.BUTTON1_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON1_DOWN_MASK)
            case 1
                rob.mousePress(InputEvent.BUTTON3_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON3_DOWN_MASK)
            case 2
                rob.mousePress(InputEvent.BUTTON3_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON3_DOWN_MASK)
                rob.mousePress(InputEvent.BUTTON3_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON3_DOWN_MASK)
        end
    end
    if strcmp(b0,'scroll')
        d1 = split(a{2}, ':');
        d2 = split(d1{2}, '}');
        scroll = str2double(d2(1));
        rob.mouseWheel(floor(scroll/50));
%         break
    end
    
    % With the toolbox
    qPhone = quaternion(orienttemp);    
    qRot = qPhone.normalize;
    eulRot(:) = quat2eul(qRot);
    % Position based
    posx =  1920/2 - sin(eulRot(1))*2700;
    posy = 1080/2 - sin(eulRot(3))*2700/1.7778;
    
%     Without the Toolbox
%     q = orienttemp;
%     eulRot(3) = atan(2*(q(1)*q(2)+q(3)*q(4)/(1-2*(q(2)^2+q(3)^2))))*180/pi;
%     eulRot(1) = atan(2*(q(1)*q(4)+q(2)*q(3))/(1-2*(q(3)^2+q(4)^2)))*180/pi;  
%   Position based
%     posx =  1920/2 - eulRot(1)*32;
%     posy = 1080/2 - eulRot(3)*18;
    
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
end
