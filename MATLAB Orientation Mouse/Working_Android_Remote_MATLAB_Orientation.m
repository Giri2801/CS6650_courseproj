%% Use Android Sensors to Move the Mouse
% Uses the self made app, UDPMouse
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

while true
    [msg,~] = judp('RECEIVE',port,300, 2000);
    a = strtrim(split(char(msg)', ','));
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
    end
    if strcmp(b0,'')
        b3 = split(a{3},':');
        b32 = split(b3{2},'}');
        click = str2double(b32(1));
        switch click
            case -1
                rob.mousePress(InputEvent.BUTTON1_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON1_DOWN_MASK)
                disp('left click')
            case 1
                rob.mousePress(InputEvent.BUTTON3_DOWN_MASK)
                rob.mouseRelease(InputEvent.BUTTON3_DOWN_MASK)
                disp('right click')
        end
                
        
    end
    qPhone = quaternion(orienttemp);    
    qRot = qPhone.normalize;
    eulRot(:) = quat2eul(qRot)*180/pi;
    
%     Velocity based
%     posx = posx - (eulRot(1));
%     posy = posy - (eulRot(3));
%     gr = gradient(eulRot);

%   Position based
    posx =  1920/2 - eulRot(1)*32;
    posy = 1080/2 - eulRot(3)*18;
    ppx = posx;
    ppy = posy;
    
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
