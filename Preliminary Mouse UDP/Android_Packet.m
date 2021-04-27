%% Use Android Sensors to Rotate an STL

% Loading the STL and centering it
% car = stlread('f1 senna.stl');
% car = triangulation(car.ConnectivityList, car.Points-mean(car.Points));

%% Figure Setup 
clear all
close all
%% Android Connect & Rotate
% Connect and prepare to get data from the Android phone
% target port as set in the app
port = 5555;
accel(1:3) = 0;
mag(1:3) = 0;
gyro(1:3) = 0;
linaccel(1:3) = 0;
qPhone = quaternion([1 0 0 0]);

fuse = ahrsfilter('SampleRate', 240);

tempAccel = [1 2 3];
index = 0;


%% Acceleration Plot
% figure
% plot([-100, 100000], [0 0], 'Color', [0.8 0.8 0.8])
% hold on
% t1 = plot(tempAccel(:,1));
% t2 = plot(tempAccel(:,2));
% t3 = plot(tempAccel(:,3));
% 
% t1.YDataSource = "tempAccel(:,1)";
% t2.YDataSource = "tempAccel(:,2)";
% t3.YDataSource = "tempAccel(:,3)";
% orient = [0 0 0]
% 
% i=0;
% t=0;
% tic
% while true
%     if ~ishghandle(t1)
%         break
%     end
%     [msg,~] = judp('RECEIVE',port,200);
%     a = strtrim(split(char(msg)', ','));
%     c(1:length(a)-1) = str2double(a(2:length(a)));
%     accel = c(2:4);
%     gyro = c(6:8);
%     if length(c)>9
%         mag = c(10:12);
%         if length(c)>12
%             orient = c(14:16);
%         end
%     end
%     
%     
% %     accel(3) = accel(3)-9.81;
%     
%     qPhone = fuse(accel, gyro, mag);
%     tempAccel(end+1,:) = orient;
% %     tempAccel(end+1,:) = quat2eul(qPhone);
% %     tempAccel(end+1,:) = rotatepoint(qPhone,accel);
% %     tempAccel(end+1,:) = accel;
% %     tempAccel(end+1,:) = linaccel;
% %     refreshdata
% %     drawnow
% %     disp(accel)
%     
% %     xlim([length(tempAccel)-200, length(tempAccel)])
%     t(i+1) = toc;
%     i = i+1;
% end


%% Position Plot??

figure
x = 0; y = 0; z = 0;
a = scatter3(x,y,z, 50, 'filled');

a.XDataSource = 'x';
a.YDataSource = 'y';
a.ZDataSource = 'z';

% a.CDataSource = 'qq';


i=1;
tic
while true
%     if ~ishghandle(a)
%         break
%     end
    [msg,~] = judp('RECEIVE',port,200);
    a = strtrim(split(char(msg)', ','));
    c(1:length(a)-1) = str2double(a(2:length(a)));
    accel = c(2:4);
    gyro = c(6:8)*pi/180;
    if length(c)>9
        mag = c(10:12)*10^-6;
        if length(c)>12
            linaccel = c(14:16);
        end
    end
        
    qPhone = fuse(accel, gyro, mag);
    accel = rotatepoint(qPhone, accel);
        accel(3) = accel(3)-9.81;
    time(i) = toc
    x(i+1) = x(i) + 0.5*accel(1)*dt^2;
    y(i+1) = y(i) + 0.5*accel(2)*dt^2;
    z(i+1) = z(i) + 0.5*accel(3)*dt^2;
%     qq(i+1) = length(x);
%     refreshdata
%     drawnow
    i = i+1;
end