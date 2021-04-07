qP2(1) = qPhone(1);

    eulRot22 = smoothdata(eulRot,'sgolay', 500);

for i=2:length(qPhone)
%     qP2(i) = slerp(qP2(i-1), qPhone(i), 0.5);
%     qRot(i) = qP2(i)*qBase.conj;
%     eulRot(i,:) = quat2eul(qRot(i))*180/pi;

    
%   Position Based
    posx = 1920/2 + R*tand(eulRot22(i,1));
    posy = 1080/2 + R*tand(eulRot22(i,3))/cosd(eulRot22(i,1));

    ppx(i) = posx;
    ppy(i) = posy;    
    
end

%% Plots
figure
scatter(ppx, 1080-ppy, 10, 1:length(ppx))
xlim([-10 1935])
ylim([-10 1095])
grid

figure
plot(eulRot22)
legend
grid